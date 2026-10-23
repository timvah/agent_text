"""Autonomous Long-Narrative AI Video Generation Pipeline Orchestrator.

Master CLI entry point with granular checkpoint resume, dry-run mode, and stage isolation:
- Stage 1: Macro-Planner & Narrative Architect (DeepSeek-V3)
- Stage 2: Narrative Drafter & Rule Normalizer (DeepSeek-V3 + text_normalizer)
- Stage 3: Voice Engine & Audio Mastering (Qwen-TTS / Fallback + EBU R128)
- Stage 4: Deterministic Alignment & Subtitle Generator (Zero-Drift Math)
- Stage 5: Resilient Multi-Source Asset Harvester (Pexels / Pixabay / Commons / AI)
- Stage 6: Video Assembly, Dynamic Ken Burns & Audio Ducking (FFmpeg)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path when executed directly
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.aligner import (
    ChapterAlignment,
    align_chapter,
    generate_srt,
    save_alignment,
)
from src.asset_harvester import (
    AssetHarvester,
    ChapterAssetManifest,
    HarvestedAsset,
)
from src.config_loader import AppSettings, load_settings
from src.drafter import ChapterScript, draft_chapter_script
from src.planner import ChapterPlan, ProjectPlan, generate_project_plan
from src.utils import (
    create_chapter_scaffolding,
    create_project_scaffolding,
    sanitize_filename,
)
from src.video_engine import VideoEngine
from src.voice_engine import ChapterAudioManifest, VoiceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")

STAGES = ["plan", "script", "voice", "assets", "align", "render"]


class VideoPipeline:
    """Master orchestrator executing stages with granular resume checkpoints."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        mock_tts: bool = False,
        mock_llm: bool = False,
        allow_network_assets: bool = True,
        burn_subtitles: bool = False,
    ):
        self.settings = settings or load_settings()
        self.mock_tts = mock_tts
        self.mock_llm = mock_llm
        self.allow_network_assets = allow_network_assets
        self.burn_subtitles = burn_subtitles

        self.voice_engine = VoiceEngine(settings=self.settings, mock=mock_tts)
        self.asset_harvester = AssetHarvester(settings=self.settings)
        self.video_engine = VideoEngine(
            fps=self.settings.default_fps,
            target_resolution=self.settings.default_resolution,
        )

    def run(
        self,
        topic: str,
        duration_minutes: int = 8,
        tone: str = "documentary, deliberate",
        visual_style: Optional[str] = None,
        project_id: Optional[str] = None,
        resume_from: Optional[str] = None,
        dry_run: bool = False,
        bgm_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Runs the complete narrative video generation pipeline.

        Args:
            topic: Premise or documentary subject.
            duration_minutes: Desired duration in minutes.
            tone: Narrative tone direction.
            visual_style: Visual aesthetic override.
            project_id: Explicit project identifier.
            resume_from: Checkpoint stage to resume from ('plan', 'script', 'voice', etc.).
            dry_run: If True, generates plan and script without audio or video render.
            bgm_dir: Custom path to background music folder.

        Returns:
            Dictionary with paths to generated project artifacts.
        """
        proj_id = project_id or sanitize_filename(topic)
        scaffolding = create_project_scaffolding(proj_id, base_dir=self.settings.output_dir)
        project_root = scaffolding["project_root"]
        meta_dir = scaffolding["meta_dir"]
        chapters_dir = scaffolding["chapters_dir"]
        renders_dir = scaffolding["renders_dir"]

        plan_file = meta_dir / "project_plan.json"
        resume_stage = resume_from.lower() if resume_from else "plan"

        logger.info("=" * 65)
        logger.info("Starting Video Pipeline: '%s' (%dm)", topic, duration_minutes)
        logger.info("Project Root: %s", project_root)
        logger.info("Resume Stage: %s | Dry Run: %s", resume_stage, dry_run)
        logger.info("=" * 65)

        # -------------------------------------------------------------
        # STAGE 1: Macro-Planner & Narrative Architect
        # -------------------------------------------------------------
        logger.info("[Stage 1/6] Macro-Planning Narrative Beats...")
        if plan_file.exists() and resume_stage != "plan":
            logger.info("Resuming: Loading existing project_plan.json")
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
            project_plan = ProjectPlan(**plan_data)
        else:
            project_plan = generate_project_plan(
                topic=topic,
                duration_minutes=duration_minutes,
                tone=tone,
                visual_style=visual_style,
                settings=self.settings,
                mock=self.mock_llm,
            )
            if visual_style:
                project_plan.visual_style_anchor = visual_style

            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(project_plan.model_dump(), f, indent=2)
            logger.info("Project plan generated with %d chapters.", len(project_plan.chapters))

        # -------------------------------------------------------------
        # STAGE 2: Chapter Script Drafting & Text Normalization
        # -------------------------------------------------------------
        logger.info("[Stage 2/6] Drafting Chapter Narration & Visual Queries...")
        chapter_scripts: List[ChapterScript] = []
        prior_summary = ""

        for ch in project_plan.chapters:
            ch_scaffolding = create_chapter_scaffolding(ch.chapter_index, chapters_dir)
            script_path = ch_scaffolding["chapter_dir"] / "chapter_script.json"

            should_reuse_script = (
                script_path.exists()
                and resume_stage in ("voice", "assets", "align", "render")
            )

            if should_reuse_script:
                logger.info("Resuming: Loading existing script for Chapter %d", ch.chapter_index)
                with open(script_path, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                script = ChapterScript(**s_data)
            else:
                logger.info("Drafting Chapter %d: '%s'...", ch.chapter_index, ch.title)
                script = draft_chapter_script(
                    chapter=ch,
                    visual_style=project_plan.visual_style_anchor,
                    prior_chapter_summary=prior_summary,
                    settings=self.settings,
                    mock=self.mock_llm,
                )
                with open(script_path, "w", encoding="utf-8") as f:
                    json.dump(script.model_dump(), f, indent=2)

            chapter_scripts.append(script)
            prior_summary = f"{ch.title}: {ch.core_tension}"

        if dry_run:
            logger.info("Dry run requested. Halting pipeline after Script Drafting.")
            return {
                "project_root": project_root,
                "project_plan": plan_file,
            }

        # -------------------------------------------------------------
        # STAGE 3: Voice Engine & Audio Mastering
        # -------------------------------------------------------------
        logger.info("[Stage 3/6] Synthesizing & Mastering Audio (Voice Engine)...")
        audio_manifests: List[ChapterAudioManifest] = []

        for script in chapter_scripts:
            ch_scaffolding = create_chapter_scaffolding(script.chapter_index, chapters_dir)
            audio_dir = ch_scaffolding["audio_dir"]
            durations_json = audio_dir / "clip_durations.json"
            mastered_wav = audio_dir / "audio_mastered.wav"

            should_reuse_audio = (
                durations_json.exists()
                and mastered_wav.exists()
                and resume_stage in ("assets", "align", "render")
            )

            if should_reuse_audio:
                logger.info("Resuming: Using existing mastered audio for Chapter %d", script.chapter_index)
                with open(durations_json, "r", encoding="utf-8") as f:
                    a_data = json.load(f)
                audio_manifest = ChapterAudioManifest(**a_data)
            else:
                audio_manifest = self.voice_engine.synthesize_chapter(
                    chapter_script=script,
                    output_dir=audio_dir,
                )
            audio_manifests.append(audio_manifest)

        # -------------------------------------------------------------
        # STAGE 4: Multi-Source Asset Harvesting
        # -------------------------------------------------------------
        logger.info("[Stage 4/6] Harvesting Visual Assets (Pexels/Pixabay/Commons/AI)...")
        asset_manifests: List[ChapterAssetManifest] = []

        for script in chapter_scripts:
            ch_scaffolding = create_chapter_scaffolding(script.chapter_index, chapters_dir)
            assets_dir = ch_scaffolding["assets_dir"]
            manifest_json = assets_dir / "assets_manifest.json"

            should_reuse_assets = (
                manifest_json.exists()
                and resume_stage in ("align", "render")
            )

            if should_reuse_assets:
                logger.info("Resuming: Using existing assets for Chapter %d", script.chapter_index)
                with open(manifest_json, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                asset_manifest = ChapterAssetManifest(**m_data)
            else:
                asset_manifest = self.asset_harvester.harvest_chapter(
                    project_id=proj_id,
                    chapter_script=script,
                    output_dir=assets_dir,
                    visual_style=project_plan.visual_style_anchor,
                    allow_network=self.allow_network_assets,
                )
            asset_manifests.append(asset_manifest)

        # -------------------------------------------------------------
        # STAGE 5: Deterministic Mathematical Alignment & SRT Generation
        # -------------------------------------------------------------
        logger.info("[Stage 5/6] Calculating Zero-Drift Timeline & Generating Subtitles...")
        chapter_alignments: List[ChapterAlignment] = []

        for script, audio_man, asset_man in zip(chapter_scripts, audio_manifests, asset_manifests):
            ch_scaffolding = create_chapter_scaffolding(script.chapter_index, chapters_dir)
            ch_dir = ch_scaffolding["chapter_dir"]
            alignment_json = ch_dir / "alignment.json"
            srt_path = ch_dir / f"chapter_{script.chapter_index:02d}.srt"

            # Create asset lookup map
            asset_map = {
                a.sentence_index: {
                    "asset_path": a.asset_path,
                    "asset_type": a.asset_type,
                }
                for a in asset_man.assets
            }

            alignment = align_chapter(
                project_id=proj_id,
                chapter_script=script,
                audio_manifest=audio_man,
                asset_map=asset_map,
            )
            save_alignment(alignment, alignment_json)
            generate_srt(alignment, srt_path)
            chapter_alignments.append(alignment)

        # -------------------------------------------------------------
        # STAGE 6: Video Assembly, Ken Burns, Ducking & Lossless Concat
        # -------------------------------------------------------------
        logger.info("[Stage 6/6] Assembling Chapter Videos & Final Master Render...")
        chapter_mp4s: List[Path] = []
        bgm_search_dir = bgm_dir or Path("assets/bgm")

        for align in chapter_alignments:
            ch_scaffolding = create_chapter_scaffolding(align.chapter_index, chapters_dir)
            render_dir = ch_scaffolding["render_dir"]
            ch_mp4 = render_dir / f"chapter_{align.chapter_index:02d}.mp4"
            ch_srt = ch_scaffolding["chapter_dir"] / f"chapter_{align.chapter_index:02d}.srt"

            if ch_mp4.exists() and resume_stage == "render":
                logger.info("Resuming: Existing render found for Chapter %d", align.chapter_index)
            else:
                self.video_engine.render_chapter(
                    alignment=align,
                    output_chapter_mp4=ch_mp4,
                    bgm_dir=bgm_search_dir,
                    srt_path=ch_srt,
                    burn_subtitles=self.burn_subtitles,
                )
            chapter_mp4s.append(ch_mp4)

        # Final Master Concatenation
        master_mp4 = renders_dir / f"{proj_id}_master.mp4"
        self.video_engine.concatenate_chapters(
            chapter_mp4_paths=chapter_mp4s,
            output_master_mp4=master_mp4,
        )

        logger.info("=" * 65)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("Master Video: %s", master_mp4)
        logger.info("=" * 65)

        return {
            "project_root": project_root,
            "project_plan": plan_file,
            "master_video": master_mp4,
        }


def main():
    """CLI Entry point for master orchestrator."""
    parser = argparse.ArgumentParser(
        description="Autonomous Long-Narrative AI Video Generation Engine"
    )
    parser.add_argument("--topic", type=str, required=True, help="Documentary or narrative topic")
    parser.add_argument(
        "--duration", type=int, default=8, help="Target video duration in minutes (default: 8)"
    )
    parser.add_argument(
        "--tone", type=str, default="documentary, deliberate", help="Narrative vocal tone"
    )
    parser.add_argument(
        "--visual-style", type=str, default=None, help="Visual style aesthetic anchor override"
    )
    parser.add_argument(
        "--project-id", type=str, default=None, help="Custom identifier slug for project"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        choices=STAGES,
        default=None,
        help="Resume pipeline execution from specified stage checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate project plan and scripts only, without running audio/video renders",
    )
    parser.add_argument(
        "--mock-tts",
        action="store_true",
        help="Use synthetic speech waveform instead of live Gradio TTS server",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use synthetic planning and drafting without calling external DeepSeek API",
    )
    parser.add_argument(
        "--offline-assets",
        action="store_true",
        help="Use offline procedural and cached assets without querying external stock APIs",
    )
    parser.add_argument(
        "--burn-subtitles",
        action="store_true",
        help="Burn subtitles directly into video frames instead of soft subtitle tracks",
    )
    parser.add_argument(
        "--bgm-dir",
        type=Path,
        default=Path("assets/bgm"),
        help="Directory containing royalty-free background music tracks",
    )

    args = parser.parse_args()

    pipeline = VideoPipeline(
        mock_tts=args.mock_tts,
        mock_llm=args.mock_llm,
        allow_network_assets=not args.offline_assets,
        burn_subtitles=args.burn_subtitles,
    )

    try:
        pipeline.run(
            topic=args.topic,
            duration_minutes=args.duration,
            tone=args.tone,
            visual_style=args.visual_style,
            project_id=args.project_id,
            resume_from=args.resume_from,
            dry_run=args.dry_run,
            bgm_dir=args.bgm_dir,
        )
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
