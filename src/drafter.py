"""Narrative Drafter & Visual Query Generator using DeepSeek-V3.

Expands chapter narrative plans into structured sentence units with:
- Rich, documentary-grade prose
- Deterministic text normalization for TTS (num2words, regnal numbers, currency)
- Concrete visual search queries optimized for stock footage and historical assets
- Dynamic camera motion directions (zoom_in, zoom_out, pan_left_to_right, pan_right_to_left)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from src.config_loader import AppSettings, load_settings
from src.planner import ChapterPlan, ProjectPlan, get_deepseek_client
from src.text_normalizer import normalize_for_tts, split_into_sentences

logger = logging.getLogger(__name__)

MotionType = Literal["zoom_in", "zoom_out", "pan_left_to_right", "pan_right_to_left"]

VALID_MOTIONS: List[MotionType] = [
    "zoom_in",
    "zoom_out",
    "pan_left_to_right",
    "pan_right_to_left",
]

MOTION_NORMALIZATION_MAP = {
    "zoom": "zoom_in",
    "zoomin": "zoom_in",
    "zoom_in": "zoom_in",
    "zoomout": "zoom_out",
    "zoom_out": "zoom_out",
    "pan_left": "pan_right_to_left",
    "pan_right": "pan_left_to_right",
    "pan_l_to_r": "pan_left_to_right",
    "pan_r_to_l": "pan_right_to_left",
    "pan_left_to_right": "pan_left_to_right",
    "pan_right_to_left": "pan_right_to_left",
    "pan_left2right": "pan_left_to_right",
    "pan_right2left": "pan_right_to_left",
}


def normalize_motion_type(val: str, index: int = 0) -> MotionType:
    """Normalizes an arbitrary camera motion string to one of the 4 supported motions."""
    if not val:
        return VALID_MOTIONS[index % len(VALID_MOTIONS)]
    cleaned = val.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned in MOTION_NORMALIZATION_MAP:
        return MOTION_NORMALIZATION_MAP[cleaned]
    # Fallback to cycling supported motions
    return VALID_MOTIONS[index % len(VALID_MOTIONS)]


def sanitize_visual_query(query: str) -> str:
    """Sanitizes visual search query, stripping quotes, punctuation, and filler words."""
    query = re.sub(r'["\'\.,!?;:]', "", query)
    query = re.sub(r"\s+", " ", query).strip().lower()
    return query


class ScriptSentence(BaseModel):
    """Data contract for an individual narrative beat sentence."""

    sentence_index: int = Field(..., description="0-based sequential sentence index")
    raw_text: str = Field(..., description="Original rich narrative sentence")
    spoken_text: str = Field(
        ..., description="Phonetically and numerically normalized spoken text"
    )
    visual_query: str = Field(
        ..., description="Concrete b-roll / image search query (3-6 words)"
    )
    motion_type: MotionType = Field(
        default="zoom_in", description="Designated camera motion direction"
    )
    pause_after_seconds: float = Field(
        default=0.4, description="Silence pause after sentence in seconds"
    )

    @field_validator("sentence_index")
    @classmethod
    def validate_sentence_index(cls, v: int) -> int:
        if v < 0:
            raise ValueError("sentence_index must be >= 0")
        return v

    @field_validator("pause_after_seconds")
    @classmethod
    def validate_pause(cls, v: float) -> float:
        # Clamp pause between 0.1 and 2.0 seconds
        return max(0.1, min(2.0, float(v)))


class ChapterScript(BaseModel):
    """Data contract for a complete chapter script (Stage 2 output)."""

    chapter_index: int = Field(..., description="1-based index matching ChapterPlan")
    chapter_title: str = Field(..., description="Chapter title")
    sentences: List[ScriptSentence] = Field(..., description="List of script sentences")

    @field_validator("sentences")
    @classmethod
    def validate_sentences(cls, v: List[ScriptSentence]) -> List[ScriptSentence]:
        if not v:
            raise ValueError("ChapterScript must contain at least one sentence")
        return v


DRAFTER_SYSTEM_PROMPT = """You are a master historical documentary scriptwriter and visual director.
Your role is to expand chapter narrative beats into rich, immersive, documentary-grade narration.

Narrative Guidelines:
- High-register, engaging, rhythmically varied prose (avoid modern conversational clichés).
- Craft self-contained, powerful sentences that build tension and curiosity.
- Each sentence must be paired with:
  1. A concrete, descriptive visual search query (3 to 6 words) optimized for stock video (Pexels/Pixabay) or historical archives. Focus on physical subjects, settings, actions, or historical artifacts (e.g., 'alexandria harbor ancient lighthouse ships', 'papyrus scroll library reading scholars'). Avoid abstract metaphors (e.g., 'symbol of knowledge' is bad).
  2. A camera motion direction chosen from: 'zoom_in', 'zoom_out', 'pan_left_to_right', 'pan_right_to_left'. Vary motions dynamically across consecutive sentences.
  3. A pause duration (typically 0.4 to 0.6 seconds) after each sentence.

Strict Requirement:
Return ONLY a valid JSON object matching the requested schema. Do NOT include markdown code blocks, explanatory prose, or commentary.
"""

DRAFTER_USER_PROMPT_TEMPLATE = """Write the complete documentary narration script for Chapter {chapter_index}: "{chapter_title}".

Context & Constraints:
- Overall Visual Style Anchor: {visual_style_anchor}
- Visual Palette: {visual_palette}
- Voice Tone: {voice_tone}
- Core Tension: {core_tension}
- Target Word Count: ~{target_word_count} words
- Target Duration: ~{target_duration_seconds} seconds
- Narrative Beats to cover:
{narrative_beats}
{prior_context_block}

Return a JSON object with this exact structure:
{{
  "chapter_index": {chapter_index},
  "chapter_title": "{chapter_title}",
  "sentences": [
    {{
      "sentence_index": 0,
      "raw_text": "<rich narrative sentence>",
      "visual_query": "<concrete 3-6 word visual search query>",
      "motion_type": "zoom_in",
      "pause_after_seconds": 0.4
    }}
  ]
}}
"""


def _clean_json_response(raw_text: str) -> str:
    """Strips markdown code fences and whitespace from LLM output."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    return raw_text.strip()


def draft_mock_chapter_script(
    chapter: ChapterPlan,
    visual_style: Optional[str] = None,
) -> ChapterScript:
    """Generates synthetic chapter script sentences for testing and offline execution."""
    sentences: List[ScriptSentence] = []
    motions: List[MotionType] = ["zoom_in", "pan_left_to_right", "zoom_out", "pan_right_to_left"]
    palette_queries = chapter.visual_palette or ["historical scene", "archival texture"]

    for idx, beat in enumerate(chapter.narrative_beats):
        raw_text = f"In {chapter.title}, {beat.rstrip('.')}."
        spoken = normalize_for_tts(raw_text)
        query = f"{palette_queries[idx % len(palette_queries)]} {chapter.title}".strip()
        motion = motions[idx % len(motions)]
        sentences.append(
            ScriptSentence(
                sentence_index=idx,
                raw_text=raw_text,
                spoken_text=spoken,
                visual_query=sanitize_visual_query(query),
                motion_type=motion,
                pause_after_seconds=0.4,
            )
        )

    # Ensure at least 2 sentences
    if len(sentences) < 2:
        sentences.append(
            ScriptSentence(
                sentence_index=len(sentences),
                raw_text=f"The story of {chapter.title} continues to echo through history.",
                spoken_text=f"The story of {chapter.title} continues to echo through history.",
                visual_query=f"{chapter.title} historical archive",
                motion_type="zoom_out",
                pause_after_seconds=0.5,
            )
        )

    return ChapterScript(
        chapter_index=chapter.chapter_index,
        chapter_title=chapter.title,
        sentences=sentences,
    )


def draft_chapter_script(
    chapter: ChapterPlan,
    visual_style: str,
    prior_summary: str = "",
    settings: Optional[AppSettings] = None,
    client: Optional[OpenAI] = None,
    prior_chapter_summary: Optional[str] = None,
    mock: bool = False,
) -> ChapterScript:
    """Drafts narration and visual queries for a single chapter.

    Args:
        chapter: The ChapterPlan containing narrative beats and targets.
        visual_style: Master visual style anchor string.
        prior_summary: Optional summary of prior chapters for narrative continuity.
        settings: Application settings.
        client: Pre-configured OpenAI/DeepSeek client.
        prior_chapter_summary: Alias for prior_summary.
        mock: If True, generates synthetic mock sentences without API calls.

    Returns:
        Validated ChapterScript with deterministic text normalization applied.
    """
    if prior_chapter_summary is not None:
        prior_summary = prior_chapter_summary
    if mock:
        return draft_mock_chapter_script(chapter=chapter, visual_style=visual_style)
    settings = settings or load_settings()

    beats_formatted = "\n".join(f"- {b}" for b in chapter.narrative_beats)
    prior_context_block = (
        f"- Prior Chapter Summary (for continuity):\n{prior_summary}"
        if prior_summary
        else ""
    )

    user_prompt = DRAFTER_USER_PROMPT_TEMPLATE.format(
        chapter_index=chapter.chapter_index,
        chapter_title=chapter.title,
        visual_style_anchor=visual_style,
        visual_palette=", ".join(chapter.visual_palette),
        voice_tone=chapter.voice_tone,
        core_tension=chapter.core_tension,
        target_word_count=chapter.target_word_count,
        target_duration_seconds=chapter.target_duration_seconds,
        narrative_beats=beats_formatted,
        prior_context_block=prior_context_block,
    )

    if client is None:
        client = get_deepseek_client(settings)

    logger.info(
        f"Calling DeepSeek API to draft Chapter {chapter.chapter_index}: '{chapter.title}'..."
    )

    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": DRAFTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    content = response.choices[0].message.content or ""
    cleaned_json = _clean_json_response(content)

    try:
        raw_dict = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode DeepSeek drafter JSON: {content}")
        raise ValueError(f"DeepSeek returned invalid JSON for drafter: {e}") from e

    raw_sentences = raw_dict.get("sentences", [])
    if not isinstance(raw_sentences, list) or len(raw_sentences) == 0:
        raise ValueError(f"Drafter output contains no sentences: {cleaned_json}")

    processed_sentences: List[ScriptSentence] = []
    for idx, item in enumerate(raw_sentences):
        if not isinstance(item, dict):
            continue

        raw_text = item.get("raw_text", "").strip()
        if not raw_text:
            continue

        # Deterministic phonetic and numerical normalization pass
        spoken_text = normalize_for_tts(raw_text)

        # Sanitize visual query
        visual_q = sanitize_visual_query(item.get("visual_query", ""))
        if not visual_q:
            # Fallback visual query from visual palette or chapter title
            visual_q = (
                sanitize_visual_query(chapter.visual_palette[0])
                if chapter.visual_palette
                else sanitize_visual_query(chapter.title)
            )

        # Normalize camera motion
        motion = normalize_motion_type(item.get("motion_type", ""), index=idx)

        # Pause after seconds
        pause = float(item.get("pause_after_seconds", settings.tts_pause_seconds))

        sentence_obj = ScriptSentence(
            sentence_index=len(processed_sentences),
            raw_text=raw_text,
            spoken_text=spoken_text,
            visual_query=visual_q,
            motion_type=motion,
            pause_after_seconds=pause,
        )
        processed_sentences.append(sentence_obj)

    if not processed_sentences:
        raise ValueError(
            f"No valid sentences could be extracted for Chapter {chapter.chapter_index}"
        )

    script = ChapterScript(
        chapter_index=chapter.chapter_index,
        chapter_title=chapter.title,
        sentences=processed_sentences,
    )

    logger.info(
        f"Drafted Chapter {script.chapter_index} with {len(script.sentences)} sentences."
    )
    return script


def draft_all_chapters(
    plan: ProjectPlan,
    settings: Optional[AppSettings] = None,
    client: Optional[OpenAI] = None,
) -> List[ChapterScript]:
    """Sequentially drafts all chapters in a ProjectPlan, carrying narrative context forward.

    Args:
        plan: The ProjectPlan object.
        settings: Application settings.
        client: Pre-configured OpenAI/DeepSeek client.

    Returns:
        List of ChapterScript objects in sequential order.
    """
    settings = settings or load_settings()
    scripts: List[ChapterScript] = []
    running_summaries: List[str] = []

    for chapter in plan.chapters:
        prior_context = " ".join(running_summaries[-2:]) if running_summaries else ""
        script = draft_chapter_script(
            chapter=chapter,
            visual_style=plan.visual_style_anchor,
            prior_summary=prior_context,
            settings=settings,
            client=client,
        )
        scripts.append(script)

        # Record brief chapter takeaway for context continuity
        first_sentence = script.sentences[0].raw_text if script.sentences else ""
        last_sentence = script.sentences[-1].raw_text if script.sentences else ""
        running_summaries.append(
            f"Chapter {chapter.chapter_index} ({chapter.title}): {first_sentence} ... {last_sentence}"
        )

    return scripts


def save_chapter_script(script: ChapterScript, output_path: Path | str) -> Path:
    """Saves ChapterScript to JSON file with indentation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(script.model_dump_json(indent=2))
    logger.info(f"Saved chapter script to {path}")
    return path


def load_chapter_script(input_path: Path | str) -> ChapterScript:
    """Loads and validates ChapterScript from a JSON file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Chapter script file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ChapterScript.model_validate(data)
