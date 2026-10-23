"""FFmpeg Video Engine with Dynamic Ken Burns, Audio Ducking & Lossless Concatenation.

Features:
- Dynamic 4-motion Ken Burns camera animations:
    - Zoom In (1.0x -> 1.15x toward center)
    - Zoom Out (1.15x -> 1.0x from center)
    - Pan Left to Right (horizontal slide across 16:9 frame)
    - Pan Right to Left (reverse horizontal slide)
- Safe 4K intermediate upscaling filtergraph eliminating memory leak risks
- Voice-activated sidechain ducking (BGM drops ~14dB during voice activity)
- Standardized intermediate rendering (Fixed 30fps CFR, 1920x1080, yuv420p)
- Automatic hardware encoding detection (h264_nvenc with libx264 fallback)
- Chapter-level rendering and lossless master concatenation via concat demuxer
"""

import logging
import math
import os
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from src.aligner import AlignedSegment, ChapterAlignment
from src.drafter import MotionType

logger = logging.getLogger(__name__)

# Cached encoder name
_DETECTED_ENCODER: Optional[str] = None


def get_video_encoder() -> str:
    """Detects available video encoder, prioritizing h264_nvenc over libx264."""
    global _DETECTED_ENCODER
    if _DETECTED_ENCODER is not None:
        return _DETECTED_ENCODER

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.1:size=64x64:rate=30",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("NVIDIA NVENC hardware encoder detected (h264_nvenc).")
            _DETECTED_ENCODER = "h264_nvenc"
            return "h264_nvenc"
    except Exception as exc:
        logger.debug("NVENC probe check failed: %s", exc)

    logger.info("Falling back to CPU libx264 encoder.")
    _DETECTED_ENCODER = "libx264"
    return "libx264"


def build_ken_burns_filter(
    motion_type: MotionType,
    duration: float,
    fps: int = 30,
    target_res: Tuple[int, int] = (1920, 1080),
) -> str:
    """Builds rock-solid mathematical FFmpeg crop/scale filter for Ken Burns animations.

    Upscales image to 4K (3840x2160) to eliminate aliasing and blur, applies continuous
    subpixel crop expressions, and scales to standard 1920x1080.
    """
    total_frames = max(1, int(round(duration * fps)))
    # 4K canvas: 3840 x 2160.
    # Scaled 1.15x zoom crop box: 3339 x 1878 (delta_w = 501, delta_h = 282).
    # Pan delta: 3840 - 3339 = 501.

    base_scale = "scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160"

    if motion_type == "zoom_in":
        # Crop window shrinks from (3840, 2160) down to (3339, 1878) centered
        motion_crop = (
            f"crop=w=3840-(501*n/{total_frames}):"
            f"h=2160-(282*n/{total_frames}):"
            "x=(3840-out_w)/2:y=(2160-out_h)/2"
        )
    elif motion_type == "zoom_out":
        # Crop window expands from (3339, 1878) up to (3840, 2160) centered
        motion_crop = (
            f"crop=w=3339+(501*n/{total_frames}):"
            f"h=1878+(282*n/{total_frames}):"
            "x=(3840-out_w)/2:y=(2160-out_h)/2"
        )
    elif motion_type == "pan_left_to_right":
        # Crop window (3339, 1878) slides from x=0 to x=501
        motion_crop = (
            "crop=w=3339:h=1878:"
            f"x=(501*n/{total_frames}):"
            "y=(2160-1878)/2"
        )
    elif motion_type == "pan_right_to_left":
        # Crop window (3339, 1878) slides from x=501 to x=0
        motion_crop = (
            "crop=w=3339:h=1878:"
            f"x=501-(501*n/{total_frames}):"
            "y=(2160-1878)/2"
        )
    else:
        # Default smooth center zoom in
        motion_crop = (
            f"crop=w=3840-(501*n/{total_frames}):"
            f"h=2160-(282*n/{total_frames}):"
            "x=(3840-out_w)/2:y=(2160-out_h)/2"
        )

    out_scale = f"scale={target_res[0]}:{target_res[1]},setsar=1"
    return f"{base_scale},{motion_crop},{out_scale}"


def generate_ambient_bgm(output_path: Path, duration_seconds: float, sr: int = 48000) -> Path:
    """Synthesizes rich, gentle ambient background drone score if no music files exist."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(sr * max(5.0, duration_seconds + 3.0))
    t = np.linspace(0, samples / sr, samples, endpoint=False, dtype=np.float32)

    # Ambient deep drone in D minor (D2: 73.42Hz, A2: 110Hz, F3: 174.61Hz)
    ch1 = (
        0.30 * np.sin(2 * np.pi * 73.42 * t)
        + 0.20 * np.sin(2 * np.pi * 110.0 * t + 0.5)
        + 0.15 * np.sin(2 * np.pi * 174.61 * t + 1.0)
    )
    # Slow LFO breathing modulation
    lfo = 0.5 * (1.0 + 0.3 * np.sin(2 * np.pi * 0.08 * t))
    bgm_audio = (ch1 * lfo).astype(np.float32)

    # Gentle stereo panning
    left = bgm_audio * 0.85
    right = bgm_audio * 0.90
    stereo = np.column_stack([left, right])

    # Fade in / out
    fade_len = int(sr * 2.0)
    if len(stereo) > 2 * fade_len:
        f_in = np.linspace(0, 1, fade_len, dtype=np.float32)[:, np.newaxis]
        f_out = np.linspace(1, 0, fade_len, dtype=np.float32)[:, np.newaxis]
        stereo[:fade_len] *= f_in
        stereo[-fade_len:] *= f_out

    sf.write(str(output_path), stereo, sr, subtype="PCM_16")
    return output_path


def get_or_create_bgm(
    bgm_dir: Optional[Path], output_bgm_path: Path, required_duration: float
) -> Path:
    """Finds ambient music file in bgm_dir or generates procedural ambient score."""
    if bgm_dir and bgm_dir.exists():
        for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
            found = list(bgm_dir.glob(f"*{ext}"))
            if found:
                logger.info("Found curated background music: %s", found[0])
                return found[0]

    # Generate procedural ambient score
    logger.info("No curated BGM found; generating procedural ambient score.")
    return generate_ambient_bgm(output_bgm_path, required_duration)


def render_segment_video(
    segment: AlignedSegment,
    output_segment_path: Path,
    fps: int = 30,
    target_res: Tuple[int, int] = (1920, 1080),
) -> Path:
    """Renders an individual segment asset into standardized 30fps CFR 1080p MP4."""
    output_segment_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = get_video_encoder()

    total_duration = max(0.5, segment.duration + segment.pause_after)
    asset_file = Path(segment.asset_path)

    if not asset_file.exists():
        raise FileNotFoundError(f"Segment asset file does not exist: {asset_file}")

    is_video = segment.asset_type == "video" or asset_file.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm")

    if is_video:
        # Scale & center crop video, looping if shorter than segment duration
        vf = f"scale={target_res[0]}:{target_res[1]}:force_original_aspect_ratio=increase,crop={target_res[0]}:{target_res[1]},setsar=1"
        cmd = [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(asset_file),
            "-vf",
            vf,
            "-t",
            f"{total_duration:.3f}",
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-an",  # Strip source video audio
            "-c:v",
            encoder,
        ]
        if encoder == "libx264":
            cmd.extend(["-preset", "fast", "-crf", "20"])
        cmd.append(str(output_segment_path))
    else:
        # Image with Ken Burns camera motion
        kb_filter = build_ken_burns_filter(
            motion_type=segment.motion_type,
            duration=total_duration,
            fps=fps,
            target_res=target_res,
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(asset_file),
            "-vf",
            kb_filter,
            "-t",
            f"{total_duration:.3f}",
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            encoder,
        ]
        if encoder == "libx264":
            cmd.extend(["-preset", "fast", "-crf", "20"])
        cmd.append(str(output_segment_path))

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.warning(
            "Video encoding failed with %s (%s). Retrying with libx264...",
            encoder,
            res.stderr[-300:],
        )
        # Fallback to libx264 if hardware encoder failed
        cmd_fallback = cmd.copy()
        idx = cmd_fallback.index(encoder)
        cmd_fallback[idx] = "libx264"
        cmd_fallback[idx + 1 : idx + 1] = ["-preset", "ultrafast", "-crf", "23"]
        sub_res = subprocess.run(cmd_fallback, capture_output=True, text=True)
        if sub_res.returncode != 0:
            raise RuntimeError(f"FFmpeg render failed for segment {segment.sentence_index}: {sub_res.stderr[-400:]}")

    return output_segment_path


def mix_audio_ducking(
    narration_path: Path,
    bgm_path: Path,
    output_audio_path: Path,
    duck_threshold: float = 0.08,
    duck_ratio: float = 6.0,
    bgm_volume: float = 0.35,
) -> Path:
    """Mixes mastered narration with background music using sidechain compression."""
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Sidechain compressor filtergraph
    # [1:a] (BGM) volume adjusted, fed through sidechaincompress keyed to [0:a] (voice)
    # amix combines ducked BGM with voice
    duck_filter = (
        f"[1:a]volume={bgm_volume}[bgm];"
        f"[bgm][0:a]sidechaincompress=threshold={duck_threshold}:"
        f"ratio={duck_ratio}:attack=30:release=400:level_in=1[ducked_bgm];"
        "[0:a][ducked_bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(narration_path),
        "-stream_loop",
        "-1",
        "-i",
        str(bgm_path),
        "-filter_complex",
        duck_filter,
        "-map",
        "[aout]",
        "-c:a",
        "pcm_s16le",
        str(output_audio_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.warning("Sidechain ducking failed (%s). Using narration audio directly.", res.stderr[-300:])
        import shutil
        shutil.copyfile(narration_path, output_audio_path)

    return output_audio_path


class VideoEngine:
    """Master video rendering and assembly engine."""

    def __init__(self, fps: int = 30, target_resolution: Tuple[int, int] = (1920, 1080)):
        self.fps = fps
        self.target_resolution = target_resolution

    def render_chapter(
        self,
        alignment: ChapterAlignment,
        output_chapter_mp4: Path,
        bgm_dir: Optional[Path] = None,
        srt_path: Optional[Path] = None,
        burn_subtitles: bool = False,
    ) -> Path:
        """Renders complete chapter video with Ken Burns, sidechain ducking, and subtitles."""
        output_chapter_mp4.parent.mkdir(parents=True, exist_ok=True)
        ch_render_dir = output_chapter_mp4.parent / "render_temp"
        segments_dir = ch_render_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Rendering Chapter %d (%d segments) -> %s",
            alignment.chapter_index,
            len(alignment.segments),
            output_chapter_mp4,
        )

        segment_paths: List[Path] = []
        for seg in alignment.segments:
            seg_out = segments_dir / f"seg_{seg.sentence_index:03d}.mp4"
            render_segment_video(
                segment=seg,
                output_segment_path=seg_out,
                fps=self.fps,
                target_res=self.target_resolution,
            )
            segment_paths.append(seg_out)

        # Create concat manifest for segments
        concat_txt = ch_render_dir / "segments_concat.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in segment_paths:
                # Use forward slashes for FFmpeg compatibility on Windows
                safe_path = str(p.resolve()).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        # Concat video track
        video_track = ch_render_dir / "video_track.mp4"
        cmd_concat = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-c",
            "copy",
            str(video_track),
        ]
        res_v = subprocess.run(cmd_concat, capture_output=True, text=True)
        if res_v.returncode != 0:
            raise RuntimeError(f"FFmpeg segment concatenation failed: {res_v.stderr[-400:]}")

        # Mix Audio with Sidechain Ducking
        narration_audio = Path(alignment.audio_file)
        bgm_file = ch_render_dir / "ambient_bgm.wav"
        bgm_path = get_or_create_bgm(
            bgm_dir=bgm_dir,
            output_bgm_path=bgm_file,
            required_duration=alignment.total_duration,
        )

        ducked_audio = ch_render_dir / "audio_ducked.wav"
        mix_audio_ducking(
            narration_path=narration_audio,
            bgm_path=bgm_path,
            output_audio_path=ducked_audio,
        )

        # Final muxing: Combine video track + ducked audio + subtitles
        cmd_final = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_track),
            "-i",
            str(ducked_audio),
        ]

        if srt_path and srt_path.exists() and not burn_subtitles:
            # Embed soft subtitle track
            cmd_final.extend([
                "-i",
                str(srt_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map",
                "2:s:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-c:s",
                "mov_text",
            ])
        else:
            cmd_final.extend([
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ])

        cmd_final.append(str(output_chapter_mp4))
        res_final = subprocess.run(cmd_final, capture_output=True, text=True)
        if res_final.returncode != 0:
            raise RuntimeError(f"Final chapter muxing failed: {res_final.stderr[-400:]}")

        logger.info("Chapter %d rendered successfully -> %s", alignment.chapter_index, output_chapter_mp4)
        return output_chapter_mp4

    def concatenate_chapters(
        self,
        chapter_mp4_paths: List[Path],
        output_master_mp4: Path,
    ) -> Path:
        """Losslessly concatenates all rendered chapter MP4s into final master video."""
        output_master_mp4.parent.mkdir(parents=True, exist_ok=True)
        concat_txt = output_master_mp4.parent / "master_chapters_concat.txt"

        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in chapter_mp4_paths:
                safe_path = str(p.resolve()).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-c",
            "copy",
            str(output_master_mp4),
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Master chapter concatenation failed: {res.stderr[-400:]}")

        logger.info(
            "Master video concatenated losslessly from %d chapters -> %s",
            len(chapter_mp4_paths),
            output_master_mp4,
        )
        return output_master_mp4
