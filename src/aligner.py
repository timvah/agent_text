"""Deterministic Mathematical Alignment & Subtitle Generator (Zero-Drift Math).

Provides:
- Exact mathematical timeline calculation from synthesized audio durations:
    Start_0 = 0.0
    End_i   = Start_i + Duration_i
    Start_i+1 = End_i + Pause_i
- Zero risk of speech-to-text hallucination on ancient or specialized proper nouns
- Standard YouTube-ready .srt subtitle generation with balanced 2-line formatting
- Alignment metadata export linking sentences, audio timestamps, and visual assets
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from src.drafter import ChapterScript, MotionType
from src.voice_engine import ChapterAudioManifest

logger = logging.getLogger(__name__)


class AlignedSegment(BaseModel):
    """Data contract for an individual aligned timeline segment."""

    sentence_index: int = Field(..., description="0-based sentence index")
    text: str = Field(..., description="Normalized spoken text")
    raw_text: Optional[str] = Field(default=None, description="Original raw prose text")
    start: float = Field(..., description="Segment start timestamp in seconds")
    end: float = Field(..., description="Segment end timestamp in seconds")
    duration: float = Field(..., description="Exact speech duration in seconds")
    pause_after: float = Field(default=0.4, description="Silence pause following segment")
    visual_query: str = Field(default="", description="Visual search query")
    asset_path: str = Field(default="", description="Path to associated visual asset")
    asset_type: str = Field(default="image", description="Asset type: image or video")
    motion_type: MotionType = Field(default="zoom_in", description="Camera motion direction")

    @field_validator("start", "end", "duration", "pause_after")
    @classmethod
    def round_timestamps(cls, v: float) -> float:
        return round(float(v), 3)


class ChapterAlignment(BaseModel):
    """Data contract for Stage 4 timeline alignment output."""

    project_id: str = Field(..., description="Unique project slug")
    chapter_index: int = Field(..., description="1-based chapter index")
    audio_file: str = Field(..., description="Path to mastered chapter audio")
    total_duration: float = Field(..., description="Total duration of chapter in seconds")
    segments: List[AlignedSegment] = Field(
        default_factory=list, description="Ordered sequence of aligned segments"
    )

    @field_validator("total_duration")
    @classmethod
    def round_duration(cls, v: float) -> float:
        return round(float(v), 3)


def seconds_to_srt_time(seconds: float) -> str:
    """Converts a floating-point timestamp in seconds to standard SRT format (HH:MM:SS,mmm)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3600000
    remainder = total_ms % 3600000
    minutes = remainder // 60000
    remainder = remainder % 60000
    secs = remainder // 1000
    ms = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def wrap_subtitle_text(text: str, max_line_len: int = 42, max_lines: int = 2) -> str:
    """Wraps subtitle text into balanced lines for high readability without breaking words.

    Args:
        text: Subtitle text string.
        max_line_len: Maximum target character count per line (default: 42).
        max_lines: Maximum lines per cue before clean division (default: 2).

    Returns:
        Cleanly wrapped subtitle text with newline separators.
    """
    words = text.strip().split()
    if not words:
        return ""

    lines: List[str] = []
    current_line: List[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        if current_line and (current_len + 1 + word_len) > max_line_len:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_len = word_len
        else:
            current_line.append(word)
            current_len += word_len + (1 if current_len > 0 else 0)

    if current_line:
        lines.append(" ".join(current_line))

    # If exceeding max_lines, compact into max_lines
    if len(lines) > max_lines:
        midpoint = len(words) // 2
        line1 = " ".join(words[:midpoint])
        line2 = " ".join(words[midpoint:])
        return f"{line1}\n{line2}"

    return "\n".join(lines)


def align_chapter(
    project_id: str,
    chapter_script: ChapterScript,
    audio_manifest: ChapterAudioManifest,
    asset_map: Optional[Dict[int, Dict[str, str]]] = None,
) -> ChapterAlignment:
    """Calculates zero-drift mathematical timestamps for all sentences in a chapter.

    Formula:
        Start_0 = 0.0
        End_i = Start_i + Duration_i
        Start_{i+1} = End_i + Pause_i

    Args:
        project_id: Project identifier.
        chapter_script: Validated ChapterScript with sentences and queries.
        audio_manifest: Synthesized ChapterAudioManifest with exact clip durations.
        asset_map: Optional mapping from sentence_index to {'asset_path': ..., 'asset_type': ...}.

    Returns:
        ChapterAlignment instance linking sentences to timeline and media assets.
    """
    clips_by_idx = {c.sentence_index: c for c in audio_manifest.clips}
    asset_map = asset_map or {}

    current_time = 0.0
    segments: List[AlignedSegment] = []

    for s_idx, sentence in enumerate(chapter_script.sentences):
        clip = clips_by_idx.get(sentence.sentence_index)
        if clip is not None:
            duration = clip.duration_seconds
            pause_after = clip.pause_after_seconds
        else:
            # Fallback estimation if clip missing
            duration = max(1.0, len(sentence.spoken_text.split()) * 0.4)
            pause_after = sentence.pause_after_seconds

        start_time = current_time
        end_time = start_time + duration

        # Asset metadata
        asset_info = asset_map.get(sentence.sentence_index, {})
        asset_path = asset_info.get("asset_path", "")
        asset_type = asset_info.get("asset_type", "image")

        segments.append(
            AlignedSegment(
                sentence_index=sentence.sentence_index,
                text=sentence.spoken_text,
                raw_text=sentence.raw_text,
                start=round(start_time, 3),
                end=round(end_time, 3),
                duration=round(duration, 3),
                pause_after=round(pause_after, 3),
                visual_query=sentence.visual_query,
                asset_path=asset_path,
                asset_type=asset_type,
                motion_type=sentence.motion_type,
            )
        )

        current_time = end_time + pause_after

    total_duration = current_time if segments else audio_manifest.total_duration

    return ChapterAlignment(
        project_id=project_id,
        chapter_index=chapter_script.chapter_index,
        audio_file=audio_manifest.audio_file,
        total_duration=round(total_duration, 3),
        segments=segments,
    )


def generate_srt(
    alignment: ChapterAlignment,
    output_path: Path,
    use_raw_text: bool = False,
    max_line_len: int = 42,
) -> Path:
    """Generates standard YouTube-compliant SubRip (.srt) subtitle file from alignment.

    Args:
        alignment: ChapterAlignment data contract.
        output_path: Filepath where .srt should be written.
        use_raw_text: If True, uses raw prose text; otherwise spoken text.
        max_line_len: Character length limit per line.

    Returns:
        Path to generated .srt file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_blocks: List[str] = []

    for idx, seg in enumerate(alignment.segments, start=1):
        content = seg.raw_text if (use_raw_text and seg.raw_text) else seg.text
        formatted_text = wrap_subtitle_text(content, max_line_len=max_line_len)
        start_str = seconds_to_srt_time(seg.start)
        # End at either speech end or up to speech end + half pause for smooth display
        display_end = min(seg.end + (seg.pause_after * 0.5), seg.end + seg.pause_after)
        end_str = seconds_to_srt_time(display_end)

        block = f"{idx}\n{start_str} --> {end_str}\n{formatted_text}\n"
        srt_blocks.append(block)

    full_srt = "\n".join(srt_blocks)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_srt)

    logger.info("Generated SRT subtitles with %d cues -> %s", len(srt_blocks), output_path)
    return output_path


def save_alignment(alignment: ChapterAlignment, output_path: Path) -> Path:
    """Saves alignment contract to alignment.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alignment.model_dump(), f, indent=2)
    return output_path
