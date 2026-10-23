"""Macro-Planner & Narrative Beat Architect using DeepSeek-V3.

Divides long-form narrative topics into structured, thematic chapters with:
- Dramatic tension arcs & narrative progression beats
- Visual palettes and aesthetic anchors
- Dynamic chapter count and timing calculations
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from src.config_loader import AppSettings, load_settings
from src.utils import sanitize_filename

logger = logging.getLogger(__name__)


class ChapterPlan(BaseModel):
    """Data contract for an individual narrative chapter plan."""

    chapter_index: int = Field(..., description="1-based sequential chapter index")
    title: str = Field(..., description="Thematic title of the chapter")
    target_word_count: int = Field(..., description="Target word count for chapter narration")
    target_duration_seconds: int = Field(..., description="Target duration in seconds")
    core_tension: str = Field(..., description="Dramatic conflict or central premise of chapter")
    narrative_beats: List[str] = Field(
        default_factory=list, description="Sequential narrative story beats"
    )
    visual_palette: List[str] = Field(
        default_factory=list, description="Visual keywords and aesthetic elements"
    )
    voice_tone: str = Field(
        default="documentary, deliberate", description="Voice direction for synthesis"
    )

    @field_validator("chapter_index")
    @classmethod
    def validate_index(cls, v: int) -> int:
        if v < 1:
            raise ValueError("chapter_index must be >= 1 (1-based indexing)")
        return v


class ProjectPlan(BaseModel):
    """Data contract for master project narrative plan (Stage 1 output)."""

    project_id: str = Field(..., description="Filesystem-safe unique slug for project")
    topic: str = Field(..., description="Main topic or premise")
    target_duration_minutes: int = Field(..., description="Total target duration in minutes")
    visual_style_anchor: str = Field(..., description="Overall visual aesthetic guidance")
    bgm_mood: str = Field(
        default="somber, mysterious, scholarly", description="Background music mood"
    )
    chapters: List[ChapterPlan] = Field(..., description="Sequential chapters list")

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        clean = sanitize_filename(v)
        if not clean:
            raise ValueError("project_id cannot be empty")
        return clean

    @field_validator("chapters")
    @classmethod
    def validate_chapters(cls, v: List[ChapterPlan]) -> List[ChapterPlan]:
        if not v:
            raise ValueError("Project plan must contain at least one chapter")
        return v


def calculate_chapter_count(duration_minutes: int) -> int:
    """Calculates dynamically target chapter count for a video duration.
    
    Rule from specification: max(3, round(duration_min / 2.0)).
    For a 12-minute video: 6 chapters.
    For an 8-minute video: 4 chapters.
    For short videos (<=4 min): minimum 3 chapters.
    """
    return max(3, round(duration_minutes / 2.0))


def get_deepseek_client(settings: Optional[AppSettings] = None) -> OpenAI:
    """Initializes OpenAI client configured for DeepSeek API."""
    settings = settings or load_settings()
    api_key = settings.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not configured. Please set DEEPSEEK_API_KEY in your "
            "environment or config/settings.yaml."
        )
    return OpenAI(
        api_key=api_key,
        base_url=settings.deepseek_base_url,
    )


PLANNER_SYSTEM_PROMPT = """You are an elite historical documentary narrative architect and master planner.
Your mission is to divide long-form documentary topics (8-60 minutes) into compelling, cinematic chapters.
Each chapter must have:
- A high-stakes core dramatic tension.
- Concrete narrative story beats.
- Rich, evocative visual palettes (settings, artifacts, landscapes, archival elements).
- Voice direction for narration pacing and mood.

Strict Requirement:
Return ONLY a valid JSON object matching the requested schema. Do NOT include markdown code blocks, explanatory prose, or commentary.
"""

PLANNER_USER_PROMPT_TEMPLATE = """Generate a master documentary project plan for:
Topic: "{topic}"
Target Duration: {duration_minutes} minutes
Tone: "{tone}"
Number of Chapters: {num_chapters}
{visual_style_hint}

Target chapter breakdown:
- Each chapter target duration: ~{seconds_per_chapter} seconds.
- Each chapter target word count: ~{words_per_chapter} words (approx. 130-150 words per minute speaking rate).

Return a JSON object with this exact structure:
{{
  "project_id": "{project_id_hint}",
  "topic": "{topic}",
  "target_duration_minutes": {duration_minutes},
  "visual_style_anchor": "<detailed visual aesthetic description>",
  "bgm_mood": "<ambient music mood descriptor>",
  "chapters": [
    {{
      "chapter_index": 1,
      "title": "<thematic title>",
      "target_word_count": {words_per_chapter},
      "target_duration_seconds": {seconds_per_chapter},
      "core_tension": "<dramatic conflict or mystery driving this chapter>",
      "narrative_beats": [
        "<beat 1>",
        "<beat 2>",
        "<beat 3>"
      ],
      "visual_palette": ["<visual keyword 1>", "<visual keyword 2>", "<visual keyword 3>", "<visual keyword 4>"],
      "voice_tone": "<voice delivery instructions>"
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


def generate_mock_project_plan(
    topic: str,
    duration_min: int = 8,
    tone: str = "documentary, deliberate",
    visual_style: Optional[str] = None,
) -> ProjectPlan:
    """Generates a synthetic, structured documentary ProjectPlan for offline testing."""
    num_chapters = calculate_chapter_count(duration_min)
    seconds_per_chapter = (duration_min * 60) // num_chapters
    words_per_chapter = int((seconds_per_chapter / 60.0) * 140)
    project_id = sanitize_filename(topic)
    style = (
        visual_style
        or "historical archival paintings, ancient papyrus textures, cinematic dramatic reenactment"
    )

    chapters = []
    for i in range(1, num_chapters + 1):
        chapters.append(
            ChapterPlan(
                chapter_index=i,
                title=f"{topic} - Part {i}",
                target_word_count=words_per_chapter,
                target_duration_seconds=seconds_per_chapter,
                core_tension=f"The unfolding critical turning point of {topic} in stage {i}.",
                narrative_beats=[
                    f"Introduction to the key events of part {i} regarding {topic}.",
                    f"A dramatic shift in circumstances forces unexpected consequences in part {i}.",
                    f"The lasting reverberations that defined part {i} of {topic}.",
                ],
                visual_palette=[
                    sanitize_filename(topic).replace("_", " "),
                    "archival painting",
                    "historical landscape",
                    "dramatic lighting",
                ],
                voice_tone=tone,
            )
        )

    return ProjectPlan(
        project_id=project_id,
        topic=topic,
        target_duration_minutes=duration_min,
        visual_style_anchor=style,
        bgm_mood="somber, mysterious, scholarly",
        chapters=chapters,
    )


def generate_project_plan(
    topic: str,
    duration_min: int = 12,
    tone: str = "scholarly, cinematic, dramatic",
    visual_style: Optional[str] = None,
    settings: Optional[AppSettings] = None,
    client: Optional[OpenAI] = None,
    duration_minutes: Optional[int] = None,
    mock: bool = False,
) -> ProjectPlan:
    """Generates a complete documentary ProjectPlan using DeepSeek-V3.
    
    Args:
        topic: Narrative topic (e.g., 'The Destruction of the Library of Alexandria').
        duration_min: Duration in minutes (default: 12).
        tone: Overall narrative tone.
        visual_style: Optional visual style anchor override.
        settings: Application settings.
        client: Pre-configured OpenAI/DeepSeek client (useful for testing and dependency injection).
        duration_minutes: Alias for duration_min.
        mock: If True, returns synthetic mock plan without API calls.

    Returns:
        Validated ProjectPlan object.
    """
    if duration_minutes is not None:
        duration_min = duration_minutes
    if mock:
        return generate_mock_project_plan(
            topic=topic,
            duration_min=duration_min,
            tone=tone,
            visual_style=visual_style,
        )
    settings = settings or load_settings()
    num_chapters = calculate_chapter_count(duration_min)
    seconds_per_chapter = (duration_min * 60) // num_chapters
    words_per_chapter = int((seconds_per_chapter / 60.0) * 140)
    project_id_hint = sanitize_filename(topic)

    visual_style_hint = (
        f'Visual Style Anchor: "{visual_style}"'
        if visual_style
        else "Visual Style Anchor: historical archival paintings, cinematic atmospheric reenactment, ancient textures"
    )

    user_prompt = PLANNER_USER_PROMPT_TEMPLATE.format(
        topic=topic,
        duration_minutes=duration_min,
        tone=tone,
        num_chapters=num_chapters,
        seconds_per_chapter=seconds_per_chapter,
        words_per_chapter=words_per_chapter,
        project_id_hint=project_id_hint,
        visual_style_hint=visual_style_hint,
    )

    if client is None:
        client = get_deepseek_client(settings)

    logger.info(
        f"Calling DeepSeek API to plan project '{topic}' ({duration_min} min, {num_chapters} chapters)..."
    )

    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
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
        logger.error(f"Failed to decode DeepSeek JSON output: {content}")
        raise ValueError(f"DeepSeek returned invalid JSON: {e}") from e

    # Ensure project_id is populated and clean
    if not raw_dict.get("project_id"):
        raw_dict["project_id"] = project_id_hint
    raw_dict["project_id"] = sanitize_filename(raw_dict["project_id"])

    # Ensure target_duration_minutes matches requested duration if omitted
    if not raw_dict.get("target_duration_minutes"):
        raw_dict["target_duration_minutes"] = duration_min

    # Ensure chapter indices are 1-based sequential
    if "chapters" in raw_dict and isinstance(raw_dict["chapters"], list):
        for idx, ch in enumerate(raw_dict["chapters"], start=1):
            if isinstance(ch, dict):
                ch["chapter_index"] = idx
                if not ch.get("target_duration_seconds"):
                    ch["target_duration_seconds"] = seconds_per_chapter
                if not ch.get("target_word_count"):
                    ch["target_word_count"] = words_per_chapter

    plan = ProjectPlan.model_validate(raw_dict)
    logger.info(
        f"Successfully generated ProjectPlan '{plan.project_id}' with {len(plan.chapters)} chapters."
    )
    return plan


def save_project_plan(plan: ProjectPlan, output_path: Path | str) -> Path:
    """Saves ProjectPlan to JSON file with indentation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(plan.model_dump_json(indent=2))
    logger.info(f"Saved project plan to {path}")
    return path


def load_project_plan(input_path: Path | str) -> ProjectPlan:
    """Loads and validates ProjectPlan from a JSON file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Project plan file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ProjectPlan.model_validate(data)
