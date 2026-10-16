"""Filesystem and scaffolding utilities for narrative video generation."""

import hashlib
import re
from pathlib import Path
from typing import Dict


def sanitize_filename(name: str) -> str:
    """Converts a title or topic into a filesystem-safe identifier."""
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[-\s]+", "_", name)


def compute_text_hash(text: str) -> str:
    """Returns a short SHA-256 hash for caching queries and prompts."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def create_project_scaffolding(project_id: str, base_dir: Path = Path("data/projects")) -> Dict[str, Path]:
    """Creates standard directory tree for a project.

    Structure:
    data/projects/{project_id}/
    ├── meta/
    ├── chapters/
    │   └── ch{index}/
    │       ├── assets/
    │       └── audio/
    └── renders/
    """
    clean_id = sanitize_filename(project_id)
    project_root = base_dir / clean_id
    meta_dir = project_root / "meta"
    chapters_dir = project_root / "chapters"
    renders_dir = project_root / "renders"

    for d in (project_root, meta_dir, chapters_dir, renders_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "meta_dir": meta_dir,
        "chapters_dir": chapters_dir,
        "renders_dir": renders_dir,
    }


def create_chapter_scaffolding(chapter_index: int, chapters_dir: Path) -> Dict[str, Path]:
    """Creates standard directory tree for an individual chapter."""
    ch_slug = f"ch{chapter_index:02d}"
    ch_dir = chapters_dir / ch_slug
    assets_dir = ch_dir / "assets"
    audio_dir = ch_dir / "audio"
    render_dir = ch_dir / "render"

    for d in (ch_dir, assets_dir, audio_dir, render_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "chapter_dir": ch_dir,
        "assets_dir": assets_dir,
        "audio_dir": audio_dir,
        "render_dir": render_dir,
    }
