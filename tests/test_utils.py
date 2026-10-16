import shutil
from pathlib import Path
from src.utils import sanitize_filename, compute_text_hash, create_project_scaffolding, create_chapter_scaffolding


def test_sanitize_filename():
    assert sanitize_filename("The Bronze Age Collapse: Why Did It Fall?") == "the_bronze_age_collapse_why_did_it_fall"
    assert sanitize_filename("Project #1 (Alexandria!)") == "project_1_alexandria"


def test_compute_text_hash():
    h1 = compute_text_hash("alexandria harbor ancient lighthouse")
    h2 = compute_text_hash("alexandria harbor ancient lighthouse")
    h3 = compute_text_hash("different query")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_project_scaffolding(tmp_path):
    paths = create_project_scaffolding("test_doc", base_dir=tmp_path)
    assert paths["project_root"].exists()
    assert paths["meta_dir"].exists()
    assert paths["chapters_dir"].exists()
    assert paths["renders_dir"].exists()

    ch_paths = create_chapter_scaffolding(1, paths["chapters_dir"])
    assert ch_paths["chapter_dir"].exists()
    assert ch_paths["assets_dir"].exists()
    assert ch_paths["audio_dir"].exists()
    assert ch_paths["render_dir"].exists()
