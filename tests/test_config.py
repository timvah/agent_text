from pathlib import Path
from src.config_loader import load_settings, AppSettings


def test_load_default_settings():
    settings = load_settings("config/settings.yaml")
    assert isinstance(settings, AppSettings)
    assert settings.deepseek_model == "deepseek-chat"
    assert settings.default_fps == 30
    assert settings.default_resolution == (1920, 1080)
    assert settings.target_lufs == -14.0
    assert settings.tts_fade_ms == 10
