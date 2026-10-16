import os
from pathlib import Path
from typing import Optional, Tuple
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application settings with environment variable and YAML file support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings (DeepSeek-V3 / OpenAI-compatible)
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL"
    )
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # Voice Engine (Qwen-TTS Gradio Endpoint)
    qwen_tts_gradio_url: str = Field(
        default="http://127.0.0.1:7860/", alias="QWEN_TTS_GRADIO_URL"
    )
    ref_voice_audio_path: Path = Field(
        default=Path("config/reference_voice.wav"), alias="REF_VOICE_AUDIO_PATH"
    )
    ref_voice_transcript_path: Path = Field(
        default=Path("config/reference_voice.txt"),
        alias="REF_VOICE_TRANSCRIPT_PATH",
    )
    tts_fade_ms: int = 10
    tts_pause_seconds: float = 0.4
    target_lufs: float = -14.0

    # Asset Harvester
    pexels_api_key: str = Field(default="", alias="PEXELS_API_KEY")
    pixabay_api_key: str = Field(default="", alias="PIXABAY_API_KEY")
    ai_image_provider: str = "pollinations"

    # Pipeline Paths & Output
    output_dir: Path = Field(default=Path("data/projects"), alias="OUTPUT_DIR")
    cache_dir: Path = Field(default=Path("data/cache"), alias="CACHE_DIR")
    default_fps: int = 30
    default_resolution: Tuple[int, int] = (1920, 1080)


def load_settings(config_file: Optional[Path | str] = None) -> AppSettings:
    """Load settings by merging config/settings.yaml and environment variables."""
    config_data = {}
    
    if config_file is None:
        default_path = Path("config/settings.yaml")
        if default_path.exists():
            config_file = default_path

    if config_file:
        path = Path(config_file)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_data = loaded

    # Let environment variables take precedence over yaml entries
    env_overrides = {}
    for key, env_var in [
        ("deepseek_api_key", "DEEPSEEK_API_KEY"),
        ("deepseek_base_url", "DEEPSEEK_BASE_URL"),
        ("deepseek_model", "DEEPSEEK_MODEL"),
        ("qwen_tts_gradio_url", "QWEN_TTS_GRADIO_URL"),
        ("pexels_api_key", "PEXELS_API_KEY"),
        ("pixabay_api_key", "PIXABAY_API_KEY"),
        ("output_dir", "OUTPUT_DIR"),
    ]:
        val = os.getenv(env_var)
        if val is not None:
            env_overrides[key] = val

    merged = {**config_data, **env_overrides}
    return AppSettings(**merged)
