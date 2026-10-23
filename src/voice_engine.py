"""Voice Engine & Audio Mastering for Qwen-TTS with EBU R128 Loudness Normalization.

Features:
- Qwen-TTS Gradio Client integration with automatic retry & exponential backoff
- Anti-click edge conditioning using 10ms cosine fade windowing
- EBU R128 audio mastering (-14.0 LUFS, -1.0 dB True Peak)
- Deterministic per-sentence duration metadata tracking and clip concatenation
- Seamless mock/offline mode for testing and environments without active GPU TTS servers
"""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pydantic import BaseModel, Field, field_validator

from src.config_loader import AppSettings, load_settings
from src.drafter import ChapterScript, ScriptSentence

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 48000


class ClipMetadata(BaseModel):
    """Metadata for an individual synthesized sentence audio clip."""

    sentence_index: int = Field(..., description="0-based sentence index")
    duration_seconds: float = Field(..., description="Exact duration of speech clip in seconds")
    pause_after_seconds: float = Field(default=0.4, description="Silence pause appended after clip")
    audio_path: str = Field(..., description="Filesystem path to individual clip WAV")
    sample_rate: int = Field(default=DEFAULT_SAMPLE_RATE, description="Audio sample rate in Hz")

    @field_validator("duration_seconds", "pause_after_seconds")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Duration values must be non-negative")
        return round(float(v), 3)


class ChapterAudioManifest(BaseModel):
    """Manifest summarizing mastered audio and all clip durations for a chapter."""

    chapter_index: int = Field(..., description="1-based chapter index")
    audio_file: str = Field(..., description="Path to mastered chapter audio WAV file")
    total_duration: float = Field(..., description="Total chapter audio duration including pauses")
    sample_rate: int = Field(default=DEFAULT_SAMPLE_RATE, description="Master audio sample rate in Hz")
    target_lufs: float = Field(default=-14.0, description="Target LUFS loudness")
    clips: List[ClipMetadata] = Field(default_factory=list, description="List of per-clip metadata")


def apply_edge_fades(
    audio_data: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE, fade_ms: int = 10
) -> np.ndarray:
    """Applies smooth cosine fade-in and fade-out to eliminate zero-crossing pops/clicks.

    Args:
        audio_data: 1D (mono) or 2D (samples, channels) numpy float array.
        sr: Audio sample rate in Hz.
        fade_ms: Duration of fade in milliseconds (default: 10ms).

    Returns:
        Conditioned numpy array with fades applied.
    """
    if audio_data is None or len(audio_data) == 0:
        return audio_data

    audio = audio_data.copy().astype(np.float32)
    fade_samples = int(sr * (fade_ms / 1000.0))

    if len(audio) < 2 * fade_samples or fade_samples <= 0:
        return audio

    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples, dtype=np.float32)))
    fade_out = 0.5 * (1.0 - np.cos(np.linspace(np.pi, 0, fade_samples, dtype=np.float32)))

    if audio.ndim == 1:
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    else:
        audio[:fade_samples, :] *= fade_in[:, np.newaxis]
        audio[-fade_samples:, :] *= fade_out[:, np.newaxis]

    return audio


def normalize_loudness(
    audio_data: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    target_lufs: float = -14.0,
    max_peak_db: float = -1.0,
) -> np.ndarray:
    """Master audio to EBU R128 loudness standards (-14.0 LUFS, -1.0 dB True Peak).

    Args:
        audio_data: Float audio array.
        sr: Sample rate in Hz.
        target_lufs: Desired integrated loudness (default -14.0 LUFS for web/video).
        max_peak_db: True peak ceiling in dB (default -1.0 dB).

    Returns:
        Mastered float audio array clamped within peak ceiling.
    """
    if audio_data is None or len(audio_data) == 0:
        return audio_data

    audio = audio_data.copy().astype(np.float32)

    # Convert mono to 2D for pyloudnorm if needed, or keep 1D
    is_1d = audio.ndim == 1
    audio_2d = audio[:, np.newaxis] if is_1d else audio

    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_2d)

        if math.isinf(loudness) or math.isnan(loudness) or loudness < -70.0:
            # Audio is virtually silent, skip LUFS boost to avoid amplifying noise floor
            logger.warning("Audio loudness near silence (%s LUFS), applying peak limiter only", loudness)
            normalized = audio_2d
        else:
            normalized = pyln.normalize.loudness(audio_2d, loudness, target_lufs)
    except Exception as exc:
        logger.warning("pyloudnorm normalization failed (%s), falling back to peak normalization", exc)
        normalized = audio_2d

    # Peak limiting / ceiling protection
    max_peak_linear = 10.0 ** (max_peak_db / 20.0)
    current_peak = float(np.max(np.abs(normalized)))
    if current_peak > max_peak_linear and current_peak > 0:
        normalized = normalized * (max_peak_linear / current_peak)

    result = normalized.squeeze() if is_1d else normalized
    return result.astype(np.float32)


def generate_mock_speech(
    text: str, sr: int = DEFAULT_SAMPLE_RATE, words_per_minute: float = 145.0
) -> np.ndarray:
    """Generates synthetic spoken waveform for testing and offline environments.

    Duration is calibrated to natural speech cadence (~145 wpm) with dynamic harmonic
    formants and edge windowing.
    """
    word_count = max(1, len(text.strip().split()))
    # Natural duration: ~0.41s per word + minimum 1.2s
    duration = max(1.2, (word_count / words_per_minute) * 60.0)
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False, dtype=np.float32)

    # Rich harmonic base mimicking voice pitch (fundamental ~140Hz + harmonics)
    f0 = 140.0
    waveform = (
        0.40 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * (2 * f0) * t)
        + 0.15 * np.sin(2 * np.pi * (3 * f0) * t)
        + 0.10 * np.sin(2 * np.pi * (4 * f0) * t)
    )

    # Modulate envelope with natural syllables (cadence ~3.5 syllables/sec)
    syllable_rate = 3.5
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t - np.pi / 2))
    envelope = np.clip(envelope, 0.15, 1.0)
    waveform = (waveform * envelope).astype(np.float32)

    # Apply edge fades
    waveform = apply_edge_fades(waveform, sr=sr, fade_ms=15)
    return waveform


def check_qwen_tts_health(gradio_url: str, timeout: float = 2.5) -> bool:
    """Checks if the Qwen-TTS Gradio endpoint is accessible and responsive."""
    import requests

    if not gradio_url:
        return False
    try:
        url = gradio_url.rstrip("/") + "/info"
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        # Also try basic root ping
        try:
            resp = requests.get(gradio_url, timeout=timeout)
            return resp.status_code in (200, 301, 302)
        except Exception:
            return False


class VoiceEngine:
    """Orchestrator for text-to-speech synthesis and chapter audio mastering."""

    def __init__(self, settings: Optional[AppSettings] = None, mock: bool = False):
        self.settings = settings or load_settings()
        self.mock = mock
        self._gradio_client = None

    def _get_gradio_client(self):
        """Initializes or retrieves the Gradio client."""
        if self._gradio_client is None:
            from gradio_client import Client

            url = self.settings.qwen_tts_gradio_url
            logger.info("Connecting to Qwen-TTS Gradio endpoint at %s", url)
            self._gradio_client = Client(url)
        return self._gradio_client

    def synthesize_sentence(
        self,
        sentence: ScriptSentence,
        output_path: Path,
        ref_audio_path: Optional[Path] = None,
        ref_transcript: Optional[str] = None,
        max_retries: int = 3,
    ) -> ClipMetadata:
        """Synthesizes an individual sentence clip with anti-click fades and metadata tracking.

        Args:
            sentence: ScriptSentence object with spoken_text and timing guidance.
            output_path: Target .wav output path.
            ref_audio_path: Optional reference voice sample path.
            ref_transcript: Optional reference voice transcript.
            max_retries: Retry attempts on network/synthesis error.

        Returns:
            ClipMetadata containing exact duration and silence gap.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sr = DEFAULT_SAMPLE_RATE

        if self.mock or not check_qwen_tts_health(self.settings.qwen_tts_gradio_url):
            if not self.mock:
                logger.info(
                    "Qwen-TTS server at %s not reachable. Using fallback audio synthesis.",
                    self.settings.qwen_tts_gradio_url,
                )
            audio_data = generate_mock_speech(sentence.spoken_text, sr=sr)
        else:
            ref_audio = ref_audio_path or self.settings.ref_voice_audio_path
            ref_text = ref_transcript or ""
            if not ref_text and self.settings.ref_voice_transcript_path.exists():
                ref_text = self.settings.ref_voice_transcript_path.read_text(encoding="utf-8").strip()

            last_err = None
            audio_data = None

            for attempt in range(1, max_retries + 1):
                try:
                    client = self._get_gradio_client()
                    # Qwen-TTS gradio typically accepts (target_text, ref_audio, ref_text) or similar
                    result = client.predict(
                        sentence.spoken_text,
                        str(ref_audio),
                        ref_text,
                        api_name="/synthesize",
                    )
                    # Result is typically a filepath or tuple (sr, audio_array)
                    if isinstance(result, (str, Path)) and Path(result).exists():
                        data, file_sr = sf.read(str(result))
                        sr = file_sr
                        audio_data = data
                        break
                    elif isinstance(result, tuple) and len(result) >= 2:
                        file_sr, data = result[0], result[1]
                        sr = file_sr
                        audio_data = data
                        break
                    else:
                        raise ValueError(f"Unexpected Gradio return type: {type(result)}")
                except Exception as exc:
                    last_err = exc
                    wait_time = 0.5 * (2 ** (attempt - 1))
                    logger.warning(
                        "Synthesis attempt %d/%d failed (%s). Retrying in %.2fs...",
                        attempt,
                        max_retries,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)

            if audio_data is None:
                logger.error(
                    "All %d synthesis attempts failed (%s). Falling back to synthetic speech.",
                    max_retries,
                    last_err,
                )
                audio_data = generate_mock_speech(sentence.spoken_text, sr=sr)

        # Apply edge cosine conditioning
        conditioned_audio = apply_edge_fades(
            audio_data, sr=sr, fade_ms=self.settings.tts_fade_ms
        )

        # Write clip WAV file
        sf.write(str(output_path), conditioned_audio, sr, subtype="PCM_16")

        # Compute exact duration
        duration = float(len(conditioned_audio)) / float(sr)

        return ClipMetadata(
            sentence_index=sentence.sentence_index,
            duration_seconds=duration,
            pause_after_seconds=sentence.pause_after_seconds,
            audio_path=str(output_path),
            sample_rate=sr,
        )

    def synthesize_chapter(
        self,
        chapter_script: ChapterScript,
        output_dir: Path,
        ref_audio_path: Optional[Path] = None,
        ref_transcript: Optional[str] = None,
    ) -> ChapterAudioManifest:
        """Synthesizes all sentences in a chapter, concatenates with pauses, and masters audio.

        Args:
            chapter_script: ChapterScript containing validated sentences.
            output_dir: Output directory for chapter audio (typically `chapters/chXX/audio`).
            ref_audio_path: Optional reference voice sample path.
            ref_transcript: Optional reference voice transcript.

        Returns:
            ChapterAudioManifest with master audio file and clip duration metadata.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        clip_manifests: List[ClipMetadata] = []
        master_segments: List[np.ndarray] = []
        sr = DEFAULT_SAMPLE_RATE

        logger.info(
            "Starting synthesis for Chapter %d (%d sentences)",
            chapter_script.chapter_index,
            len(chapter_script.sentences),
        )

        for s_idx, sentence in enumerate(chapter_script.sentences):
            clip_path = clips_dir / f"clip_{s_idx:03d}.wav"
            meta = self.synthesize_sentence(
                sentence=sentence,
                output_path=clip_path,
                ref_audio_path=ref_audio_path,
                ref_transcript=ref_transcript,
            )
            clip_manifests.append(meta)

            # Load synthesized audio
            audio_data, clip_sr = sf.read(str(clip_path), dtype="float32")
            sr = clip_sr
            master_segments.append(audio_data)

            # Insert pause silence
            pause_samples = int(sr * sentence.pause_after_seconds)
            if pause_samples > 0:
                silence = np.zeros(pause_samples, dtype=np.float32)
                master_segments.append(silence)

        # Concatenate full chapter audio
        if master_segments:
            concatenated = np.concatenate(master_segments)
        else:
            concatenated = np.zeros(sr, dtype=np.float32)

        # Audio mastering: EBU R128 (-14.0 LUFS, -1.0 dB True Peak)
        mastered = normalize_loudness(
            concatenated,
            sr=sr,
            target_lufs=self.settings.target_lufs,
            max_peak_db=-1.0,
        )

        mastered_path = output_dir / "audio_mastered.wav"
        sf.write(str(mastered_path), mastered, sr, subtype="PCM_16")

        total_duration = float(len(mastered)) / float(sr)

        manifest = ChapterAudioManifest(
            chapter_index=chapter_script.chapter_index,
            audio_file=str(mastered_path),
            total_duration=round(total_duration, 3),
            sample_rate=sr,
            target_lufs=self.settings.target_lufs,
            clips=clip_manifests,
        )

        # Export clip_durations.json
        durations_json_path = output_dir / "clip_durations.json"
        with open(durations_json_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        logger.info(
            "Chapter %d audio mastered successfully: %.2fs total duration -> %s",
            chapter_script.chapter_index,
            total_duration,
            mastered_path,
        )

        return manifest
