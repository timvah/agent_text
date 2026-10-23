# Autonomous Long-Narrative AI Video Generation System
## Production Engineering Specification & Master Implementation Plan

This document is the architectural blueprint, data specification, and execution plan for an automated, long-form (8–60 minute) narrative video creation engine. It is optimized for execution by **Antigravity** as the orchestrator and code-runner.

---

## 1. System Architecture & High-Level Flow

```
[User Input: Topic / Premise / Duration (8-60m) / Tone]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Macro-Planner & Narrative Architect (DeepSeek-V3 / Cloud LLM) │
│ - Divides duration into 2–4 minute thematic chapters                    │
│ - Generates dramatic tension arcs, narrative beats, and visual palettes │
│ - Establishes visual style anchors (e.g., historical art vs modern b-roll)│
└────────────────────────┬────────────────────────────────────────────────┘
                         │ (project_plan.json)
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Narrative Drafter & Rule Normalizer (DeepSeek-V3 + Python TN)  │
│ - DeepSeek-V3 expands chapters into rich, documentary-grade prose       │
│ - Deterministic Text Normalizer (num2words, acronyms, phonetic cleanup) │
│ - Maps concrete visual search queries & camera directions per sentence  │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ (chapter_script.json)
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Voice Engine & Audio Mastering (Local Qwen-TTS via Gradio)     │
│ - Synthesizes sentence clips using reference voice sample               │
│ - Anti-click conditioning: 10ms cosine windowing & zero-crossing fades  │
│ - Audio Mastering: EBU R128 loudness normalization (-14 LUFS)           │
│ - Computes exact duration metadata per clip during synthesis            │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ (chapter_audio.wav & clip_durations.json)
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Deterministic Alignment & Subtitle Generator (Zero-Drift Math) │
│ - Mathematical timestamping (Start_i = Sum(Durations[:i]) + Silence_i)   │
│ - Zero CPU Whisper hallucination risk on ancient/specialized names      │
│ - Generates standard non-karaoke .srt and cut manifests instantly       │
│ - (Optional: Whisper forced-alignment toggle for word-level karaoke)   │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ (alignment.json & subtitles.srt)
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Resilient Multi-Source Asset Harvester                         │
│ - Multi-tier fallback: Pexels API -> Pixabay API -> Wikimedia Commons   │
│ - AI Image Fallback (Local SDXL/Flux or API) for niche historical scenes│
│ - Asset cache, rate-limit backoff, and clip duration looping/scaling    │
│ - Validates 16:9 orientation, smart-crops/scales to minimum 1080p       │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ (assets_manifest.json)
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 6: Video Assembly, Dynamic Ken Burns & Audio Ducking (FFmpeg)     │
│ - Multi-motion Ken Burns (Pan L->R, R->L, Zoom In, Zoom Out, Tilt)      │
│ - Standardized intermediate rendering (Fixed 30fps CFR, yuv420p, 48kHz) │
│ - Ambient background score mixing with voice-activated sidechain ducking│
│ - Chapter-level rendering + final lossless concatenation into master MP4│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
narrative_video_pipeline/
├── config/
│   ├── settings.yaml            # API keys, endpoint URLs, hardware profiles
│   ├── reference_voice.wav      # Pristine 5–8s voice clone reference sample
│   └── reference_voice.txt      # Verbatim transcript of reference sample
├── assets/
│   ├── bgm/                     # Curated royalty-free ambient music tracks
│   └── fonts/                   # High-legibility subtitle fonts (e.g., Montserrat, Inter)
├── src/
│   ├── __init__.py
│   ├── config_loader.py         # Pydantic settings validator
│   ├── text_normalizer.py       # Deterministic G2P, num2words, punctuation cleaner
│   ├── planner.py               # DeepSeek-V3 macro beat architect
│   ├── drafter.py               # DeepSeek-V3 script expansion & visual query generator
│   ├── voice_engine.py          # Qwen-TTS Gradio Client, seam smoothing & LUFS normalizer
│   ├── aligner.py               # Deterministic mathematical timeline & SRT generator
│   ├── asset_harvester.py       # Pexels, Pixabay, Wikimedia & AI image fallback
│   ├── video_engine.py          # FFmpeg Ken Burns, sidechain audio ducking & stitcher
│   └── pipeline.py              # Master CLI orchestrator with checkpoint resume
├── tests/
│   ├── test_normalizer.py
│   ├── test_voice_engine.py
│   ├── test_asset_harvester.py
│   ├── test_video_engine.py
│   └── test_end_to_end_mock.py
├── data/
│   └── projects/                # Working directory for generated projects
├── requirements.txt
└── README.md
```

---

## 3. Data Contracts & Schemas

### 3.1 `project_plan.json` (Stage 1 Output)
```json
{
  "project_id": "alexandria_rise_fall",
  "topic": "The Destruction of the Library of Alexandria",
  "target_duration_minutes": 12,
  "visual_style_anchor": "historical archival paintings, ancient papyrus textures, cinematic dramatic reenactment",
  "bgm_mood": "somber, mysterious, scholarly",
  "chapters": [
    {
      "chapter_index": 1,
      "title": "The Sanctuary of Scrolls",
      "target_word_count": 300,
      "target_duration_seconds": 120,
      "core_tension": "The unprecedented policy of confiscating every book that entered the port.",
      "narrative_beats": [
        "Ptolemy the First founds the museum and library around three hundred BC.",
        "Customs officials inspect every ship docking in Alexandria harbor.",
        "Original scrolls are seized; owners receive only cheap copies while originals are kept."
      ],
      "visual_palette": ["ancient harbor", "papyrus scrolls", "classical scholars", "mediterranean coast"],
      "voice_tone": "wonder, scholarly, deliberate"
    }
  ]
}
```

### 3.2 `chapter_script.json` (Stage 2 Output)
```json
{
  "chapter_index": 1,
  "chapter_title": "The Sanctuary of Scrolls",
  "sentences": [
    {
      "sentence_index": 0,
      "raw_text": "In 300 BC, the Mediterranean witnessed an intellectual ambition unlike anything seen before.",
      "spoken_text": "In three hundred BC, the Mediterranean witnessed an intellectual ambition unlike anything seen before.",
      "visual_query": "ancient mediterranean sea historical map",
      "motion_type": "zoom_in",
      "pause_after_seconds": 0.4
    },
    {
      "sentence_index": 1,
      "raw_text": "Every ship docking at Alexandria was boarded by armed guards with a single objective: confiscate every written work on board.",
      "spoken_text": "Every ship docking at Alexandria was boarded by armed guards with a single objective: confiscate every written work on board.",
      "visual_query": "alexandria harbor ancient lighthouse ships",
      "motion_type": "pan_left_to_right",
      "pause_after_seconds": 0.5
    }
  ]
}
```

### 3.3 `alignment.json` (Stage 4 Output)
```json
{
  "project_id": "alexandria_rise_fall",
  "chapter_index": 1,
  "audio_file": "data/projects/alexandria_rise_fall/chapters/ch01/audio_mastered.wav",
  "total_duration": 124.60,
  "segments": [
    {
      "sentence_index": 0,
      "text": "In three hundred BC, the Mediterranean witnessed an intellectual ambition unlike anything seen before.",
      "start": 0.00,
      "end": 6.85,
      "duration": 6.85,
      "pause_after": 0.40,
      "asset_path": "data/projects/alexandria_rise_fall/chapters/ch01/assets/asset_00.mp4",
      "asset_type": "video",
      "motion_type": "zoom_in"
    },
    {
      "sentence_index": 1,
      "text": "Every ship docking at Alexandria was boarded by armed guards with a single objective: confiscate every written work on board.",
      "start": 7.25,
      "end": 15.10,
      "duration": 7.85,
      "pause_after": 0.50,
      "asset_path": "data/projects/alexandria_rise_fall/chapters/ch01/assets/asset_01.jpg",
      "asset_type": "image",
      "motion_type": "pan_left_to_right"
    }
  ]
}
```

---

## 4. Module Specifications

### Module 1: `config_loader.py` & `text_normalizer.py`
* **Dependencies**: `pydantic-settings`, `pyyaml`, `num2words`.
* **Config Attributes**:
  * `DEEPSEEK_API_KEY`: API key for DeepSeek-V3.
  * `DEEPSEEK_BASE_URL`: default `"https://api.deepseek.com/v1"`.
  * `QWEN_TTS_GRADIO_URL`: default `"http://127.0.0.1:7860/"`.
  * `REF_VOICE_AUDIO_PATH`: local path to clean reference `.wav`.
  * `REF_VOICE_TRANSCRIPT`: verbatim text spoken in reference sample.
  * `PEXELS_API_KEY`: string.
  * `PIXABAY_API_KEY`: string (secondary stock provider).
  * `AI_IMAGE_PROVIDER`: `"none"`, `"pollinations"` (free), or `"comfyui"` (local).
  * `OUTPUT_DIR`: default `./data/projects`.
* **Text Normalizer Functions**:
  * `expand_numbers(text: str) -> str`: Converts digits (`"1945"` $\to$ `"nineteen forty-five"`, `"$4.5M"` $\to$ `"four point five million dollars"`, `"30%"` $\to$ `"thirty percent"`).
  * `sanitize_phonetics(text: str) -> str`: Removes brackets, markdown asterisks, em-dashes, and non-ASCII characters; replaces with natural commas/ellipses for breathing cadences.

---

### Module 2 & 3: `planner.py` & `drafter.py` (Unified DeepSeek Engine)
* **Economics**: Powered by DeepSeek-V3 ($0.15–$0.30/1M input, $0.60–$1.20/1M output, prefix cache hits at $0.003/1M).
* **Benefits**:
  * Frees 100% of GPU VRAM for local Qwen-TTS and FFmpeg.
  * Generates high-register, human-sounding documentary narration without 7B clichés.
  * Completes full 15-minute script generation in <10 seconds for <$0.01.
* **Planning Function**: `generate_project_plan(topic: str, duration_min: int, tone: str) -> ProjectPlan`
  * Chapters calculated dynamically: `max(3, round(duration_min / 2.0))`.
  * Generates visual palettes and tension beats.
* **Drafting Function**: `draft_chapter_script(chapter: ChapterPlan, visual_style: str, prior_summary: str) -> ChapterScript`
  * Prompts DeepSeek to output structured sentences with paired visual queries.
  * Each sentence is assigned a dynamic motion direction (`zoom_in`, `zoom_out`, `pan_left_to_right`, `pan_right_to_left`).
  * Automatic post-processing pass through `text_normalizer.py`.

---

### Module 4: `voice_engine.py` (Qwen-TTS + Audio Mastering)
* **Goal**: High-fidelity, click-free vocal synthesis with strict LUFS mastering.
* **Anti-Click & Audio Conditioning**:
  ```python
  def apply_edge_fades(audio_data: np.ndarray, sr: int, fade_ms: int = 10) -> np.ndarray:
      """Applies a smooth cosine fade-in and fade-out to prevent zero-crossing pops."""
      fade_samples = int(sr * (fade_ms / 1000.0))
      if len(audio_data) < 2 * fade_samples:
          return audio_data
      fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
      fade_out = 0.5 * (1 - np.cos(np.linspace(np.pi, 0, fade_samples)))
      audio_data[:fade_samples] *= fade_in
      audio_data[-fade_samples:] *= fade_out
      return audio_data
  ```
* **Loudness Normalization**: Master audio using `pyloudnorm` or FFmpeg `loudnorm` filter to EBU R128 standard (-14.0 LUFS, -1.0 dB True Peak).
* **Metadata Export**: Records `sentence_index`, `duration_seconds`, and silence gap to `clip_durations.json` during synthesis.
* **Health Check & Retry**: Pings Gradio endpoint `/info` before batch run; retries failed sentences up to 3 times with exponential backoff.

---

### Module 5: `aligner.py` (Deterministic Mathematical Alignment)
* **Goal**: Flawless timestamp and SRT synchronization without speech-to-text hallucination risks.
* **Logic**:
  * Reads the exact array lengths from `voice_engine.py`.
  * Computes:
    $$\text{Start}_0 = 0.0$$
    $$\text{End}_i = \text{Start}_i + \text{Duration}_i$$
    $$\text{Start}_{i+1} = \text{End}_i + \text{Pause}_i$$
  * Outputs `alignment.json` directly linking each sentence to its timestamp range and designated visual asset.
  * Emits standard YouTube-ready `.srt` subtitle file with maximum 2-line formatting.

---

### Module 6: `asset_harvester.py` (Multi-Source Resilient Harvester)
* **Goal**: Acquire high-quality 16:9 media for every sentence query without crashing on rate limits or obscure topics.
* **Search Hierarchy**:
  1. **Pexels Video API** (orientation: `landscape`, min duration $\ge 3\text{s}$).
  2. **Pixabay Video & Image API** (handles stock overflow).
  3. **Wikimedia Commons API** (historical public domain artwork/scans).
  4. **AI Generation Fallback** (Pollinations.ai free API or local ComfyUI) when searching historical/ancient/conceptual topics where stock footage does not exist.
* **Asset Transformation**:
  * If video clip duration < sentence duration: apply ping-pong loop or slow-mo via FFmpeg `setpts`.
  * If image aspect ratio $\ne$ 16:9: smart-pad with blurred ambient background or crop to 1920×1080.
* **Caching & Rate Limits**:
  * Local asset store under `data/cache/assets/` keyed by SHA-256 hash of query.
  * Automatically respects API rate-limits with backoff.

---

### Module 7: `video_engine.py` (FFmpeg Dynamic Ken Burns & Audio Ducking)
* **Goal**: Assemble chapter videos and final master MP4 with broadcast-quality visuals and audio.
* **Dynamic Ken Burns Motions** (Eliminates monotonic center-zoom):
  * **Zoom In**: Scales from $1.0\times \to 1.15\times$ toward center.
  * **Zoom Out**: Scales from $1.15\times \to 1.0\times$ from center.
  * **Pan Left $\to$ Right**: Horizontal slide across 16:9 frame.
  * **Pan Right $\to$ Left**: Reverse horizontal slide.
* **Safe Upscaling Filtergraph** (Replaces memory-leaking `scale=8000`):
  * Upscale image to $3840\times2160$ (4K), apply smooth subpixel crop/pan, downscale to $1920\times1080$.
* **Audio Ducking**:
  * Automatically loads ambient background music (BGM) from `assets/bgm/`.
  * Applies FFmpeg `sidechaincompress` so BGM lowers by 14dB whenever narration is active, smoothly recovering during pauses.
* **Lossless Chapter Stitching**:
  * All intermediate segments rendered strictly to: `1920x1080`, `30fps CFR`, `h264_nvenc` (or `libx264`), `yuv420p`, `48000Hz stereo PCM`.
  * Final chapter concatenation performed via FFmpeg concat demuxer (`-f concat -safe 0 -c copy`), preventing timestamp drift.

---

### Module 8: `pipeline.py` (Antigravity Master Orchestrator)
* **CLI Interface**:
  ```bash
  python src/pipeline.py --topic "The Bronze Age Collapse" --duration 15 --tone "investigative"
  ```
* **Checkpoint & Resume Flags**:
  * `--resume-from [plan|script|voice|assets|render]`
  * Granular sentence-level resume: skips existing `.wav` and downloaded assets if already cached on disk.
* **Dry Run Mode**:
  * `--dry-run`: Generates plan, script, and tests asset search queries without running TTS or FFmpeg renders.

---

## 5. Coding Agent Task Checklist (Execution Roadmap)

- [x] **Phase 1: Environment, Dependencies & Text Normalizer**
  - Create `requirements.txt` (`pydantic-settings`, `pyyaml`, `num2words`, `gradio_client`, `soundfile`, `numpy`, `requests`, `pyloudnorm`).
  - Implement `src/config_loader.py` and `config/settings.yaml`.
  - Implement `src/text_normalizer.py` with full unit tests for dates, currencies, and acronyms.

- [x] **Phase 2: DeepSeek Macro-Planner & Chapter Drafter**
  - Implement `src/planner.py` using DeepSeek-V3 API with strict Pydantic JSON validation.
  - Implement `src/drafter.py` generating sentence arrays, visual queries, and camera motions.
  - Test end-to-end generation from prompt to validated `chapter_script.json`.

- [x] **Phase 3: Voice Engine & Mathematical Alignment**
  - Implement `src/voice_engine.py` connecting to Qwen-TTS Gradio endpoint with edge cosine fades.
  - Add EBU R128 loudness normalization and duration metadata recording.
  - Implement `src/aligner.py` with zero-drift mathematical timestamping and `.srt` exporter.

- [x] **Phase 4: Multi-Source Asset Harvester**
  - Implement `src/asset_harvester.py` with Pexels, Pixabay, Wikimedia, and AI image fallback.
  - Add asset hash-caching and rate-limit retry logic.
  - Implement aspect ratio padding/cropping utility for non-16:9 images.

- [x] **Phase 5: FFmpeg Video Engine with Ken Burns & Audio Ducking**
  - Implement `src/video_engine.py` with 4 distinct Ken Burns camera motions.
  - Implement BGM mixing with sidechain audio ducking.
  - Implement chapter assembly and final lossless concat demuxer.

- [x] **Phase 6: CLI Runner & End-to-End Dry Run**
  - Build `src/pipeline.py` with granular `--resume-from` checkpoint flags.
  - Conduct full 1-chapter test run and inspect final rendered MP4.




---

## 6. Hardware & Long-Form Scalability Guidelines (Up to 60 Minutes)

1. **VRAM Freedom**: Because text planning and drafting are handled by DeepSeek API, local GPU VRAM is 100% reserved for Qwen-TTS voice synthesis and NVENC hardware video encoding (`-c:v h264_nvenc`).
2. **Chunked Rendering**: For videos over 15 minutes, never render 800 clips in a single FFmpeg command. Render each chapter into an intermediate chapter MP4, then concatenate chapter MP4s losslessly with `-c copy`.
3. **Storage Hygiene**: Clean up intermediate `.ts` and raw frame buffers post-render; retain only `audio_mastered.wav`, final assets, and chapter `.mp4` files.
4. **Deterministic Reproducibility**: All intermediate JSON files (`project_plan.json`, `chapter_script.json`, `alignment.json`) remain cached. If an FFmpeg styling parameter is tweaked, neither LLM calls nor TTS synthesis need to be re-run.
