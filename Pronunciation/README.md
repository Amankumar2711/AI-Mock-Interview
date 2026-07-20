# Pronunciation Assessment Pipeline

Industry-standard, modular pronunciation assessment system.  
Accepts raw browser audio → returns a 0–100 score with per-word, per-phoneme breakdowns.

---

## Architecture

```
Browser Audio (WAV / WebM)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  PREPROCESSING (CPU)                                │
│                                                     │
│  AudioValidator   → mono / 16kHz / float32          │
│  DCOffsetRemover  → remove mic bias                 │
│  VADStage         → Silero / WebRTC                 │
│  DenoiseStage     → DeepFilterNet / SepFormer       │
│  LoudnessNorm     → -23 LUFS / RMS                  │
│  QualityAssessor  → SNR, clipping, reverb, DNSMOS   │
│  AudioChunker     → 10s overlapping segments        │
└─────────────────────────────────────────────────────┘
        │
        ├─────────────────────┐
        ▼ Branch A (text)     ▼ Branch B (audio)
   ASRStage              Wav2Vec2Stage
   (WhisperX)        (moxeeeem phoneme model)
        │                     │
   TextCleaner          Phoneme cleanup
   G2PStage             (CTC + repeat collapse)
   (expected phones)    (actual phones)
        │                     │
        └──────────┬──────────┘
                   ▼
           AlignmentStage (MFA)
                   ▼
              GOPScorer
                   ▼
         PronunciationReport
```

---

## Usage

```python
from pronunciation_pipeline import PronunciationPipeline, get_production_config
import numpy as np

pipeline = PronunciationPipeline(get_production_config())
pipeline.load()   # load all models once at startup

report = pipeline.evaluate(
    audio_samples=samples,       # np.ndarray float32, any sample rate
    sample_rate=16000,
    reference_text="Hello, how are you today?",
    session_id="sess_001",
    candidate_id="cand_42",
)

print(report.overall_score)      # 78.5
print(report.error_summary)      # {"hello": ["AH→EH"], "today": ["T→D"]}
print(report.word_scores[0])     # WordScore(word="hello", score=62.0, ...)
```

### Celery integration (existing stack)

```python
from pronunciation_pipeline.tasks import evaluate_pronunciation_task
import base64

task = evaluate_pronunciation_task.delay(
    audio_b64=base64.b64encode(wav_bytes).decode(),
    sample_rate=16000,
    reference_text="Please read the following sentence.",
    session_id="sess_001",
    candidate_id="cand_42",
)
result = task.get(timeout=60)
# result is a plain dict — safe to store in PostgreSQL / Redis
```

---

## Config presets

| Preset | Whisper | Denoise | VAD | Use case |
|---|---|---|---|---|
| `get_production_config()` | large-v2, fp16 | DeepFilterNet | Silero | GPU worker |
| `get_fast_config()` | base, int8 | None | WebRTC | CPU-only / low latency |
| `get_test_config()` | base, int8 | None | WebRTC | Unit tests |

Override any field:
```python
cfg = get_production_config()
cfg.whisper.model_size = "medium"
cfg.scoring.substitution_penalty = 0.8
```

---

## Module map

```
pronunciation_pipeline/
├── __init__.py              public API surface
├── pipeline.py              PronunciationPipeline orchestrator
├── tasks.py                 Celery task definitions
│
├── config/
│   └── pipeline_config.py   all dataclass configs + presets
│
├── core/
│   ├── models.py            typed result dataclasses (all stages)
│   └── stage_base.py        PipelineStage[In, Out] ABC
│
├── preprocessing/
│   └── stages.py            AudioValidator, DCOffsetRemover, VADStage,
│                            DenoiseStage, LoudnessNormaliser,
│                            QualityAssessor, AudioChunker
│
├── models/
│   └── stages.py            ASRStage (WhisperX), G2PStage,
│                            Wav2Vec2Stage, AlignmentStage (MFA)
│
├── scoring/
│   └── gop_scorer.py        align_sequences, GOPScorer
│
└── tests/
    └── test_pipeline.py     unit tests (no GPU required)
```

---
## NLTK Download
```bash
py -3.12 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```
---
## MFA on Conda
*not in .venv but in different terminal
```bash
conda create -n mfa python=3.12 -y
conda activate mfa
conda install -c conda-forge montreal-forced-aligner -y
```
---
## Check mfa version
```bash
conda run -n mfa mfa version
```

## MFA models download
```bash
conda run -n mfa mfa model download acoustic english_us_arpa
conda run -n mfa mfa model download dictionary english_us_arpa
```
---

## Installing

```bash
pip install -r requirements.txt

# MFA (Montreal Forced Aligner) — required for alignment stage
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

---

## Running tests

```bash
# Unit tests only (no GPU, no models needed)
pytest tests/test_pipeline.py -v -m "not integration"

# Full integration tests (requires models)
pytest tests/test_pipeline.py -v
```

---

## GOP Scoring formula

```
gop_phoneme = confidence_w * phoneme_confidence
            + duration_w   * duration_score
            + posterior_w  * acoustic_posterior

word_score  = 100 × mean(gop_phonemes) × (1 − error_rate × 0.5)
overall     = mean(word_scores)
```

Default weights: `confidence=0.4, duration=0.2, posterior=0.4`  
All weights configurable via `ScoringConfig`.
