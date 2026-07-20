"""
Configuration for the simplified pronunciation pipeline.
No audio preprocessing, no MFA -- raw WAV in, weighted-Levenshtein score out.

Device policy:
  - WhisperX (ASRStage) targets GPU — see STT/asr_stage.py for device resolution.
  - Wav2Vec2 is strictly CPU-only. VRAM is reserved for WhisperX and the LLM.
"""

import os
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV2VEC2_PATH = os.path.join(BASE_DIR, "wav2vec2_base")


@dataclass
class WhisperConfig:
    model_size: str = "base"        # tiny | base | small | medium | large-v2
    device: str = "cpu"             # resolved to "cuda" at runtime in ASRStage if available
    compute_type: str = "int8"      # "float16" on GPU, "int8" on CPU (ASRStage picks automatically)
    batch_size: int = 4
    language: str = "en"
    min_word_confidence: float = 0.4

    # Pyannote VAD inside WhisperX breaks on Windows (torchcodec/FFmpeg).
    # These options soften Whisper's own no-speech rejection so it doesn't
    # silently return empty transcripts on quiet/short clips.
    no_speech_threshold: float = 0.3
    log_prob_threshold: float = -1.0


@dataclass
class Wav2Vec2Config:
    model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"  # phoneme CTC model
    model_path: str = WAV2VEC2_PATH
    # Wav2Vec2 is always CPU — do not change this.
    device: str = "cpu"


@dataclass
class G2PConfig:
    backend: str = "g2p_en"        # g2p_en | phonemizer
    language: str = "en-us"
    stress_markers: bool = False


@dataclass
class ScoringConfig:
    """
    Reserved for future use / backward compatibility.
    Actual phoneme substitution/deletion/insertion costs now live in
    phoneme_costs.PhonemeCostWeights, which is passed separately to
    GOPScorer so it can be tuned independently per deployment
    (e.g. looser vowel tolerance for Indian-English candidates).
    """
    pass


@dataclass
class PipelineConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    wav2vec2: Wav2Vec2Config = field(default_factory=Wav2Vec2Config)
    g2p: G2PConfig = field(default_factory=G2PConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


def get_cpu_config() -> PipelineConfig:
    """Default -- Windows / CPU friendly."""
    return PipelineConfig()


def get_gpu_config() -> PipelineConfig:
    """GPU config — WhisperX uses GPU (resolved in ASRStage at runtime).
    Wav2Vec2 remains on CPU regardless of GPU availability.
    """
    cfg = PipelineConfig()
    cfg.whisper.model_size = "large-v2"
    # Note: cfg.whisper.device is overridden in ASRStage to "cuda" if available.
    # Setting it here as a hint for any non-ASRStage consumers.
    cfg.whisper.device = "cuda"
    cfg.whisper.compute_type = "float16"
    cfg.whisper.batch_size = 16
    # Wav2Vec2 stays on CPU — do NOT move it to GPU.
    # cfg.wav2vec2.device is intentionally left as "cpu".
    return cfg
