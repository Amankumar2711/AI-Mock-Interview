"""
registry.py — process-level model singleton registry.


"""

import threading

import spacy
import torch


class IsolatedUnsafeLoad:
    """
    Temporarily allows weights_only=False specifically for older
    HuggingFace/Pyannote model checkpoints. Avoids overriding
    torch.load globally for the entire application lifecycle.
    """

    def __enter__(self):
        self.orig_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return self.orig_load(*args, **kwargs)

        torch.load = patched_load

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.load = self.orig_load


class ModelRegistry:
    """
    Process-level singleton store for heavyweight ML models.

    All get_* methods are thread-safe via double-checked locking on
    _lock. The fast path (model already loaded) never acquires the
    lock, keeping inference-time overhead near zero. The slow path
    (first load) serialises competing callers so the loader_func is
    called exactly once per model per process, regardless of how many
    gevent greenlets or OS threads call concurrently.
    """

    # One lock shared by all class-level model slots.
    # A single lock is sufficient because model loads are rare
    # (once at warmup) and the critical section is short.
    _lock = threading.Lock()

    _spacy_nlp = None
    _emotion_classifier = None
    _whisper_asr = None
    _wav2vec2 = None
    _semantic_model = None       # SentenceTransformer for semantic/technical eval

    @classmethod
    def get_spacy(cls):
        # Fast path — no lock needed for a read of an already-set reference.
        if cls._spacy_nlp is not None:
            return cls._spacy_nlp
        with cls._lock:
            # Re-check inside lock: another thread may have loaded while we waited.
            if cls._spacy_nlp is None:
                print("ModelRegistry: Loading Spacy model...")
                try:
                    cls._spacy_nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("ModelRegistry: Spacy model 'en_core_web_sm' not found. Downloading...")
                    from spacy.cli import download
                    download("en_core_web_sm")
                    cls._spacy_nlp = spacy.load("en_core_web_sm")
        return cls._spacy_nlp

    @classmethod
    def get_emotion_model(cls, loader_func):
        if cls._emotion_classifier is not None:
            return cls._emotion_classifier
        with cls._lock:
            if cls._emotion_classifier is None:
                print("ModelRegistry: Loading Emotion model...")
                with IsolatedUnsafeLoad():
                    cls._emotion_classifier = loader_func()
        return cls._emotion_classifier

    @classmethod
    def get_whisper(cls, loader_func):
        if cls._whisper_asr is not None:
            return cls._whisper_asr
        with cls._lock:
            if cls._whisper_asr is None:
                print("ModelRegistry: Loading Whisper model...")
                with IsolatedUnsafeLoad():
                    cls._whisper_asr = loader_func()
        return cls._whisper_asr

    @classmethod
    def get_wav2vec2(cls, loader_func):
        """Slot for Wav2Vec2 model — same double-checked locking pattern."""
        if cls._wav2vec2 is not None:
            return cls._wav2vec2
        with cls._lock:
            if cls._wav2vec2 is None:
                print("ModelRegistry: Loading Wav2Vec2 model...")
                with IsolatedUnsafeLoad():
                    cls._wav2vec2 = loader_func()
        return cls._wav2vec2

    @classmethod
    def get_semantic_model(cls, loader_func):
        """Slot for SentenceTransformer semantic embedding model.

        Called from technical_eval/core_tech/semantic_eval.py so the
        model is loaded exactly once per process and pinned in RAM,
        rather than being re-created lazily on every cold-start task.
        """
        if cls._semantic_model is not None:
            return cls._semantic_model
        with cls._lock:
            if cls._semantic_model is None:
                print("ModelRegistry: Loading SentenceTransformer semantic model...")
                cls._semantic_model = loader_func()
        return cls._semantic_model