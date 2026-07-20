"""G2P stage — converts transcribed words to expected ARPABET phonemes."""

from __future__ import annotations

import logging
from typing import List

from Pronunciation.config_pron.config import G2PConfig
from Pronunciation.models_pron.models import ASRResult, ExpectedPhonemes

logger = logging.getLogger(__name__)


class G2PStage:
    def __init__(self, config: G2PConfig):
        self.config = config
        self._g2p = None

    def load(self) -> None:
        if self.config.backend == "g2p_en":
            from g2p_en import G2p
            self._g2p = G2p()
        elif self.config.backend == "phonemizer":
            from phonemizer import phonemize
            self._g2p = phonemize
        logger.info("G2P backend: %s", self.config.backend)

    def process(self, asr_result: ASRResult) -> List[ExpectedPhonemes]:
        out: List[ExpectedPhonemes] = []
        for token in asr_result.words:
            out.append(ExpectedPhonemes(word=token.word, phonemes=self._convert(token.word)))
        return out

    def _convert(self, word: str) -> List[str]:
        word = word.lower().strip()
        if not word:
            return []
        try:
            if self.config.backend == "g2p_en":
                phones = self._g2p(word)
                return [
                    p.rstrip("012") if not self.config.stress_markers else p
                    for p in phones
                    if p.isalpha() or (len(p) > 1 and p[:-1].isalpha())
                ]
            elif self.config.backend == "phonemizer":
                result = self._g2p(
                    word, backend="espeak",
                    language=self.config.language,
                    with_stress=self.config.stress_markers,
                )
                return result.strip().split()
        except Exception as e:
            logger.warning("G2P failed for '%s': %s", word, e)
            return []
        return []
