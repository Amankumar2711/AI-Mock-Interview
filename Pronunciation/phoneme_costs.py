"""
ARPABET phoneme classification + substitution/deletion cost matrix.

Used by the weighted Levenshtein aligner to penalise pronunciation errors
proportionally to how perceptually/articulatorily severe they are —
instead of treating every mismatch as cost 1.

Design (per user spec):
  - Vowel <-> Vowel substitution, phonetically close pair   -> cheap (0.2-0.5)
  - Vowel <-> Vowel substitution, distant pair               -> moderate (1.0)
  - Consonant deletion (esp. stops/fricatives)               -> expensive (1.5-2.0)
  - Consonant <-> Vowel substitution                          -> very expensive (3.0)
  - Consonant <-> Consonant, same manner/place                -> cheap-moderate
  - Consonant <-> Consonant, different manner/place           -> expensive
  - Insertions                                                -> moderate (1.0)
  - Identical phonemes                                        -> 0
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional

# ── ARPABET phoneme sets (stress digits already stripped upstream) ──────────

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
}

CONSONANTS = {
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M",
    "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
}

# Vowel articulatory groups — vowels in the same group are "phonetically close"
# (close in tongue height/backness, common accent-driven confusions).
VOWEL_GROUPS = {
    # front vowels
    "IY": "front_high", "IH": "front_high",
    "EH": "front_mid", "EY": "front_mid", "AE": "front_low",
    # central vowels
    "AH": "central", "ER": "central",
    # back vowels
    "UW": "back_high", "UH": "back_high",
    "OW": "back_mid", "AO": "back_mid", "AA": "back_low",
    # diphthongs
    "AY": "diphthong", "AW": "diphthong", "OY": "diphthong",
}

# Common Indian-English accent vowel confusion pairs — treated as cheap.
# e.g. IH/IY ("bit"/"beat"), EH/AE ("bet"/"bat"), AA/AO ("cot"/"caught")
ACCENT_TOLERANT_VOWEL_PAIRS: FrozenSet[FrozenSet[str]] = frozenset({
    frozenset({"IH", "IY"}),
    frozenset({"EH", "AE"}),
    frozenset({"AA", "AO"}),
    frozenset({"AH", "AA"}),
    frozenset({"UH", "UW"}),
    frozenset({"EH", "EY"}),
    frozenset({"AO", "OW"}),
})

# Consonant articulatory features: (place, manner)
CONSONANT_FEATURES: Dict[str, tuple] = {
    "P":  ("bilabial", "stop"),       "B":  ("bilabial", "stop"),
    "T":  ("alveolar", "stop"),       "D":  ("alveolar", "stop"),
    "K":  ("velar", "stop"),          "G":  ("velar", "stop"),
    "F":  ("labiodental", "fricative"), "V": ("labiodental", "fricative"),
    "TH": ("dental", "fricative"),    "DH": ("dental", "fricative"),
    "S":  ("alveolar", "fricative"),  "Z":  ("alveolar", "fricative"),
    "SH": ("postalveolar", "fricative"), "ZH": ("postalveolar", "fricative"),
    "HH": ("glottal", "fricative"),
    "CH": ("postalveolar", "affricate"), "JH": ("postalveolar", "affricate"),
    "M":  ("bilabial", "nasal"),      "N": ("alveolar", "nasal"), "NG": ("velar", "nasal"),
    "L":  ("alveolar", "liquid"),     "R": ("alveolar", "liquid"),
    "W":  ("labiovelar", "glide"),    "Y": ("palatal", "glide"),
}

# Phonemes that, if deleted, severely hurt intelligibility (stops + fricatives
# carry most lexical contrast information).
HIGH_IMPACT_CONSONANTS = {
    "P", "B", "T", "D", "K", "G",            # stops
    "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "CH", "JH",  # fricatives/affricates
}


# ── Cost functions ────────────────────────────────────────────────────────

def is_vowel(p: str) -> bool:
    return p in VOWELS


def is_consonant(p: str) -> bool:
    return p in CONSONANTS


def substitution_cost(expected: str, actual: str, weights: "PhonemeCostWeights") -> float:
    """Cost of substituting `expected` -> `actual`."""
    if expected == actual:
        return 0.0

    exp_vowel, act_vowel = is_vowel(expected), is_vowel(actual)
    exp_cons, act_cons = is_consonant(expected), is_consonant(actual)

    # ── Vowel <-> Vowel ───────────────────────────────────────────────────
    if exp_vowel and act_vowel:
        pair = frozenset({expected, actual})
        if pair in ACCENT_TOLERANT_VOWEL_PAIRS:
            return weights.vowel_close_substitution
        if VOWEL_GROUPS.get(expected) == VOWEL_GROUPS.get(actual):
            return weights.vowel_close_substitution
        return weights.vowel_distant_substitution

    # ── Consonant <-> Vowel  (most severe — articulation failure) ──────────
    if (exp_cons and act_vowel) or (exp_vowel and act_cons):
        return weights.consonant_vowel_substitution

    # ── Consonant <-> Consonant ──────────────────────────────────────────────
    if exp_cons and act_cons:
        exp_feat = CONSONANT_FEATURES.get(expected)
        act_feat = CONSONANT_FEATURES.get(actual)
        if exp_feat and act_feat:
            same_place = exp_feat[0] == act_feat[0]
            same_manner = exp_feat[1] == act_feat[1]
            if same_place and same_manner:
                return weights.consonant_close_substitution
            if same_place or same_manner:
                return weights.consonant_moderate_substitution
        return weights.consonant_distant_substitution

    # Unknown phoneme symbols — moderate default
    return weights.default_substitution


def deletion_cost(expected: str, weights: "PhonemeCostWeights") -> float:
    """Cost of `expected` phoneme being missing entirely from the actual sequence."""
    if expected in HIGH_IMPACT_CONSONANTS:
        return weights.consonant_deletion
    if is_consonant(expected):
        return weights.consonant_deletion * 0.6   # glides/liquids/nasals — still costly
    if is_vowel(expected):
        return weights.vowel_deletion
    return weights.default_deletion


def insertion_cost(actual: str, weights: "PhonemeCostWeights") -> float:
    """Cost of an extra phoneme appearing that wasn't expected."""
    return weights.insertion


class PhonemeCostWeights:
    """
    Tunable cost matrix. Defaults follow the spec:
      - vowel-vowel close substitutions: cheap (forgive accent variation)
      - consonant deletions: expensive
      - consonant<->vowel substitutions: maximum cost
    """

    def __init__(
        self,
        vowel_close_substitution: float = 0.3,
        vowel_distant_substitution: float = 1.0,
        consonant_close_substitution: float = 0.5,
        consonant_moderate_substitution: float = 1.2,
        consonant_distant_substitution: float = 1.8,
        consonant_vowel_substitution: float = 3.0,
        consonant_deletion: float = 1.8,
        vowel_deletion: float = 1.0,
        insertion: float = 1.0,
        default_substitution: float = 1.0,
        default_deletion: float = 1.0,
    ):
        self.vowel_close_substitution = vowel_close_substitution
        self.vowel_distant_substitution = vowel_distant_substitution
        self.consonant_close_substitution = consonant_close_substitution
        self.consonant_moderate_substitution = consonant_moderate_substitution
        self.consonant_distant_substitution = consonant_distant_substitution
        self.consonant_vowel_substitution = consonant_vowel_substitution
        self.consonant_deletion = consonant_deletion
        self.vowel_deletion = vowel_deletion
        self.insertion = insertion
        self.default_substitution = default_substitution
        self.default_deletion = default_deletion

    @property
    def max_cost(self) -> float:
        """Highest possible single-edit cost — used to normalise scores."""
        return max(
            self.consonant_vowel_substitution,
            self.consonant_deletion,
            self.consonant_distant_substitution,
        )
