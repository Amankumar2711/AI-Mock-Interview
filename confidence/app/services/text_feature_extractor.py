import spacy
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core_main.registry import ModelRegistry

# FILLER_WORDS list remains unchanged
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "er", "erm", "ah", "hm", "hmm", "mm",
    "eh", "oh", "ooh", "aah",
    "like", "youknow", "actually", "basically", "literally",
    "essentially", "technically", "fundamentally",
    "sortof", "kindof", "kinda", "sorta",
    "right", "okay", "ok", "okayy", "alright",
    "well", "so", "now", "then",
    "anyway", "anyways",
    "mean", "imean",
    "honestly", "frankly", "seriously",
    "obviously", "clearly",
    "simply", "just", "really", "quite", "pretty",
    "perhaps", "maybe", "probably", "possibly",
    "basicallyspeaking",
    "yousee",
    "look", "listen",
    "yeah", "yep", "yup", "nah", "nope",
    "correct", "exactly", "absolutely",
    "indistinct", "inaudible", "unintelligible",
    "mumble", "mumbles", "mumbling",
    "sigh", "sighs", "sighing",
    "pause", "hesitation", "stutter",
    "uhm", "huh"
}

HEDGING_WORDS = {
    "maybe", "perhaps", "probably", "possibly", "potentially",
    "conceivably", "presumably", "apparently", "arguably",
    "reportedly", "allegedly", "supposedly",

    "think", "guess", "believe", "feel", "suppose",
    "reckon", "assume", "suspect", "imagine",
    "estimate", "infer", "speculate", "predict",
    "expect", "anticipate",

    "might", "could", "may", "can", "would", "should",

    "seem", "seems", "seemed",
    "appear", "appears", "appeared",

    "hopefully", "likely", "unlikely",

    "roughly", "approximately", "about",
    "around", "nearly", "almost",

    "somewhat", "rather", "fairly",
    "relatively", "moderately", "partially",

    "generally", "typically", "usually",
    "often", "sometimes", "occasionally",

    "sorta", "kinda", "sort", "kind",

    "basically", "technically",
    "virtually", "practically",

    "guessing", "thinking",
    "believing", "feeling",

    "uncertain", "unsure",
    "doubtful", "questionable",

    "ostensibly", "seemingly",

    "suggest", "suggests",
    "indicate", "indicates",

    "assumed", "estimated",
    "projected", "expected",

    "hope", "wish",
    "preferably"
}
# HEDGING_REGEX = re.compile(r"^(maybe|perhaps|probably|think|guess|might|could|possibly|seem)$")

def extract_text_features(text):
    nlp = ModelRegistry.get_spacy()
    doc = nlp(text)

    words = [
        token.text.lower()
        for token in doc
        if token.is_alpha
    ]

    word_count = len(words)

    filler_count = sum(
        1 for word in words
        if word in FILLER_WORDS
    )

    hedging_count = sum(
        1 for word in words
        if word in HEDGING_WORDS
    )

    return {
        "word_count": word_count,
        "fillers": filler_count,
        "hedging": hedging_count
    }
