# ========================================================
# GLOBAL VARIABLES
# ========================================================

PRETRAINED_MODEL = "microsoft/deberta-base"

LEXICON_PATHS = {
    "VAD": "../../lexicons/NRC-VAD-Lexicon.txt",
    "EmoLex": "../../lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "EmotionIntensity": "../../lexicons/NRC-Emotion-Intensity-Lexicon-v1.txt",
    "WorryWords": "../../lexicons/worrywords-v1.txt",
    "LIWC": "../../lexicons/liwc2015.dic",
    "MFD": "../../lexicons/Moral-Foundations-Dictionary.wmodel"
}

MODEL_CONFIG = {
    "growth_selfprotection": {
        "pretrained_model": "microsoft/deberta-base",
        "labels": ["Growth Anxiety-Free", "Self-Protection Anxiety-Avoidance"]
    },
}