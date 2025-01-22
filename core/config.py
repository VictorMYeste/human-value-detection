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
    "presence": {
        "pretrained_model": "microsoft/deberta-base",
        "labels": ["Presence"],
        "custom_stopwords": [
            # Reporting Verbs (different tenses and forms)
            "say", "says", "said", "saying",
            "report", "reports", "reported", "reporting",
            "state", "states", "stated", "stating",
            "tell", "tells", "told", "telling",
            "claim", "claims", "claimed", "claiming",
            "announce", "announces", "announced", "announcing",
            "note", "notes", "noted", "noting",
            "mention", "mentions", "mentioned", "mentioning",
            "describe", "describes", "described", "describing",
            "publish", "publishes", "published", "publishing",
            "reveal", "reveals", "revealed", "revealing",
            "explain", "explains", "explained", "explaining",
            "quote", "quotes", "quoted", "quoting",
            "comment", "comments", "commented", "commenting",

            # Journalistic phrases
            "allege", "alleges", "alleged", "alleging",
            "interview", "interviews", "interviewed", "interviewing",
            "source", "sources",
            "article", "articles",
            "media", "journalism", "news", "press",
            "official", "officials",
            "reported", "reportedly",
            "coverage", "broadcast", "headline",

            # General non-moral contextual terms
            "situation", "case", "problem", "issue", "matter",
            "people", "person", "individual", "group", "community",
            "leaders", "authorities", "organization", "institutions"
        ]
    },
    "growth_selfprotection": {
        "pretrained_model": "microsoft/deberta-base",
        "labels": ["Growth Anxiety-Free", "Self-Protection Anxiety-Avoidance"]
    },
    "social_personal_focus": {
        "pretrained_model": "microsoft/deberta-base",
        "labels": ["Social Focus", "Personal Focus"]
    },
}