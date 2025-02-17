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
    "MFD": "../../lexicons/Moral-Foundations-Dictionary.wmodel",
    "Schwartz": None, # No file, using hardcoded dictionary
    "LIWC-22-training": "../../lexicons/LIWC-22-training-sentences.csv",
    "LIWC-22-validation": "../../lexicons/LIWC-22-validation-sentences.csv",
    "LIWC-22-test": "../../lexicons/LIWC-22-test-sentences.csv",
    "eMFD-training": "../../lexicons/eMFD-training-sentences.csv",
    "eMFD-validation": "../../lexicons/eMFD-validation-sentences.csv",
    "eMFD-test": "../../lexicons/eMFD-test-sentences.csv",
    "MFD-20-training": "../../lexicons/MFD-20-training-sentences.csv",
    "MFD-20-validation": "../../lexicons/MFD-20-validation-sentences.csv",
    "MFD-20-test": "../../lexicons/MFD-20-test-sentences.csv",
    "MJD-training": "../../lexicons/MJD-training-sentences.csv",
    "MJD-validation": "../../lexicons/MJD-validation-sentences.csv",
    "MJD-test": "../../lexicons/MJD-test-sentences.csv"
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

SCHWARTZ_VALUE_LEXICON = {
    "Power": [
        "power", "powers", "strength", "strengths", "control", "controls", 
        "controlling", "dominance", "dominate", "dominating", "dominated", 
        "influence", "influences", "influencing", "authority", "command", 
        "commands", "ruling", "governance", "leadership"
    ],
    "Achievement": [
        "achievement", "achievements", "achieve", "achieving", "achieved", 
        "ambition", "ambitions", "ambitious", "success", "successes", 
        "succeed", "succeeded", "succeeding", "accomplishment", "accomplishments",
        "accomplish", "accomplishing", "accomplished"
    ],
    "Hedonism": [
        "luxury", "luxuries", "pleasure", "pleasures", "delight", "delights",
        "enjoyment", "enjoy", "enjoyed", "enjoying", "satisfaction", "satisfy",
        "satisfied", "satisfying", "indulgence", "indulging", "indulged", 
        "recreation", "fun"
    ],
    "Stimulation": [
        "excitement", "excite", "excited", "exciting", "novelty", "novelties",
        "thrill", "thrills", "thrilling", "adventure", "adventures", 
        "adventurous", "curiosity", "exploration", "exploring", "explored"
    ],
    "Self-direction": [
        "independence", "independent", "independently", "freedom", "free", 
        "liberty", "autonomy", "self-reliance", "self-sufficient", "self-rule",
        "choice", "choices", "choosing", "chosen"
    ],
    "Universalism": [
        "unity", "united", "justice", "equal", "equality", "equity", 
        "fairness", "fair", "fairly", "tolerance", "tolerant", "acceptance",
        "diversity", "inclusion", "humanity", "compassion", "respect"
    ],
    "Benevolence": [
        "kindness", "kind", "charity", "charitable", "mercy", "merciful",
        "generosity", "generous", "support", "supporting", "supported", 
        "help", "helping", "helped", "care", "caring", "selflessness", 
        "selfless"
    ],
    "Tradition": [
        "tradition", "traditions", "traditional", "custom", "customs", 
        "respect", "respecting", "respected", "heritage", "heritages",
        "belief", "beliefs", "ritual", "rituals", "convention", "conventions"
    ],
    "Conformity": [
        "restraint", "restrained", "restraining", "regard", "regarded",
        "consideration", "considering", "considered", "obedience", 
        "obedient", "obey", "obeyed", "obeying", "conventional", 
        "complying", "compliance"
    ],
    "Security": [
        "security", "secure", "secured", "safety", "safe", "safeguard", 
        "safeguarding", "protection", "protect", "protecting", "protected", 
        "stability", "stable", "certainty", "risk-free", "assurance"
    ]
}