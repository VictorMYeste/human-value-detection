import re
from typing import Dict, List, Tuple
from core.config import LEXICON_PATHS, SCHWARTZ_VALUE_LEXICON
from core.utils import validate_file, skip_invalid_line, read_file_lines
import logging
logger = logging.getLogger("HVD")

# ========================================================
# EMBEDDINGS LOADING
# ========================================================

def load_embeddings(lexicon: str) -> tuple[dict, int]:
    lexicon_embeddings = {}
    num_categories = 0
    if lexicon:
        lexicon_path = LEXICON_PATHS.get(lexicon)
        lexicon_embeddings, num_categories = load_lexicon(lexicon, lexicon_path)
    return lexicon_embeddings, num_categories


# ========================================================
# EMBEDDINGS LOADING FUNCTIONS
# ========================================================

def load_vad_embeddings(path: str) -> dict[str, dict[str, float]]:
    """Load NRC VAD lexicon into a dictionary."""
    validate_file(path)
    embeddings = {}
    for line in read_file_lines(path, skip_header=True):
        word, valence, arousal, dominance = line.strip().split("\t")
        embeddings[word] = {
            "valence": float(valence),
            "arousal": float(arousal),
            "dominance": float(dominance)
            }
    logger.debug(f"Loaded VAD embeddings: {len(embeddings)} words.")
    return embeddings

def load_emolex_embeddings(path: str) -> dict[str, dict[str, float]]:
    """Load EmoLex lexicon into a dictionary."""
    validate_file(path)
    # Define the emotions in the order they appear in the file
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]
    embeddings = {}
    for line in read_file_lines(path, skip_header=True):
        word, emotion, score = line.strip().split("\t")
        if word not in embeddings:
            embeddings[word] = [0] * len(emotions)
        if emotion in emotions:
            emotion_idx = emotions.index(emotion)
            embeddings[word][emotion_idx] = int(score)
    logger.debug(f"Loaded EmoLex embeddings: {len(embeddings)} words.")
    return embeddings

def load_emotionintensity_embeddings(path: str) -> dict[str, dict[str, float]]:
    """Load NRC Emotion Intensity Lexicon into a dictionary."""
    validate_file(path)
    embeddings = {}
    for line in read_file_lines(path, skip_header=True):
        word, emotion, score = line.strip().split("\t")
        score = float(score)  # Intensity scores are continuous
        if word not in embeddings:
            embeddings[word] = {emotion: 0.0 for emotion in ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]}
        embeddings[word][emotion] = score
    logger.debug(f"Loaded Emotion Intensity embeddings: {len(embeddings)} words.")
    return embeddings

def load_worrywords_embeddings(path: str) -> dict[str, dict[str, float]]:
    """Load NRC WorryWords Lexicon into a dictionary."""
    validate_file(path)
    embeddings = {}
    for line in read_file_lines(path, skip_header=True):
        parts = line.strip().split("\t")
        term = parts[0]
        mean_score = float(parts[1])  # Mean score column
        embeddings[term.lower()] = mean_score
    logger.debug(f"Loaded WorryWords embeddings: {len(embeddings)} words.")
    return embeddings

def load_liwc_embeddings(path: str) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """Load LIWC dictionary and process it into usable categories and scores."""
    validate_file(path)
    embeddings, category_names = {}, {}
    lines = read_file_lines(path)
    
    # Skip the first line if it starts with `%`
    if lines[0].strip() == "%":
        lines = lines[1:]

    # Find the `%` marker that separates categories and words
    divider_index = next((i for i, line in enumerate(lines) if line.strip() == "%"), None)
    if divider_index is None:
        raise ValueError("LIWC file format error: '%' marker not found.")

    # Parse Category Section
    category_section = lines[:divider_index]  # Include all lines up to the `%` marker
    # Parse category definitions
    for line in category_section:
        parts = line.strip().split("\t")
        if len(parts) == 2:  # Ensure there are exactly two parts (ID and name)
            try:
                category_id = int(parts[0])  # First column is the category ID
                category_name = parts[1]    # Second column is the category name
                category_names[category_id] = category_name
            except ValueError:
                skip_invalid_line(line, "invalid category")

    # Parse Word Section
    word_section = lines[divider_index + 1:]
    # Parse word entries
    for line in word_section:
        parts = line.strip().split("\t")
        if len(parts) < 2:  # Ensure there is at least a word and one category
            skip_invalid_line(line, "invalid word")
            continue

        word = parts[0].lower()  # Normalize the word to lowercase
        try:
            categories = [int(category) for category in parts[1:] if category.strip().isdigit()]
            if categories:
                embeddings[word] = categories
            else:
                skip_invalid_line(line, "non-numeric categories")
        except ValueError:
            skip_invalid_line(line, "invalid category IDs")
        
    # Debugging output
    #logger.debug("Category Names Sample:", list(category_names.items())[:5])
    #logger.debug("Embeddings Sample:", list(embeddings.items())[:5])
    logger.debug(f"Loaded LIWC embeddings: {len(embeddings)} words, {len(category_names)} categories.")
    return embeddings, category_names

def load_mfd_embeddings(path: str) -> dict[str, dict[str, float]]:
    """Load the Moral Foundations Dictionary into a usable format."""
    validate_file(path)
    embeddings = {}
    current_category = None

    lines = read_file_lines(path)

    for line in lines:
        line = line.strip()

        if not line or line.startswith("[") or line.startswith(";"):
            continue
        
        # Detect new category headers (e.g., "CARE.VIRTUE")
        category_match = re.match(r'^(\S+)$', line)
        if category_match:
            current_category = category_match.group(1)
            continue

        # Parse words with scores (e.g., "ALLEVIATE (1)")
        if current_category:
            word_match = re.match(r'^([A-Z_]+(?:\s?[A-Z]+)*)\s*\((\d+)\)$', line)
            if word_match:
                word = word_match.group(1).strip()
                embeddings[word] = current_category
    logger.debug(f"Loaded MFD embeddings: {len(embeddings)} words.")
    return embeddings

def load_schwartz_embeddings(path=None) -> tuple[dict, int]:
    """Return the hardcoded Schwartz lexicon and category count."""
    return SCHWARTZ_VALUE_LEXICON, len(SCHWARTZ_VALUE_LEXICON)

# ========================================================
# MAIN LEXICON LOADING
# ========================================================

EMBEDDING_PARSERS = {
    "VAD": {"function": load_vad_embeddings, "num_categories": 3},
    "EmoLex": {"function": load_emolex_embeddings, "num_categories": 10},
    "EmotionIntensity": {"function": load_emotionintensity_embeddings, "num_categories": 8},
    "WorryWords": {"function": load_worrywords_embeddings, "num_categories": 1},
    "LIWC": {"function": load_liwc_embeddings, "is_liwc": True},
    "MFD": {"function": load_mfd_embeddings, "num_categories": 10},
    "Schwartz": {"function": load_schwartz_embeddings, "num_categories": len(SCHWARTZ_VALUE_LEXICON)},
}

def load_lexicon(lexicon_name: str, path: str) -> tuple[dict, int]:
    parser = EMBEDDING_PARSERS.get(lexicon_name)
    if not parser:
        raise ValueError(f"Unknown lexicon: {lexicon_name}")
    
    # Handle complex parsers like LIWC with category mappings
    if parser.get("is_liwc"):
        embeddings, category_names = parser["function"](path)
        return embeddings, len(category_names)
    
    # Default handling for simple parsers
    embeddings = parser["function"](path)
    logger.debug(f"Loaded {len(embeddings)} embeddings from {path}")
    return embeddings, parser["num_categories"]