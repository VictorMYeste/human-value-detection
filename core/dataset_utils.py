import pandas
import numpy as np
import datasets
import os
from collections import defaultdict
import spacy
import re
from nltk.tokenize import word_tokenize
import transformers
from transformers import AutoModel, AutoTokenizer
import torch
from typing import Optional, Dict, Tuple, List
from core.config import MODEL_CONFIG
from core.utils import validate_args, normalize_token, slice_for_testing
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HVD")
#logger.setLevel(logging.DEBUG)

# ========================================================
# DATASETS PREPARATION
# ========================================================

def prepare_datasets(
    training_path: str,
    validation_path: Optional[str],
    tokenizer: transformers.PreTrainedTokenizer,
    labels: List[str],
    slice_data: bool = False,
    lexicon_embeddings: Optional[Dict] = None,
    num_categories: int = 0,
    previous_sentences: bool = False,
    linguistic_features: bool = False,
    ner_features: bool = False,
    lexicon: str = None,
    custom_stopwords: List[str] = []
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    # Training dataset
    training_dataset = load_dataset(
        training_path,
        tokenizer,
        labels,
        slice_data,
        lexicon_embeddings,
        num_categories,
        previous_sentences,
        linguistic_features,
        ner_features,
        lexicon,
        custom_stopwords
    )

    # Log class distribution
    labels_array = np.array(training_dataset["labels"])
    logger.debug(f"Class distribution: {labels_array.sum(axis=0)}")

    # Validation dataset
    validation_dataset = training_dataset
    if validation_path:
        validation_dataset = load_dataset(
            validation_path,
            tokenizer,
            labels,
            slice_data,
            lexicon_embeddings,
            num_categories,
            previous_sentences,
            linguistic_features,
            ner_features,
            lexicon,
            custom_stopwords
        )
    
    validate_args(labels, training_dataset, validation_dataset)

    logger.debug(f"Training dataset columns: {training_dataset.column_names}")
    logger.debug(f"Validation dataset columns: {validation_dataset.column_names}")
    logger.debug(f"Sample training dataset: {training_dataset[0]}")
    logger.debug(f"Sample validation dataset: {validation_dataset[0]}")

    return training_dataset, validation_dataset

# ========================================================
# DATASET LOADING
# ========================================================

def validate_input_shapes(inputs):
    for key, value in inputs.items():
        if isinstance(value, list) and len(value) == 0:
            logger.error(f"Empty feature detected in {key}")
            raise ValueError(f"Feature {key} is empty")
        elif isinstance(value, list) and any(len(v) == 0 for v in value):
            logger.error(f"Empty elements detected within feature {key}")
            raise ValueError(f"Feature {key} contains empty elements")

def remove_custom_stopwords(text, custom_stopwords):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Remove reporting stopwords
    filtered_tokens = [word for word in tokens if word not in custom_stopwords]

    # Reconstruct sentence
    cleaned_text = " ".join(filtered_tokens)
    
    # Remove extra whitespace or special characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def load_dataset(
    directory,
    tokenizer,
    labels,
    slice_data: bool = False,
    lexicon_embeddings = {},
    num_categories=0,
    previous_sentences: bool = False,
    linguistic_features: bool = False,
    ner_features: bool = False,
    lexicon: str = None,
    custom_stopwords: List[str] = []
):
    """Load dataset and add lexicon embeddings if specified."""
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    labels_file_path = os.path.join(directory, "labels-cat.tsv")
    
    data_frame = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0)

    # Slicing for testing purposes
    if slice_data:
        data_frame = slice_for_testing(data_frame)

    # Fill missing text
    data_frame['Text'] = data_frame['Text'].fillna('')

    # Apply the reporting language removal before tokenization
    if custom_stopwords:
        data_frame['Text'] = data_frame['Text'].apply(lambda text: remove_custom_stopwords(text, custom_stopwords))

    # Tokenize the text
    texts = data_frame["Text"]

    if previous_sentences:
        # Create a new column for concatenated text
        concatenated_texts = []

        for idx in range(len(data_frame)):
            if idx == 0:
                # First sentence, no preceding sentences
                concatenated_texts.append(str(data_frame.iloc[idx]["Text"]))
            elif idx == 1:
                # Second sentence, only one preceding sentence
                concatenated_texts.append(
                    str(data_frame.iloc[idx - 1]["Text"]) + " [SEP] " + str(data_frame.iloc[idx]["Text"])
                )
            else:
                # Concatenate the two preceding sentences with a separator
                concatenated_texts.append(
                    str(data_frame.iloc[idx - 2]["Text"]) + " " +
                    str(data_frame.iloc[idx - 1]["Text"]) + " [SEP] " +
                    str(data_frame.iloc[idx]["Text"])
                )
        
        texts = concatenated_texts
    
    texts = [str(text) for text in texts]
    
    encoded_sentences = tokenizer(texts, padding=True, truncation=True, max_length=512)

    texts = pandas.Series(texts)

    if linguistic_features or ner_features:
        logger.info("Loading en_core_web_sm for extra features")
        nlp = spacy.load("en_core_web_sm")
    
    # Initialize a list to hold combined features
    combined_features = []

    # Compute Linguistic features
    if linguistic_features:
        logger.info("Adding Linguistic features")
        data_frame['Linguistic_Scores'] = texts.apply(lambda text: compute_linguistic_features(text, nlp))
        combined_features.append(data_frame['Linguistic_Scores'].tolist())

    # Compute NER features
    if ner_features:
        """
        logger.info("Adding NER features")
        data_frame["NER_Features"] = texts.apply(lambda text: compute_ner_features(text, nlp))
        logger.debug(f"NER features example: {data_frame['NER_Features'].head()}")
        combined_features.append(data_frame["NER_Features"].tolist())
        """
        logger.info("Adding NER embeddings")
        data_frame["NER_Features"] = texts.apply(lambda text: compute_ner_embeddings(text, nlp))
        logger.debug(f"NER embedding example: {data_frame['NER_Features'].head()}")
        combined_features.append(data_frame["NER_Features"].tolist())

    # Compute lexicon embeddings for each sentence
    if lexicon:
        data_frame['Lexicon_Scores'] = texts.apply(
            lambda x: compute_lexicon_scores(x, lexicon, lexicon_embeddings, tokenizer, num_categories)
        )
        data_frame['Lexicon_Scores'] = data_frame['Lexicon_Scores'].apply(
            lambda x: x if isinstance(x, list) and len(x) == num_categories else [0.0] * num_categories
        )
        combined_features.append(data_frame['Lexicon_Scores'].tolist())

        #logger.debug(f"Sample Lexicon Scores: {data_frame['Lexicon_Scores'].head()}")
    
    # Combine all features
    if combined_features:
        # Stack features together (ensure consistent dimensions across all features)
        max_length = max(len(f) for features in combined_features for f in features)
        combined_features = [
            np.concatenate([np.asarray(f, dtype=np.float32) if len(f) == max_length else np.pad(f, (0, max_length - len(f)), 'constant') for f in features])
            for features in zip(*combined_features)
        ]
        #combined_features = [np.concatenate([np.asarray(f, dtype=np.float32) for f in features]) for features in zip(*combined_features)]
    else:
        combined_features = [[0.0] * num_categories] * len(data_frame)

    if linguistic_features:
        logger.debug(f"Sample Linguistic Features: {data_frame['Linguistic_Scores'].head()}")
    if ner_features:
        logger.debug(f"Sample NER Features: {data_frame['NER_Features'].head()}")
    if lexicon:
        logger.debug(f"Sample Lexicon Features: {data_frame['Lexicon_Scores'].head()}")

    for i, row in enumerate(combined_features[:5]):
        logger.debug(f"Combined features for sample {i}: {len(row)} elements")
    
    if lexicon:
        encoded_sentences["lexicon_features"] = np.array(combined_features, dtype=np.float32).tolist()
    elif ner_features:
        encoded_sentences["ner_features"] = np.array(combined_features, dtype=np.float32).tolist()

    # Validate input shapes before dataset conversion
    if linguistic_features or ner_features or lexicon:
        validate_input_shapes(encoded_sentences)

    # Load labels if available
    if os.path.isfile(labels_file_path):
        labels_frame = pandas.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)
        # Slicing for testing purposes
        if slice_data:
            labels_frame = slice_for_testing(labels_frame)
        #labels_matrix = numpy.zeros((labels_frame.shape[0], len(labels)))
        #for idx, label in enumerate(labels):
        #    if label in labels_frame.columns:
        #        labels_matrix[:, idx] = (labels_frame[label] >= 0.5).astype(int)
        labels_matrix = labels_frame[labels].ge(0.5).astype(int).to_numpy()
        encoded_sentences["labels"] = labels_matrix.astype(np.float32).tolist()

    encoded_sentences = datasets.Dataset.from_dict(encoded_sentences)

    return encoded_sentences

# ========================================================
# SCORE COMPUTATION
# ========================================================

def compute_vad_scores(text, lexicon_embeddings, tokenizer, num_categories=None):
    """Compute the average VAD scores for a given text."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [normalize_token(token) for token in tokens]
    scores = {"valence": 0, "arousal": 0, "dominance": 0}
    count = 0

    for token in normalized_tokens:
        token = token.lower()
        if token in lexicon_embeddings:
            count += 1
            scores["valence"] += lexicon_embeddings[token]["valence"]
            scores["arousal"] += lexicon_embeddings[token]["arousal"]
            scores["dominance"] += lexicon_embeddings[token]["dominance"]

    if count > 0:
        for key in scores:
            scores[key] /= count
    else:
        scores = {"valence": 0.5, "arousal": 0.5, "dominance": 0.5}  # Default neutral values

    return [scores["valence"], scores["arousal"], scores["dominance"]]

def compute_emolex_scores(text, lexicon_embeddings, tokenizer, num_categories=10):
    """Compute the average EmoLex scores for a given text."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [normalize_token(token) for token in tokens]
    scores = [0.0] * num_categories  # EmoLex typically has 10 emotions
    count = 0

    for token in normalized_tokens:
        if token.lower() in lexicon_embeddings:
            count += 1
            for i in range(num_categories):
                scores[i] += lexicon_embeddings[token.lower()][i]
    
    if count > 0:
        scores = [score / count for score in scores]  # Average scores
    else:
        scores = [0.0] * num_categories  # Neutral scores
    
    #token_matches = [token for token in normalized_tokens if token in lexicon_embeddings]

    return scores

def compute_emotionintensity_scores(text, lexicon_embeddings, tokenizer, num_categories=None):
    """Compute the average Intensity scores for the 8 emotions for a given text."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [normalize_token(token) for token in tokens]

    # Initialize scores with neutral values
    scores = {emotion: 0.5 for emotion in ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]}
    count = 0

    # Sum up scores for tokens present in lexicon
    for token in normalized_tokens:
        if token.lower() in lexicon_embeddings:
            count += 1
            for emotion in scores.keys():
                scores[emotion] += lexicon_embeddings[token].get(emotion, 0.0) - 0.5
    
    # Normalize if tokens matched
    if count > 0:
        for key in scores:
            scores[key] = scores[key] / count + 0.5  # Normalize and re-center around 0.5

    return list(scores.values())  # Return as a list of 8 scores

def compute_worrywords_scores(text, lexicon_embeddings, tokenizer, num_categories=None):
    """Compute the average worry score for a given text."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [normalize_token(token) for token in tokens]
    total_score = 0.0
    count = 0
    for token in normalized_tokens:
        if token.lower() in lexicon_embeddings:
            count += 1
            total_score += lexicon_embeddings[token.lower()]
    return [total_score / count] if count > 0 else [0.0]  # Return the average worry score as a list

def compute_liwc_scores(text, lexicon_embeddings, tokenizer, num_categories):
    """Compute the average LIWC scores for a given text."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [normalize_token(token) for token in tokens]
    scores = [0.0] * num_categories

    # Create a mapping of valid category IDs to their indices in `scores`
    category_mapping = {category_id: idx for idx, category_id in enumerate(range(1, num_categories + 1))}

    for token in normalized_tokens:
        token_lower = token.lower()
        matched = False

        # Exact match
        if token_lower in lexicon_embeddings:
            matched = True
            for category in lexicon_embeddings[token_lower]:
                if category in category_mapping:  # Ensure category is valid
                    scores[category_mapping[category]] += 1  # Increment score for the category

        # Stem match
        if not matched:
            for word, categories in lexicon_embeddings.items():
                if word.endswith("*") and token_lower.startswith(word[:-1]):
                    for category in categories:
                        if category in category_mapping:  # Ensure category is valid
                            scores[category_mapping[category]] += 1  # Increment score for the category
                    break

    # Normalize scores by total words (if tokens matched)
    if tokens:
        scores = [score / len(tokens) for score in scores]
    else:
        scores = [0.0] * num_categories  # Neutral scores for LIWC categories

    return scores

def compute_linguistic_features(text, nlp):
    """Compute expanded linguistic and discourse features for moral value prediction."""
    doc = nlp(text)

    if not text.strip():
        return [0] * 17  # Return zeroes for empty text
    
    # Initialize feature counts
    feature_counts = {
        "verbs_present": 0,
        "verbs_past": 0,
        "verbs_future": 0,
        "active_voice": 0,
        "passive_voice": 0,
        "sentence_length": 0,
        "questions": 0,
        "first_person": 0,
        "second_person": 0,
        "third_person": 0,
        "repetition": 0,
        "all_caps_words": 0,
        "exclamation_marks": 0,
        "emphasis_signals": 0,
        "comparison_signals": 0,
        "contrast_signals": 0,
        "illustration_signals": 0
    }
    total_verbs = 0
    total_sentences = len(list(doc.sents)) or 1
    word_counts = {}

    # Predefined transition or signal words
    # Source: https://www.cpp.edu/ramp/program-materials/recognizing-transitions.shtml
    emphasis_signals = {"important to note", "most of all", "a significant factor", "a primary concern", "a key feature", "the main value", "especially valuable", "most noteworthy", "remember that", "a major event", "the chief outcome", "the principal item", "pay particular attention to", "the chief factor", "a vital force", "above all", "a central issue", "a distinctive quality", "especially relevant", "should be noted", "the most substantial issue"}
    comparison_signals = {"like", "likewise", "just", "equally", "in like manner", "in the same way", "alike", "similarity", "similarly", "just as", "as in a similar fashion"}
    contrast_signals = {"but", "however", "in contrast", "yet", "differ", "difference", "variation", "still", "on the contrary", "conversely", "otherwise", "on the other hand"}
    illustration_signals = {"for example", "to illustrate", "specifically", "once", "for instance", "such as"}

    # Check for multi-word signals in the text

    text_lower = text.lower()

    # Create regex patterns
    def count_custom_words(text, word_list):
        """Count occurrences of words or phrases in a text."""
        pattern = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, word_list))))
        return len(pattern.findall(text))

    # Count matches
    feature_counts["emphasis_signals"] = count_custom_words(text_lower, emphasis_signals)
    feature_counts["comparison_signals"] = count_custom_words(text_lower, comparison_signals)
    feature_counts["contrast_signals"] = count_custom_words(text_lower, contrast_signals)
    feature_counts["illustration_signals"] = count_custom_words(text_lower, illustration_signals)
    
    # Check for single-word phrases using SpaCy tokens

    for token in doc:
        # Verb tenses and voice
        if token.pos_ == "VERB":
            total_verbs += 1
            if "Tense=Pres" in token.morph:
                feature_counts["verbs_present"] += 1
            elif "Tense=Past" in token.morph:
                feature_counts["verbs_past"] += 1
            elif "Tense=Fut" in token.morph:
                feature_counts["verbs_future"] += 1
            if "Voice=Act" in token.morph:
                feature_counts["active_voice"] += 1
            elif "Voice=Pass" in token.morph:
                feature_counts["passive_voice"] += 1

        # Repetition
        token_lower = token.text.lower()
        word_counts[token_lower] = word_counts.get(token_lower, 0) + 1

        # Emphasis (all-caps words)
        if token.text.isupper() and len(token.text) > 1:
            feature_counts["all_caps_words"] += 1

    # Sentence-level features
    feature_counts["sentence_length"] = len(doc) / total_sentences
    feature_counts["questions"] = text.count("?")
    feature_counts["exclamation_marks"] = text.count("!")

    # Normalize verb features by total verbs
    if total_verbs > 0:
        for key in ["verbs_present", "verbs_past", "verbs_future", "active_voice", "passive_voice"]:
            feature_counts[key] /= total_verbs

    # Compute repetition (words repeated more than once)
    feature_counts["repetition"] = sum(1 for count in word_counts.values() if count > 1)

    # Return as a feature vector
    return list(feature_counts.values())

def compute_ner_features(text, nlp):
    """
    Extract NER features from text using spaCy.
    Returns a dictionary of entity counts for each entity type.
    """
    doc = nlp(text)

    # Define known entity types (ensure consistency across all texts)
    known_entities = [
        "PERSON", "ORG", "GPE", "DATE", "MONEY", "LAW", "EVENT", "PERCENT"
    ]

    entity_counts = {ent: 0 for ent in known_entities}

    for ent in doc.ents:
        if ent.label_ in entity_counts:
            entity_counts[ent.label_] += 1
    
    # Ensure the returned vector is always the same length
    return [entity_counts[ent] for ent in known_entities]

# Load the tokenizer and model once to avoid reloading for each text
ner_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
ner_model = AutoModel.from_pretrained("microsoft/deberta-base")

def compute_ner_embeddings(text, nlp):
    doc = nlp(text)

    # Collect entity texts
    entity_texts = [ent.text for ent in doc.ents]

    if entity_texts:
        # Tokenize entity texts and get embeddings
        encoded = ner_tokenizer(entity_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = ner_model(**encoded)
        
        # Take the CLS token representation for each entity
        embeddings = output.last_hidden_state[:, 0, :].mean(dim=0)  # Mean pooling
        return embeddings.cpu().numpy().tolist()
    
    # Return zero vector if no entities found
    return [0.0] * ner_model.config.hidden_size

def compute_mfd_scores(text, mfd_embeddings, tokenizer):
    """Compute the count of words matching each Moral Foundation dimension."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [normalize_token(token) for token in tokens]

    scores = defaultdict(int)
    
    # Compute counts for each category
    for token in normalized_tokens:
        if token.upper() in mfd_embeddings:
            category = mfd_embeddings[token.upper()]
            scores[category] += 1

    return dict(scores)

LEXICON_COMPUTATION_FUNCTIONS = {
    "VAD": compute_vad_scores,
    "EmoLex": compute_emolex_scores,
    "EmotionIntensity": compute_emotionintensity_scores,
    "WorryWords": compute_worrywords_scores,
    "LIWC": compute_liwc_scores,
    "MFD": compute_mfd_scores,
}

def compute_lexicon_scores(text, lexicon, lexicon_embeddings, tokenizer, num_categories):
    """Compute lexicon scores with fallback for missing words."""
    compute_fn = LEXICON_COMPUTATION_FUNCTIONS.get(lexicon)
    if not compute_fn:
        raise ValueError(f"Unknown lexicon: {lexicon}")
    
    # Pass `num_categories` only if the function supports it
    try:
        scores = compute_fn(text, lexicon_embeddings, tokenizer, num_categories=num_categories)
    except TypeError:
        scores = compute_fn(text, lexicon_embeddings, tokenizer)
    
    # Ensure consistent length
    if len(scores) != num_categories:
        scores = [0.0] * num_categories
    
    return scores