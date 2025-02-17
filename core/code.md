# Collected Python Files
## cli.py
```python
import argparse

def parse_args(prog_name) -> argparse.Namespace:
    cli = argparse.ArgumentParser(prog=prog_name)
    cli.add_argument("-dg", "--debug", action='store_true', help="Show debug messages")
    cli.add_argument("-t", "--training-dataset", required=True, help="Path to training dataset")
    cli.add_argument("-v", "--validation-dataset", default=None, help="Path to validation dataset")
    cli.add_argument("-p", "--previous-sentences", action='store_true', help="If to add the two previous sentences of every sentence to the model")
    cli.add_argument("-f", "--linguistic-features", action='store_true', help="If to add linguistic features to the model")
    cli.add_argument("-n", "--ner-features", action='store_true', help="If to add NER features to the model")
    cli.add_argument("-l", "--lexicon", default=None, help="Lexicon to be added on top of the model")
    cli.add_argument("-m", "--model-name", help="Name of the model if being uploaded to HuggingFace")
    cli.add_argument("-d", "--model-directory", default="models", help="Directory to save the trained model")
    cli.add_argument("-y", "--multilayer", action='store_true', help="Use multilayer design instead of single layer")
    cli.add_argument("-s", "--slice", action='store_true', help="Slice for testing with size = 100")
    cli.add_argument("-o", "--optimize", action='store_true', help="If set, run hyperparameter optimization with Optuna")
    cli.add_argument("-a", "--augment-data", action="store_true", help="Apply data augmentation through paraphrasing")
    cli.add_argument("-td", "--topic-detection", choices=["bertopic", "lda", "nmf", "none"], default="none", help="Choose topic detection method: BERTopic, LDA, NMF, or None")
    return cli.parse_args()
```
-----------
## config.py
```python
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
```
-----------
## dataset_utils.py
```python
import pandas as pd
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
from core.config import LEXICON_PATHS
from core.lexicon_utils import load_lexicon
from core.utils import validate_args, normalize_token, slice_for_testing
from core.topic_detection import TopicModeling

from core.log import logger

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
    custom_stopwords: List[str] = [],
    augment_data: bool = False,
    topic_detection: str = None,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    
     # 1) Possibly load separate training vs. validation embeddings if we have a known “LIWC-22” style
    if lexicon == "LIWC-22":
        # Example: We handle training and validation differently
        train_lexicon, train_num_cat = load_lexicon(lexicon, LEXICON_PATHS[lexicon+"-training"])
        val_lexicon, val_num_cat     = load_lexicon(lexicon, LEXICON_PATHS[lexicon+"-validation"])
    else:
        # Fallback: single lexicon for everything (the old approach)
        train_lexicon, train_num_cat = lexicon_embeddings, num_categories
        val_lexicon, val_num_cat     = lexicon_embeddings, num_categories

    # Training dataset
    training_dataset = load_dataset(
        directory=training_path,
        tokenizer=tokenizer,
        labels=labels,
        slice_data=slice_data,
        lexicon_embeddings=train_lexicon,
        num_categories=train_num_cat,
        previous_sentences=previous_sentences,
        linguistic_features=linguistic_features,
        ner_features=ner_features,
        lexicon=lexicon,
        custom_stopwords=custom_stopwords,
        augment_data=augment_data,
        topic_detection=topic_detection
    )

    # Log class distribution
    labels_array = np.array(training_dataset["labels"])
    logger.debug(f"Class distribution: {labels_array.sum(axis=0)}")

    # Validation dataset
    validation_dataset = training_dataset
    if validation_path:
        validation_dataset = load_dataset(
            directory=validation_path,
            tokenizer=tokenizer,
            labels=labels,
            slice_data=slice_data,
            lexicon_embeddings=val_lexicon,
            num_categories=val_num_cat,
            previous_sentences=previous_sentences,
            linguistic_features=linguistic_features,
            ner_features=ner_features,
            lexicon=lexicon,
            custom_stopwords=custom_stopwords,
            topic_detection=topic_detection
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
    custom_stopwords: List[str] = [],
    augment_data: bool = False,
    topic_detection: str = None
):
    """Load dataset and add embeddings if specified."""
    if augment_data:
        sentences_file_name = "sentences-aug.tsv"
        labels_file_name = "labels-cat-aug.tsv"
    else:
        sentences_file_name = "sentences.tsv"
        labels_file_name = "labels-cat.tsv"
    sentences_file_path = os.path.join(directory, sentences_file_name)
    labels_file_path = os.path.join(directory, labels_file_name)
    
    data_frame = pd.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0)

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

    texts = pd.Series(texts)

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
        # 1) Is this lexicon in the dict of token-based scorers?
        if lexicon in LEXICON_COMPUTATION_FUNCTIONS:
            # Same old token-based approach
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_lexicon_scores(x, lexicon, lexicon_embeddings, tokenizer, num_categories)
            )
        else:
            # 2) This must be a precomputed (row-level) lexicon
            #    e.g. "LIWC-22", "MyCustomLexicon", etc.
            data_frame['Lexicon_Scores'] = data_frame.apply(
                lambda row: compute_precomputed_scores(row, lexicon_embeddings, num_categories),axis=1
            )

        data_frame['Lexicon_Scores'] = data_frame['Lexicon_Scores'].apply(
            lambda x: x if isinstance(x, list) and len(x) == num_categories else [0.0] * num_categories
        )
        combined_features.append(data_frame['Lexicon_Scores'].tolist())

        #logger.debug(f"Sample Lexicon Scores: {data_frame['Lexicon_Scores'].head()}")
    
    # Compute Topic features
    if topic_detection:
        logger.info(f"Applying {topic_detection} for topic modeling.")
        topic_model = TopicModeling(method=topic_detection)
        topic_vectors = None
        topic_vectors = topic_model.fit_transform(texts)

        if topic_vectors is not None:
            encoded_sentences["topic_features"] = topic_vectors.tolist()
            combined_features.append(topic_vectors.tolist())
    
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
    if topic_detection:
        logger.debug(f"Sample Topic Detection Features: {encoded_sentences['topic_features'][:5]}")

    for i, row in enumerate(combined_features[:5]):
        logger.debug(f"Combined features for sample {i}: {len(row)} elements")
    
    if lexicon:
        encoded_sentences["lexicon_features"] = np.array(combined_features, dtype=np.float32).tolist()
    elif ner_features:
        encoded_sentences["ner_features"] = np.array(combined_features, dtype=np.float32).tolist()

    # Validate input shapes before dataset conversion
    if combined_features:
        validate_input_shapes(encoded_sentences)

    # Load labels if available
    if os.path.isfile(labels_file_path):
        labels_frame = pd.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)
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

def compute_schwartz_values(text: str, lexicon: dict) -> List[int]:
    """Compute the count of Schwartz value words in the text."""
    words = word_tokenize(text.lower())

    # Convert token list back to a string for regex processing
    text_str = " ".join(words)

    # Initialize value counts
    value_counts = {key: 0 for key in lexicon.keys()}
    
    for value, phrases in lexicon.items():
        for phrase in phrases:
            pattern = r'\b' + re.escape(phrase) + r'\b'  # Match whole words
            matches = re.findall(pattern, text_str)
            value_counts[value] += len(matches)
    
    return list(value_counts.values())

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
    "Schwartz": compute_schwartz_values
}

def compute_lexicon_scores(text, lexicon, lexicon_embeddings, tokenizer, num_categories):
    """Compute lexicon scores with fallback for missing words."""
    compute_fn = LEXICON_COMPUTATION_FUNCTIONS.get(lexicon)
    if not compute_fn:
        raise ValueError(f"Unknown lexicon: {lexicon}")
    
    # Schwartz lexicon does not use tokenizer or num_categories, handle it separately
    if lexicon == "Schwartz":
        schwartz_lexicon, _ = lexicon_embeddings
        scores = compute_fn(text, schwartz_lexicon)
    else:
        # Pass `num_categories` only if the function supports it
        try:
            scores = compute_fn(text, lexicon_embeddings, tokenizer, num_categories=num_categories)
        except TypeError:
            scores = compute_fn(text, lexicon_embeddings, tokenizer)
    
    # Ensure consistent length
    if len(scores) != num_categories:
        scores = [0.0] * num_categories
    
    return scores


def compute_precomputed_scores(
    row,
    precomputed_dict: dict, 
    num_categories: int
):
    """
    For a 'precomputed' lexicon like LIWC-22, look up the row's (Text-ID, Sentence-ID)
    in precomputed_dict and return the features. 
    If not found, fall back to zeros.
    """
    text_id = str(row["Text-ID"])
    sent_id = str(row["Sentence-ID"])
    key = (text_id, sent_id)

    if key in precomputed_dict:
        return precomputed_dict[key]
    else:
        # If no match, fallback
        return [0.0] * num_categories
```
-----------
## lexicon_utils.py
```python
import re
from typing import Dict, List, Tuple
from core.config import LEXICON_PATHS, SCHWARTZ_VALUE_LEXICON
from core.utils import validate_file, skip_invalid_line, read_file_lines
import pandas as pd

from core.log import logger

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

def load_liwc22_embeddings(path: str) -> tuple[dict, int]:
    """
    Loads a CSV produced by LIWC-22, returning a dictionary that maps 
    (TextID, SentenceID) -> list of precomputed scores, plus the number of columns.
    """
    validate_file(path)
    data = pd.read_csv(path, encoding='utf-8')
    
    # Suppose your CSV has columns like:
    # [Text-ID, Sentence-ID, Text, Segment, WC, Analytic, Clout, Authentic, Tone, WPS, BigWords, Dic, ...]
    # We'll skip the first 4 columns (Text-ID, Sentence-ID, Text, Segment) and use the rest as features:
    columns_to_use = data.columns[4:]  # or whichever subset you want
    num_features = len(columns_to_use)

    # Build dictionary from (Text-ID, Sentence-ID) -> feature vector
    embeddings = {}
    for _, row in data.iterrows():
        text_id = str(row["Text-ID"])
        sent_id = str(row["Sentence-ID"])
        # Collect numeric columns as a list of floats
        scores = row[columns_to_use].astype(float).tolist()
        embeddings[(text_id, sent_id)] = scores

    logger.debug(f"Loaded LIWC-22 embeddings: {len(embeddings)} rows, {num_features} features each.")
    return embeddings, num_features

# ========================================================
# MAIN LEXICON LOADING
# ========================================================

EMBEDDING_PARSERS = {
    "VAD": {"function": load_vad_embeddings, "num_categories": 3},
    "EmoLex": {"function": load_emolex_embeddings, "num_categories": 10},
    "EmotionIntensity": {"function": load_emotionintensity_embeddings, "num_categories": 8},
    "WorryWords": {"function": load_worrywords_embeddings, "num_categories": 1},
    "LIWC": {"function": load_liwc_embeddings, "num_categories": None},
    "MFD": {"function": load_mfd_embeddings, "num_categories": 10},
    "Schwartz": {"function": load_schwartz_embeddings, "num_categories": len(SCHWARTZ_VALUE_LEXICON)},
    "LIWC-22": {"function": load_liwc22_embeddings, "num_categories": None},
}

def load_lexicon(lexicon_name: str, path: str) -> tuple[dict, int]:
    parser = EMBEDDING_PARSERS.get(lexicon_name)
    if not parser:
        raise ValueError(f"Unknown lexicon: {lexicon_name}")
    
    embeddings, dynamic_num_categories = parser["function"](path)
    
    # If the parser's dictionary says "num_categories" is None, then use the
    # dynamic value returned by the loader. Otherwise, use the fixed one:
    if parser["num_categories"] is not None:
        num_categories = parser["num_categories"]
    else:
        num_categories = dynamic_num_categories

    logger.debug(f"Loaded {lexicon_name} embeddings with {len(embeddings)} items, {num_categories} features.")
    return embeddings, num_categories

```
-----------
## log.py
```python
import sys
import logging
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# Remove any existing log handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Force stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True)

# Custom handler that ensures immediate flushing
class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # Ensure logs are flushed immediately
    
# Configure logging to always use stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[FlushStreamHandler(sys.stdout)]  # Force logs to stdout
)

logger = logging.getLogger("HVD")

# Suppress duplicate logs on multi-GPU runs (only rank 0 logs)
if not accelerator.is_main_process:
    logger.setLevel(logging.WARNING)  # Reduce logging for non-primary ranks
```
-----------
## models.py
```python
import torch
from torch import nn
import transformers
from transformers import TrainerCallback
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.distributed as dist

from core.log import logger

# ========================================================
# UTILS
# ========================================================

def save_model(trainer, model_name, model_directory):
    # Ensure only the main GPU (rank 0) saves the model
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    
    if model_name:
        logger.info(f"UPLOAD to https://huggingface.co/{model_name} (using HF_TOKEN environment variable)")
        # trainer.push_to_hub()

    if model_directory and model_name:
        logger.info(f"SAVE to {model_directory}")
        trainer.save_model(f"{model_directory}/{model_name}")

        # Ensure the model's configuration is also saved
        if hasattr(trainer.model, 'config'):
            trainer.model.config.save_pretrained(f"{model_directory}/{model_name}")

        if trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(f"{model_directory}/{model_name}")

def move_to_device(model):
    if torch.cuda.is_available():
        logger.info("Using CUDA for training.")
        return model.to('cuda')
    else:
        logger.info("Using CPU for training.")
        return model

# ========================================================
# MODELS
# ========================================================

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_groups=8):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            # nn.GroupNorm(num_groups, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, output_dim),
            # nn.GroupNorm(num_groups, output_dim),
            nn.ReLU()
        )
        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.linear_layers(x) + self.projection(x)  # Add residual connection

class EnhancedDebertaModel(nn.Module):
    """Enhanced DeBERTa model with added lexicon feature layer."""
    def __init__(self, pretrained_model, config, num_labels, id2label, label2id, num_categories=0, ner_feature_dim=0, multilayer = False, num_groups=8, topic_feature_dim=0):
        #super(EnhancedDebertaModel, self).__init__()
        super().__init__()
        self.config = config  # Store config attribute

        self.transformer = transformers.AutoModel.from_pretrained(pretrained_model)

        """
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs.")
            self.transformer = torch.nn.DataParallel(self.transformer)  # Enables Multi-GPU
        """

        # Optional Lexicon Layer
        #self.lexicon_layer = nn.Linear(num_categories, 128)  # Map categories to 128 dimensions
        if num_categories > 0:
            self.lexicon_layer = nn.Sequential(
                nn.Linear(num_categories, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            logger.debug("Lexicon layer initialized at model")
        else:
            self.lexicon_layer = None
            logger.debug("No lexicon layer initialized at model")

        # Optional NER Layer
        #self.ner_layer = nn.Linear(num_categories, 128)  # Map categories to 128 dimensions
        if ner_feature_dim > 0:
            self.ner_layer = nn.Sequential(
                nn.Linear(ner_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
            )
            logger.debug("NER layer initialized at model")
        else:
            self.ner_layer = None
            logger.debug("No NER layer initialized at model")
        
        if topic_feature_dim > 0:
            self.topic_layer = nn.Sequential(
                nn.Linear(topic_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.4)
            )
            logger.debug("Topic Detection layer initialized at model")
        else:
            self.topic_layer = None
            logger.debug("No Topic Detection layer initialized at model")
        
        # Multi-layer processing for transformer embeddings
        self.multilayer = multilayer
        if multilayer:
            self.text_embedding_layer = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, 512),
                nn.GroupNorm(num_groups, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.GroupNorm(num_groups, 256),
                nn.ReLU()
            )
            
            # self.text_embedding_layer = ResidualBlock(self.transformer.config.hidden_size, 256)
            hidden_size = 256
        else:
            hidden_size = self.transformer.config.hidden_size

        # Classification head. Combine all features
        input_dim = hidden_size
        if self.lexicon_layer:
            input_dim += 128
        if self.ner_layer:
            input_dim += 128
        if self.topic_layer:
            input_dim += 128
        self.classification_head = nn.Linear(input_dim, num_labels)
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)

        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

    def forward(self, input_ids, attention_mask, lexicon_features=None, ner_features=None, topic_features=None, labels=None, **kwargs):
        """Forward pass for the enhanced model."""

        logger.debug(f"Lexicon features received: {lexicon_features is not None}")
        logger.debug(f"NER features received: {ner_features is not None}")

        if lexicon_features is not None:
            logger.debug(f"Lexicon feature shape: {lexicon_features.shape}")

        if ner_features is not None:
            logger.debug(f"NER feature shape: {ner_features.shape}")

        # Extract transformer embeddings
        hidden_state = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        transformer_output = hidden_state[:, 0, :] # CLS token representation

        # Process transformer embeddings through additional layers
        if self.multilayer:
            text_embeddings = self.text_embedding_layer(transformer_output)
        else:
            text_embeddings = transformer_output
        
        combined_output = text_embeddings

        # Handle lexicon features if provided
        if self.lexicon_layer and lexicon_features is not None:
            logger.debug(f"Lexicon features shape before processing: {lexicon_features.shape}")
            lexicon_features = lexicon_features.to(input_ids.device)
            #lexicon_output = torch.relu(self.lexicon_layer(lexicon_features))
            lexicon_output = self.lexicon_layer(lexicon_features)
            logger.debug(f"Lexicon output shape: {lexicon_output.shape}")
            combined_output = torch.cat([combined_output, lexicon_output], dim=-1)
        
        if self.ner_layer and ner_features is not None:
            logger.debug(f"NER features shape before processing: {ner_features.shape}")
            ner_features = ner_features.to(input_ids.device)
            ner_output = self.ner_layer(ner_features)
            logger.debug(f"NER output shape: {ner_output.shape}")
            combined_output = torch.cat([combined_output, ner_output], dim=-1)

        if self.topic_layer and topic_features is not None:
            logger.debug(f"Topic Detection features shape before processing: {topic_features.shape}")
            topic_features = topic_features.to(input_ids.device)
            topic_output = self.topic_layer(topic_features)
            logger.debug(f"Topic Detection output shape: {topic_output.shape}")
            combined_output = torch.cat([combined_output, topic_output], dim=-1)
        
        logger.debug(f"Final combined output shape: {combined_output.shape}")

        combined_output = self.dropout(combined_output)
        logits = self.classification_head(combined_output)
        #logger.debug(f"Logits Shape: {logits.shape}")  # Should be [batch_size, num_labels]
        loss = None
        if labels is not None:
            labels = labels.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)  # Ensure 2D labels
            loss = binary_cross_entropy_with_logits(logits, labels)

        return {"logits": logits, "loss": loss}

# ========================================================
# TRAINERS
# ========================================================
    
class CustomTrainer(transformers.Trainer):
    """Custom Trainer with modified loss function for multi-label classification."""
    def compute_loss(self, model, inputs, return_outputs=False):
        logger.debug(f"Input IDs Shape: {inputs['input_ids'].shape}")
        logger.debug(f"Attention Mask Shape: {inputs['attention_mask'].shape}")

        # Debug lexicon features
        if "lexicon_features" in inputs:
            logger.debug(f"Lexicon Features Shape: {inputs['lexicon_features'].shape}")
        else:
            logger.debug("No lexicon features")
        
        # Pop labels for loss computation
        labels = inputs.pop("labels")
        logger.debug(f"Labels Shape: {labels.shape}")

        if labels.dim() == 1:  # Ensure labels are 2D
            labels = labels.unsqueeze(1)

        # Forward pass through the model
        outputs = model(**inputs, labels=labels)

        # Retrieve loss and logits from the model's outputs
        logits = outputs["logits"]
        loss = outputs["loss"]
        if loss.dim() > 0:
            loss = loss.mean()  # Reduce to a scalar value if necessary
        logger.debug(f"Logits Shape: {logits.shape}")
        logger.debug(f"Loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss

# ========================================================
# CALLBACKS
# ========================================================

class WarmupEvalCallback(TrainerCallback):
    def __init__(self, warmup_epochs=2):
        self.warmup_epochs = warmup_epochs

    def on_evaluate(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch <= self.warmup_epochs:
            logger.info(f"Skipping evaluation for warm-up phase (epoch {current_epoch}).")
            control.should_evaluate = False
            control.should_save = False
        else:
            control.should_evaluate = True
            control.should_save = True
        return control
```
-----------
## runner.py
```python
import transformers
from core.dataset_utils import prepare_datasets
from core.lexicon_utils import load_embeddings
from core.training import train
from core.models import save_model
import torch.distributed as dist
from accelerate import Accelerator

from core.log import logger

def run_training(
    pretrained_model: str,
    labels: list[str],
    training_dataset_path: str,
    validation_dataset_path: str = None,
    lexicon: str = None,
    previous_sentences: bool = False,
    linguistic_features: bool = False,
    ner_features: bool = False,
    model_name: str = None,
    model_directory: str = "models",
    multilayer: bool = False,
    slice_data: bool = False,
    batch_size: int = 4,
    num_train_epochs: int = 9,
    learning_rate: float = 2.07e-05,
    weight_decay: float = 1.02e-05,
    gradient_accumulation_steps: int = 2,
    early_stopping_patience: int = 3,
    custom_stopwords: list[str] = [],
    augment_data: bool = False,
    topic_detection: str = None
):

    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    # Tokenizer
    logger.info("Initializing tokenizer for model: %s", pretrained_model)
    tokenizer = transformers.DebertaTokenizer.from_pretrained(
        pretrained_model,
        truncation_side = "left" if previous_sentences else "right"
    )

    # Lexicon embeddings
    logger.info("Loading lexicon embeddings for: %s", lexicon if lexicon else "No lexicon used")
    if lexicon and not lexicon.startswith("LIWC-22"):
        lexicon_embeddings, num_categories = load_embeddings(lexicon)
    else:
        lexicon_embeddings, num_categories = None, 0  # No lexicon features

    # Linguistic embeddings
    if linguistic_features:
        num_linguistic_features = 17  # Total number of linguistic features
        num_categories += num_linguistic_features
    
    # Prepare datasets
    logger.info("Preparing datasets for training and validation")
    training_dataset, validation_dataset = prepare_datasets(
        training_dataset_path,
        validation_dataset_path,
        tokenizer,
        labels,
        slice_data,
        lexicon_embeddings,
        num_categories,
        previous_sentences,
        linguistic_features,
        ner_features,
        lexicon,
        custom_stopwords,
        augment_data,
        topic_detection=topic_detection
    )

    # Train and evaluate
    trainer = train(
        training_dataset,
        validation_dataset,
        pretrained_model,
        tokenizer,
        labels = labels,
        label2id = label2id,
        id2label = id2label,
        model_name = model_name,
        num_categories=num_categories,
        lexicon = lexicon,
        previous_sentences = previous_sentences,
        linguistic_features = linguistic_features,
        ner_features = ner_features,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        multilayer=multilayer,
        augment_data=augment_data,
        topic_detection=topic_detection
    )

    # Save the model if required
    accelerator = Accelerator()
    if model_name and accelerator.is_main_process:
        logger.info(f"Saving best model to {model_directory} directory")
        save_model(trainer, model_name, model_directory)

    # Return the trainer so that caller (objective function) can evaluate
    return trainer
```
-----------
## topic_detection.py
```python
from bertopic import BERTopic
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
import gc
import numpy as np

from core.log import logger

class TopicModeling:
    def __init__(self, method="bertopic", num_topics=40):
        """
        Initialize the topic modeling method.
        
        Args:
            method (str): "bertopic", "lda", or "nmf".
            num_topics (int): The number of topics
        """
        self.method = method
        if method == "bertopic":
            num_topics = 40
        if method == "lda":
            num_topics = 60
        elif method == "nmf":
            num_topics = 90
        self.num_topics = num_topics
        self.model = None

    def fit_transform(self, sentences):
        """
        Train the topic model and transform sentences into topic representations.

        Args:
            sentences (list): List of textual sentences.

        Returns:
            np.ndarray: One-hot encoded topic vectors.
        """
        if self.method == "bertopic":
            # Use GPU-based embedding model for BERTopic
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

            # Initialize BERTopic with a fixed number of topics
            self.model = BERTopic(nr_topics=self.num_topics, embedding_model=embedding_model, verbose=True)
            topics, _ = self.model.fit_transform(sentences)

            # Ensure topics are within a valid range
            self.num_topics = max(topics) + 1  # Update num_topics dynamically

            # Free GPU memory
            del embedding_model  # Delete the embedding model
            torch.cuda.empty_cache()  # Clear unused GPU memory
            gc.collect()  # Run garbage collector

            return self.get_topic_vectors(topics)

        # Use CountVectorizer for LDA, TfidfVectorizer for NMF
        vectorizer = CountVectorizer() if self.method == "lda" else TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)

        # Train LDA or NMF
        if self.method == "lda":
            self.model = LDA(n_components=self.num_topics, random_state=42)
        elif self.method == "nmf":
            self.model = NMF(n_components=self.num_topics, random_state=42)

        # Get topic distributions (probabilities)
        topic_probs = self.model.fit_transform(X)

        # Convert probabilities to topic indices (argmax)
        topic_indices = np.argmax(topic_probs, axis=1)

        return self.get_topic_vectors(topic_indices)

    def get_topic_vectors(self, topics):
        """
        Convert topic indices into one-hot encoded vectors.

        Args:
            topics (list or np.ndarray): Topic indices.

        Returns:
            np.ndarray: One-hot encoded topic representation.
        """
        num_sentences = len(topics)
        topic_vectors = np.zeros((num_sentences, self.num_topics))

        for i, topic in enumerate(topics):
            if topic >= 0 and topic < self.num_topics:  # Ensure topic index is within bounds
                topic_vectors[i, topic] = 1  # One-hot encode the topic assignment
            else:
                continue  # Ignore -1 topics

        return topic_vectors
```
-----------
## training.py
```python
import numpy as np
import torch
import transformers
from transformers import EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import AutoConfig
import torch.distributed as dist

from core.models import EnhancedDebertaModel, CustomTrainer, move_to_device, WarmupEvalCallback
from core.utils import clear_directory

from core.log import logger

# ========================================================
# METRICS
# ========================================================

METRIC_F1_SCORE = "eval_f1-score"
METRIC_MACRO_F1_SCORE = "eval_marco-avg-f1-score"

def compute_metrics(eval_prediction, id2label):
    """Compute evaluation metrics like F1-score."""
    prediction_scores, label_scores = eval_prediction

    logger.debug(f"Prediction scores: {prediction_scores[:5]}")  # Log a few predictions
    logger.debug(f"Label scores: {label_scores[:5]}")            # Log a few labels
    logger.debug(f"Prediction scores shape: {np.array(prediction_scores).shape}")
    logger.debug(f"Label scores shape: {np.array(label_scores).shape}")

    # Convert to tensors
    predictions = torch.sigmoid(torch.tensor(np.array(prediction_scores))) >= 0.5
    labels = torch.tensor(label_scores) >= 0.5

    # Ensure tensors have proper dimensions
    predictions = predictions.unsqueeze(0) if predictions.dim() == 1 else predictions
    labels = labels.unsqueeze(0) if labels.dim() == 1 else labels
    if predictions.shape != labels.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}")

    logger.debug(f"Predictions shape: {predictions.shape}")
    logger.debug(f"Labels shape: {labels.shape}")

    # Compute F1 scores for each label
    f1_scores = {}
    for i in range(predictions.shape[1]):
        predicted = predictions[:, i].sum().item()
        true = labels[:, i].sum().item()
        true_positives = torch.logical_and(predictions[:,i], labels[:,i]).sum().item()
        precision = 0 if predicted == 0 else true_positives / predicted
        recall = 0 if true == 0 else true_positives / true
        f1_scores[id2label[i]] = round(0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall), 2)

    # Compute macro-average F1 score
    macro_average_f1_score = round(np.mean(list(f1_scores.values())), 2)
    return {'f1-score': f1_scores, 'marco-avg-f1-score': macro_average_f1_score}

# ========================================================
# TRAINING
# ========================================================

def create_training_args(output_dir, model_name, batch_size, num_train_epochs, learning_rate, weight_decay, gradient_accumulation_steps):
    return transformers.TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        hub_model_id=model_name,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='marco-avg-f1-score',
        gradient_accumulation_steps=gradient_accumulation_steps,
        #fp16=True,
        #bf16=True,
        ddp_find_unused_parameters=False,
        save_on_each_node=True if dist.is_initialized() else False  # Ensure model is saved on each GPU node
    )

def train(
        training_dataset,
        validation_dataset,
        pretrained_model: str,
        tokenizer: transformers.PreTrainedTokenizer,
        labels: list[str],
        label2id: dict[str, int],
        id2label: dict[int, str],
        model_name: str = None,
        batch_size: int = 4,
        num_train_epochs: int = 10,
        learning_rate: float = 2e-05,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        early_stopping_patience=4,
        num_categories: int = 0,
        lexicon: str = None,
        previous_sentences: bool = False,
        linguistic_features: bool = False,
        ner_features: bool = False,
        multilayer: bool = False,
        augment_data: bool = False,
        topic_detection: str = None,
    ) -> transformers.Trainer:
    """Train the model and evaluate performance."""

    if previous_sentences or augment_data:
        scaled_gradient_accumulation_steps = int(gradient_accumulation_steps * batch_size / 2)
        logger.info(f"Previous sentences or augmented data detected. Adjusting batch size: {batch_size} -> 2 and gradient accumulation steps: {gradient_accumulation_steps} -> {scaled_gradient_accumulation_steps}")
        gradient_accumulation_steps = scaled_gradient_accumulation_steps
        batch_size = 2
    
    # Detect number of available GPUs
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        scaled_batch_size = batch_size * num_gpus  # Multiply batch size per GPU count
        scaled_gradient_accumulation_steps = int(gradient_accumulation_steps * scaled_batch_size / 2)
        logger.info(f"Multi-GPU detected ({num_gpus} GPUs). Adjusting batch size: {batch_size} -> {scaled_batch_size} and gradient accumulation steps: {gradient_accumulation_steps} -> {scaled_gradient_accumulation_steps}")
        batch_size = scaled_batch_size
        gradient_accumulation_steps = scaled_gradient_accumulation_steps
    """

    output_dir = "models/checkpoints"

    # Ensure only rank 0 (primary process) clears the directory before training starts
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(f"Clearing old checkpoints in {output_dir}")
        clear_directory(output_dir)
    
    training_args = create_training_args(
        output_dir, model_name, batch_size, num_train_epochs, learning_rate, weight_decay, gradient_accumulation_steps
    )

    if ner_features:
        # ner_feature_dim = 8
        ner_feature_dim = 768 # DeBERTa hidden size
    else:
        ner_feature_dim = 0
    
    topic_feature_dim = 10 if topic_detection != None else 0  # Adjust based on method
    
    config = AutoConfig.from_pretrained(pretrained_model)
    # Add necessary attributes to config
    config.id2label = id2label
    config.label2id = label2id
    config.problem_type = "multi_label_classification"
    config.architectures = ["DebertaForSequenceClassification"]
    model = EnhancedDebertaModel(pretrained_model, config, len(labels), id2label, label2id, num_categories, ner_feature_dim, multilayer, topic_feature_dim)
    """
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, problem_type="multi_label_classification",
        num_labels=len(labels), id2label=id2label, label2id=label2id)
    """
    
    model = move_to_device(model)

    logger.info("TRAINING")
    logger.info("========")

    early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=0.0)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    # Log the training arguments
    config_details = (
        f"Pre-trained model: {pretrained_model}\n"
        f"Model name: {model_name if model_name else 'None'}\n"
        f"Batch size: {batch_size}\n"
        f"Number of epochs: {num_train_epochs}\n"
        f"Learning rate: {learning_rate}\n"
        f"Weight decay: {weight_decay}\n"
        f"Gradient accumulation steps: {gradient_accumulation_steps}\n"
        f"Early stopping patience: {early_stopping_patience}\n"
        f"Multilayer: {'Yes' if multilayer else 'No'}\n"
        f"Previous sentences used: {'Yes' if previous_sentences else 'No'}\n"
        f"Using lexicon: {lexicon if lexicon else 'No'}\n"
        f"Adding linguistic features: {'Yes' if linguistic_features else 'No'}\n"
        f"Adding NER features: {'Yes' if ner_features else 'No'}\n"
        f"Number of categories (lexicon): {num_categories}\n"
        f"Using data augmentation with paraphrasing: {'Yes' if augment_data else 'No'}\n"
        f"Adding topic detection features: {'Yes' if topic_detection else 'No'}\n"
    )
    logger.info("Training configuration:\n" + config_details)

    if lexicon:
        trainer = CustomTrainer(
            model,
            training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=lambda p: compute_metrics(p, id2label),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )
    else:
        trainer = transformers.Trainer(
            model,
            training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=lambda p: compute_metrics(p, id2label),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )

    # Add a warmup of 2 epochs to avoid initial flukes
    warmup_callback = WarmupEvalCallback(warmup_epochs=2)
    trainer.add_callback(warmup_callback)

    trainer.train()

    logger.info("\n\nVALIDATION")
    logger.info("==========")
    evaluation = trainer.evaluate()
    for label in labels:
        logger.info(f"{label}: {evaluation[METRIC_F1_SCORE][label]:.2f}")
    logger.info(f"Macro average: {evaluation[METRIC_MACRO_F1_SCORE]:.2f}")

    # Ensure distributed training cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    return trainer
```
-----------
## utils.py
```python
import os
import shutil

import pandas as pd
import nltk

from core.log import logger

def validate_args(labels, training_dataset, validation_dataset):
    assert len(labels) > 0, "Labels cannot be empty."
    assert training_dataset is not None, "Training dataset cannot be None."
    assert validation_dataset is not None, "Validation dataset cannot be None."
    logger.info("Arguments validated successfully.")

def slice_for_testing(dataset, size=1000):
    if hasattr(dataset, 'select'):  # Assuming it's a dataset with a select method
        return dataset.select(range(size))
    elif isinstance(dataset, pd.DataFrame):  # Check if it's a pandas DataFrame
        return dataset.iloc[:size]  # Use iloc to slice DataFrame rows
    else:
        raise TypeError("Unsupported dataset type. Expected Dataset or DataFrame.")

def validate_file(path):
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    if not os.path.isfile(path):
        logger.error(f"Path is not a file: {path}")
        raise ValueError(f"Path is not a valid file: {path}")

def normalize_token(token):
    return token.lower().lstrip("ġ")

def skip_invalid_line(line, reason):
    logger.warning(f"Skipping line due to {reason}: {line}")

def read_file_lines(path, skip_header=False, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()
    return lines[1:] if skip_header else lines

def download_nltk_resources():
    """Ensure necessary NLTK resources are available."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def clear_directory(directory):
    """Remove all files and subdirectories in the given directory."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
```
-----------
## main.py


```python
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.config import MODEL_CONFIG
from core.utils import download_nltk_resources
from core.runner import run_training
from core.cli import parse_args
import optuna

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.parallel")
import logging
from core.log import logger


def main() -> None:

    # Load model-specific configuration
    model_group = "presence"
    model_config = MODEL_CONFIG[model_group]

    # Define CLI arguments for training script
    args = parse_args(prog_name=model_group)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Download resources only once
    download_nltk_resources()

    def objective(trial):
        # Suggest hyperparameters
        """
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
        batch_size = trial.suggest_categorical("batch_size", [2, 4])
        gradient_accumulation_steps = 2 if batch_size == 4 else 4
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.1, 0.3, log=True)
        """

        # Run training with these hyperparameters
        trainer = run_training(
            pretrained_model=model_config["pretrained_model"],
            labels=model_config["labels"],
            training_dataset_path=args.training_dataset,
            validation_dataset_path=args.validation_dataset,
            lexicon=args.lexicon,
            previous_sentences=args.previous_sentences,
            linguistic_features=args.linguistic_features,
            ner_features=args.ner_features,
            model_name=args.model_name,
            model_directory=args.model_directory,
            multilayer=args.multilayer,
            slice_data=args.slice,
            batch_size=4,
            num_train_epochs=10,
            learning_rate=2e-05,
            weight_decay=0.15,
            gradient_accumulation_steps=4,
            early_stopping_patience=4,
            #custom_stopwords = model_config["custom_stopwords"],
            augment_data=args.augment_data,
            topic_detection=args.topic_detection
        )

        # Evaluate and return metric
        eval_results = trainer.evaluate()
        macro_avg_f1 = eval_results["eval_marco-avg-f1-score"]
        return macro_avg_f1

    # If user passes --optimize, run Optuna optimization
    if args.optimize:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best params: {study.best_params}")

    else:
        # Normal training run
        model_group = "presence"
        model_config = MODEL_CONFIG[model_group]
    
        # Run the training pipeline
        run_training(
            pretrained_model=model_config["pretrained_model"],
            labels=model_config["labels"],
            training_dataset_path=args.training_dataset,
            validation_dataset_path=args.validation_dataset,
            lexicon=args.lexicon,
            previous_sentences=args.previous_sentences,
            linguistic_features=args.linguistic_features,
            ner_features=args.ner_features,
            model_name=args.model_name,
            model_directory=args.model_directory,
            multilayer=args.multilayer,
            slice_data=args.slice,
            batch_size=4,
            num_train_epochs=10,
            learning_rate=2e-05,
            weight_decay=0.15,
            gradient_accumulation_steps=4,
            early_stopping_patience=4,
            #custom_stopwords = model_config["custom_stopwords"],
            augment_data=args.augment_data,
            topic_detection=args.topic_detection
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
```
-----------
