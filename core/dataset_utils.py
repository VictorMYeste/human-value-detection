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
from sklearn.feature_extraction.text import TfidfVectorizer
import numbers

from core.config import LEXICON_PATHS
from core.lexicon_utils import load_lexicon
from core.utils import validate_args, normalize_token, slice_for_testing
from core.topic_detection import TopicModeling
from core.log import logger

# ========================================================
# DATASETS PREPARATION
# ========================================================

def build_idf_map(texts):
    """
    Build a token -> IDF map from a list of raw texts using a TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r'\b\w+\b'
        # You can tweak min_df or max_df here if you like
    )
    vectorizer.fit(texts)  # Fit on training texts only

    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_

    # Build dict: token -> idf
    idf_map = {feature_names[i]: idf_scores[i] for i in range(len(feature_names))}
    return idf_map

def prune_text(text, idf_map, threshold=2.5):
    """
    Remove tokens whose IDF is below a threshold (i.e., extremely common).
    """
    tokens = text.split()  # naive whitespace split; can replace with nltk word_tokenize
    pruned_tokens = []
    for token in tokens:
        token_l = token.lower()
        idf = idf_map.get(token_l, None)
        # If token not found in the IDF map, keep it 
        # or decide a rule. For now, we keep it if None
        if idf is None or idf >= threshold:
            pruned_tokens.append(token)
    return " ".join(pruned_tokens)

def load_and_optionally_prune_df(
    dataset_path: str,
    augment_data: bool,
    slice_data: bool,
    custom_stopwords: List[str],
    token_pruning: bool,
    idf_map: Optional[Dict[str, float]] = None,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    1. Load the sentences file (augmented or normal).
    2. Slice if needed.
    3. Remove custom stopwords.
    4. Prune tokens using the provided idf_map, if token_pruning is True and idf_map is not None.
    5. Return the cleaned DataFrame.
    """
    # Decide which file to load
    if augment_data:
        sentences_file_name = "sentences-aug.tsv"
    else:
        sentences_file_name = "sentences.tsv"

    # Read DataFrame
    sentences_file_path = os.path.join(dataset_path, sentences_file_name)
    df = pd.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0)

    # Slice if needed
    if slice_data:
        df = slice_for_testing(df)

    # Fill missing text
    df['Text'] = df['Text'].fillna('')

    # Remove custom stopwords
    if custom_stopwords:
        df['Text'] = df['Text'].apply(lambda txt: remove_custom_stopwords(txt, custom_stopwords))

    # If we have an IDF map, prune
    if token_pruning and idf_map is not None:
        logger.info(f"Pruning tokens in dataset '{dataset_path}' (threshold={threshold})")

        # Compute token counts before pruning
        df["raw_token_count"] = df["Text"].apply(lambda txt: len(txt.split()))
        avg_before = df["raw_token_count"].mean()
        logger.info(f"Token pruning (validation): Average token count before pruning: {avg_before:.2f}")

        # Apply pruning
        df['Text'] = df['Text'].apply(lambda txt: prune_text(txt, idf_map, threshold=threshold))

        # Compute token counts after pruning
        df["pruned_token_count"] = df["Text"].apply(lambda txt: len(txt.split()))
        avg_after = df["pruned_token_count"].mean()
        reduction = avg_before - avg_after
        perc_reduction = (reduction / avg_before) * 100 if avg_before > 0 else 0
        logger.info(
            f"Token pruning (validation): Average token count after pruning: {avg_after:.2f} "
            f"(Reduction: {reduction:.2f} tokens, {perc_reduction:.1f}% reduction)"
        )
        df.drop(columns=["raw_token_count", "pruned_token_count"], inplace=True)

    # Store original text before modifying it
    df["Original_Text"] = df["Text"]  

    return df

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
    token_pruning: str = None
) -> Tuple[datasets.Dataset, datasets.Dataset, int]:
    """
    Build & prune (if requested) the training set, then do the same for validation
    using the same IDF map.
    """

    # ------------------------------------------------
    # (A) Load the training DataFrame, no IDF map yet
    # ------------------------------------------------
    train_df = load_and_optionally_prune_df(
        dataset_path=training_path,
        augment_data=augment_data,
        slice_data=slice_data,
        custom_stopwords=custom_stopwords,
        token_pruning=False,   # (We haven't built the IDF map yet)
        idf_map=None,
    )

    # Build the IDF map from the (unpruned) training text, if pruning is requested
    idf_map = None
    pruning_threshold = 3.0
    if token_pruning:
        logger.info(f"Pruning tokens in dataset '{training_path}' (threshold={pruning_threshold})")
        logger.info("Building IDF map from unpruned training text...")
        idf_map = build_idf_map(train_df['Text'].tolist())
        logger.debug(f"IDF map size: {len(idf_map)}")

        # Compute token counts before pruning
        train_df["raw_token_count"] = train_df["Text"].apply(lambda txt: len(txt.split()))
        avg_before = train_df["raw_token_count"].mean()
        logger.info(f"Token pruning (training): Average token count before pruning: {avg_before:.2f}")

        # Now prune the training set in-place
        train_df['Text'] = train_df['Text'].apply(lambda txt: prune_text(txt, idf_map, pruning_threshold))

        # Compute token counts after pruning
        train_df["pruned_token_count"] = train_df["Text"].apply(lambda txt: len(txt.split()))
        avg_after = train_df["pruned_token_count"].mean()
        reduction = avg_before - avg_after
        perc_reduction = (reduction / avg_before) * 100 if avg_before > 0 else 0
        logger.info(
            f"Token pruning (training): Average token count after pruning: {avg_after:.2f} "
            f"(Reduction: {reduction:.2f} tokens, {perc_reduction:.1f}% reduction)"
        )
        train_df.drop(columns=["raw_token_count", "pruned_token_count"], inplace=True)

    # ------------------------------------------------
    # (B) Convert the pruned training DataFrame -> HF dataset
    # ------------------------------------------------
    if augment_data:
        labels_file = "labels-cat-aug.tsv"
    else:
        labels_file = "labels-cat.tsv"
    labels_file_path = os.path.join(training_path, labels_file)
    
     # 1) Possibly load separate training vs. validation embeddings if we have a known “LIWC-22” style
    if lexicon in ["LIWC-22", "eMFD", "MFD-20", "MJD"]:
        # Example: We handle training and validation differently
        train_lexicon, train_num_cat = load_lexicon(lexicon, LEXICON_PATHS[lexicon+"-training"])
        val_lexicon, val_num_cat     = load_lexicon(lexicon, LEXICON_PATHS[lexicon+"-validation"])
        logger.debug(f"Lexicon produced from LIWC 22 software with train_num_cat = {train_num_cat} and val_num_cat = {val_num_cat}")
    else:
        # Fallback: single lexicon for everything (the old approach)
        train_lexicon, train_num_cat = lexicon_embeddings, num_categories
        val_lexicon, val_num_cat     = lexicon_embeddings, num_categories

    # Training dataset
    training_dataset = load_dataset(
        df=train_df,
        tokenizer=tokenizer,
        labels=labels,
        slice_data=slice_data,
        lexicon_embeddings=train_lexicon,
        num_categories=train_num_cat,
        previous_sentences=previous_sentences,
        linguistic_features=linguistic_features,
        ner_features=ner_features,
        lexicon=lexicon,
        topic_detection=topic_detection,
        labels_file_path=labels_file_path
    )

    # Log class distribution
    labels_array = np.array(training_dataset["labels"])
    logger.debug(f"Class distribution: {labels_array.sum(axis=0)}")

    # ------------------------------------------------
    # (C) Prepare the VALIDATION set, reusing the IDF map
    # ------------------------------------------------
    validation_dataset = training_dataset # default if none
    if validation_path:
        val_df = load_and_optionally_prune_df(
            dataset_path=validation_path,
            augment_data=False,
            slice_data=slice_data,
            custom_stopwords=custom_stopwords,
            token_pruning=token_pruning,  # now we do want to prune if user asked
            idf_map=idf_map,              # reuse from training
            threshold=pruning_threshold
        )

        labels_file = "labels-cat.tsv"
        val_labels_path = os.path.join(validation_path, labels_file)

        validation_dataset = load_dataset(
            df=val_df,
            tokenizer=tokenizer,
            labels=labels,
            slice_data=slice_data,
            lexicon_embeddings=val_lexicon,
            num_categories=val_num_cat,
            previous_sentences=previous_sentences,
            linguistic_features=linguistic_features,
            ner_features=ner_features,
            lexicon=lexicon,
            topic_detection=topic_detection,
            labels_file_path=val_labels_path,
            is_training=False,
        )
    
    # ------------------------------------------------
    # (D) Validate & return
    # ------------------------------------------------
    validate_args(labels, training_dataset, validation_dataset)

    logger.debug(f"Training dataset columns: {training_dataset.column_names}")
    logger.debug(f"Validation dataset columns: {validation_dataset.column_names}")
    logger.debug(f"Sample training dataset: {training_dataset[0]}")
    logger.debug(f"Sample validation dataset: {validation_dataset[0]}")

    return training_dataset, validation_dataset, train_num_cat

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

def add_previous_label_features(
    df: pd.DataFrame, 
    labels_df: pd.DataFrame, 
    labels: list,
    is_training: bool,
    model: Optional[torch.nn.Module] = None,
    tokenizer_for_dynamic: Optional[transformers.PreTrainedTokenizer] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> list[list[float]]:
    """
    For training, return ground-truth from row i-1.
    For validation (is_training=False), compute dynamically the prediction for row i-1
    by performing a sequential inference pass using the current model.
    Generates previous label features:
    - First sentence of each Text-ID: [0.0] * (2 * num_labels)
    - Second sentence: [0.0] * num_labels + previous sentence's labels
    - Third and on: prev-2 labels + prev-1 labels
    """
    num_labels = len(labels)
    prev_label_features = []

    # Ensure dataframe is sorted by Text-ID and Sentence-ID
    df = df.sort_values(by=["Text-ID", "Sentence-ID"]).reset_index(drop=True)

    logger.debug(f"Inside add_previous_label_features. is_training={is_training}")
    
    # **Training Dataset: Use Ground-Truth Labels**
    if is_training:
        label_matrix = labels_df.set_index(["Text-ID", "Sentence-ID"])[labels].to_numpy(dtype=float)
        logger.debug(f"Label matrix shape: {label_matrix.shape}")

        for index, row in df.iterrows():
            current_text_id = row["Text-ID"]
            current_sentence_id = row["Sentence-ID"]

            # **Handle first sentence of new Text-ID**
            if current_sentence_id == 1:
                feats = [0.0] * (2 * num_labels)
            
            # **Handle second sentence → Use prev-1**
            elif current_sentence_id == 2:
                prev_1_idx = labels_df[
                    (labels_df["Text-ID"] == current_text_id) & 
                    (labels_df["Sentence-ID"] == current_sentence_id - 1)
                ].index
                prev_1 = label_matrix[prev_1_idx[0]].tolist() if len(prev_1_idx) > 0 else [0.0] * num_labels
                feats = prev_1 + [0.0] * num_labels
            
            # **Handle third+ sentences → Use prev-1 and prev-2**
            else:
                prev_1_idx = labels_df[
                    (labels_df["Text-ID"] == current_text_id) & 
                    (labels_df["Sentence-ID"] == current_sentence_id - 1)
                ].index
                prev_1 = label_matrix[prev_1_idx[0]].tolist() if len(prev_1_idx) > 0 else [0.0] * num_labels
                prev_2_idx = labels_df[
                    (labels_df["Text-ID"] == current_text_id) & 
                    (labels_df["Sentence-ID"] == current_sentence_id - 2)
                ].index
                prev_2 = label_matrix[prev_2_idx[0]].tolist() if len(prev_2_idx) > 0 else [0.0] * num_labels
                feats = prev_1 + prev_2

            prev_label_features.append(feats)

    # **Validation Mode: Dynamically Predict Previous Labels**
    else:
        # Dynamic auto-regressive evaluation: compute predictions sequentially.
        if model is None or tokenizer_for_dynamic is None:
            raise ValueError("For dynamic validation, 'model' and 'tokenizer_for_dynamic' must be provided.")
        
        model.eval()

        # Step 1: Initialize Previous Predictions
        prev_pred_1 = [0.0] * (2 * num_labels)  # Placeholder for prev-1 labels
        prev_pred_2 = [0.0] * (2 * num_labels)  # Placeholder for prev-2 labels

        for index, row in df.iterrows():
            current_text_id = row["Text-ID"]
            current_sentence_id = row["Sentence-ID"]

            # Step 2: Assign Previous Labels to Features

            # **Handle first sentence of new Text-ID**
            if current_sentence_id == 1:
                feats = [0.0] * (2 * num_labels)
                prev_pred_1, prev_pred_2 = [0.0] * num_labels, [0.0] * num_labels
                logger.debug(f"sentence 1!!!! prev_pred_1 = {prev_pred_1} and prev_pred_2 = {prev_pred_2}")

            # **Handle second sentence**
            elif current_sentence_id == 2:
                feats = prev_pred_1 + [0.0] * num_labels
                logger.debug(f"sentence 2!!!! prev_pred_1 = {prev_pred_1} and prev_pred_2 = {prev_pred_2}")

            # **Handle third+ sentences**
            else:
                feats = prev_pred_1 + prev_pred_2
                logger.debug(f"sentence 3+!!!! prev_pred_1 = {prev_pred_1} and prev_pred_2 = {prev_pred_2}")

            prev_label_features.append(feats.copy())

            # Step 3: Predict the Current Sentence
            text_prev = row["Text"]
            encoded = tokenizer_for_dynamic(
                [text_prev],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            plf = torch.tensor(prev_pred_1 + prev_pred_2, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], prev_label_features=plf)
            
            logits = outputs["logits"]
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float().cpu().numpy()[0].tolist()

            # **Ensure shape consistency before storing**
            if len(pred) < num_labels:
                pred = pred + [0.0] * (num_labels - len(pred))

            #torch.cuda.synchronize()
            prev_pred_2 = prev_pred_1[:]  # Shift prev-1 to prev-2
            #torch.cuda.synchronize()
            prev_pred_1 = pred[:]  # Update prev-1 with latest prediction

            logger.debug(f"Text-ID {current_text_id} - Sentence-ID {current_sentence_id}: Stored prev_pred_1 = {prev_pred_1}")
            logger.debug(f"Text-ID {current_text_id} - Sentence-ID {current_sentence_id}: Stored prev_pred_2 = {prev_pred_2}")

    for i in range(len(prev_label_features[:10])):  # Limit to first 10 samples
        logger.debug(f"Row {i}: prev_label_features = {prev_label_features[i]}")
        logger.debug(f"Row {i}: prev_1_labels = {prev_label_features[i][:len(labels)]}")
        logger.debug(f"Row {i}: prev_2_labels = {prev_label_features[i][len(labels):]}")
    
    return prev_label_features

def load_dataset(
    df: pd.DataFrame,
    tokenizer: transformers.PreTrainedTokenizer,
    labels: list,
    slice_data: bool = False,
    lexicon_embeddings = {},
    num_categories=0,
    previous_sentences: bool = False,
    linguistic_features: bool = False,
    ner_features: bool = False,
    lexicon: str = None,
    topic_detection: str = None,
    labels_file_path: Optional[str] = None,
    is_training: bool = True,
    model: Optional[torch.nn.Module] = None,
    tokenizer_for_dynamic: Optional[transformers.PreTrainedTokenizer] = None
):
    """
    Convert a pandas DataFrame (already pruned/cleaned if needed)
    into a HuggingFace Dataset with optional extra features.
    """

    data_frame = df.copy()  # Just in case, keep local copy
    labels_df = pd.read_csv(labels_file_path, sep="\t") if labels_file_path else None

    if previous_sentences and is_training:
        concatenated_texts = []

        # Sort by Text-ID and Sentence-ID
        data_frame = data_frame.sort_values(by=["Text-ID", "Sentence-ID"]).reset_index(drop=True)
        
        for idx, row in data_frame.iterrows():
            prev_sentences = []
            current_text_id = row["Text-ID"]
            current_sentence_id = row["Sentence-ID"]

            for offset in [1, 2]:  # Now, we get prev-1 first, then prev-2
                if current_sentence_id > offset:
                    prev_row = data_frame[
                        (data_frame["Text-ID"] == current_text_id) & 
                        (data_frame["Sentence-ID"] == current_sentence_id - offset)
                    ]
                    
                    if not prev_row.empty:
                        prev_text = prev_row.iloc[0]["Text"]

                        # Get previous sentence labels
                        if is_training:
                            prev_labels = labels_df.loc[
                                (labels_df["Text-ID"] == current_text_id) & 
                                (labels_df["Sentence-ID"] == current_sentence_id - offset),
                                labels
                            ].values.flatten().tolist()
                        else:
                            prev_labels = [0.0] * 2 * len(labels)

                        # Convert labels into a readable format
                        label_str = " ".join(f"<{label}>" for label, value in zip(labels, prev_labels) if value >= 0.5)
                        label_str = label_str if label_str else "<NONE>"

                        prev_sentences.append(f"{label_str} {prev_text}")

            current_text = row["Text"]

            if prev_sentences:
                full_text = current_text + " </s> " + " </s> ".join(prev_sentences).strip()
            else:
                full_text = current_text
                
            concatenated_texts.append(full_text)

        logger.debug(f"Sample concatenated texts after preprocessing:\n{concatenated_texts[:10]}")
        
        texts = concatenated_texts
    else:
        texts = data_frame["Text"].tolist()
    
    encoded_sentences = tokenizer(texts, padding=True, truncation=True, max_length=512)
    
    # Possibly compute extra features (linguistic, NER, etc.)
    combined_features = []
    if linguistic_features or ner_features:
        logger.info("Loading en_core_web_sm for extra features")
        nlp = spacy.load("en_core_web_sm")

    # Compute Linguistic features
    if linguistic_features:
        logger.info("Adding Linguistic features")
        data_frame['Linguistic_Scores'] = [compute_linguistic_features(txt, nlp) for txt in texts]
        logger.debug(f"[LING] shape: {data_frame['Linguistic_Scores'].shape}, first: {data_frame['Linguistic_Scores'].iloc[0]}")
        logger.debug("Before appending linguistic features: combined_features has length = %d", len(combined_features))
        combined_features.append(data_frame['Linguistic_Scores'].tolist())
        logger.debug("After appending linguistic features: combined_features has length = %d", len(combined_features))

    # Compute NER features
    if ner_features:
        logger.info("Adding NER embeddings")
        data_frame["NER_Features"] = [compute_ner_embeddings(txt, nlp) for txt in texts]
        logger.debug(f"NER embedding example: {data_frame['NER_Features'].head()}")
        logger.debug(f"[NER] shape: {data_frame['NER_Features'].shape}, first: {data_frame['NER_Features'].iloc[0]}")
        combined_features.append(data_frame["NER_Features"].tolist())

    # Compute lexicon embeddings for each sentence
    if lexicon:
        # 1) Is this lexicon in the dict of token-based scorers?
        if lexicon in LEXICON_COMPUTATION_FUNCTIONS:
            # Token-based approach
            data_frame['Lexicon_Scores'] = [
                compute_lexicon_scores(txt, lexicon, lexicon_embeddings, tokenizer, num_categories)
                for txt in texts
            ]
        else:
            # 2) This must be a precomputed (row-level) lexicon (e.g. LIWC-22 software generated)
            data_frame['Lexicon_Scores'] = data_frame.apply(
                lambda row: compute_precomputed_scores(row, lexicon_embeddings, num_categories),axis=1
            )
        # If the above can produce NaNs, fix them
        data_frame['Lexicon_Scores'] = data_frame['Lexicon_Scores'].apply(
            lambda arr: [0.0 if np.isnan(x) else x for x in arr]
        )
        logger.debug(f"[LEX] shape: {data_frame['Lexicon_Scores'].shape}, first: {data_frame['Lexicon_Scores'].iloc[0]}")
        combined_features.append(data_frame['Lexicon_Scores'].tolist())

        #logger.debug(f"Sample Lexicon Scores: {data_frame['Lexicon_Scores'].head()}")
    
    # Compute Topic features
    if topic_detection:
        logger.info(f"Applying {topic_detection} for topic modeling.")
        topic_model = TopicModeling(method=topic_detection)
        topic_vectors = None
        topic_vectors = topic_model.fit_transform(texts)
        encoded_sentences["topic_features"] = topic_vectors.tolist()
        logger.debug(f"[Topic Detection] shape: {encoded_sentences['topic_features'].shape}, first: {encoded_sentences['topic_features'].iloc[0]}")
        combined_features.append(topic_vectors.tolist())
    
    # Combine all features
    if combined_features:
        merged_rows = []
        for row_tuple in zip(*combined_features):
            arrays = []
            for feat in row_tuple:
                # feat should be a list (or numpy array) of floats
                arrays.append(np.array(feat, dtype=np.float32))
            merged_rows.append(np.concatenate(arrays))
        combined_features = merged_rows
    else:
        if linguistic_features:
            num_categories += 17
        combined_features = [[0.0] * num_categories] * len(data_frame)  # Default zero vectors

    # Debug logging
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
    if len(combined_features) > 0:
        logger.debug(f"First sample’s final combined_features array:\n{combined_features[0]}")

    
    # Save them in the encoded_sentences
    if lexicon and combined_features:
        all_lengths = [len(row) for row in combined_features]  # length of each row
        unique_lengths = set(all_lengths)
        logger.debug(f"unique_lengths in combined_features: {unique_lengths}")

        # If more than one unique length, pinpoint which row(s)
        if len(unique_lengths) != 1:
            for i, length in enumerate(all_lengths):
                if length != 135:  # or whatever you expect
                    logger.error(f"Row {i} in combined_features has length {length} instead of 135!")
                    logger.error(f"Row {i}: {combined_features[i]}")
        
        for i, row in enumerate(combined_features):
            for j, val in enumerate(row):
                if not isinstance(val, numbers.Number):
                    logger.error(f"Non-numeric: {type(val)} -> {val}")


        encoded_sentences["lexicon_features"] = np.array(combined_features, dtype=np.float32).tolist()
        logger.debug(
            f"Shape of 'lexicon_features' before dataset conversion: "
            f"{np.array(encoded_sentences['lexicon_features']).shape}"
        )
    elif ner_features and combined_features:
        encoded_sentences["ner_features"] = np.array(combined_features, dtype=np.float32).tolist()

    # Validate input shapes before dataset conversion
    if combined_features:
        validate_input_shapes(encoded_sentences)

    # ------------------------------------------------
    # Load main labels for this split if available
    # ------------------------------------------------
    labels_matrix = None
    labels_df = None
    if labels_file_path and os.path.isfile(labels_file_path):
        labels_df = pd.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)
        logger.debug(f"Loaded labels file from {labels_file_path} with shape {labels_df.shape} and columns: {list(labels_df.columns)}")
        # Slicing for testing purposes
        if slice_data:
            labels_df = slice_for_testing(labels_df)
        labels_matrix = labels_df[labels].ge(0.5).astype(int).to_numpy()
        logger.debug(f"Extracted labels matrix with shape {labels_matrix.shape} from columns: {labels}")
        encoded_sentences["labels"] = labels_matrix.astype(np.float32).tolist()
    else:
        logger.warning("No labels file found at the expected path.")

    # ------------------------------------------------
    # Now build "prev_label_features"
    # ------------------------------------------------
    if previous_sentences:
        logger.info("Adding Previous Sentences Labels features")
        # Add new numeric feature for the label of the previous sentence
        if is_training:
            prev_label_feats = add_previous_label_features(
                df=df,
                labels_df=labels_df,
                labels=labels,
                is_training=is_training
                )
        else:
            prev_label_feats = [[0.0] * 2 * len(labels)] * len(df)  # Return zero features initially
            
        # Store them in the dictionary for the model
        encoded_sentences["prev_label_features"] = prev_label_feats

        prev_shape = np.array(encoded_sentences["prev_label_features"]).shape
        logger.debug(f"Final prev_label_features shape before dataset conversion: {prev_shape}")
        logger.debug(f"First few prev_label_features: {prev_label_feats[:3]}")

    # Turn into HF dataset
    dataset = datasets.Dataset.from_dict(encoded_sentences)

    logger.debug(f"Constructed dataset with keys: {list(dataset.column_names)}")
    if "labels" in dataset.column_names:
        logger.debug(f"'labels' column found with {len(dataset['labels'])} entries. Sample: {dataset['labels'][:3]}")
    else:
        logger.error("The 'labels' column is missing from the dataset!")

    return dataset

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
        scores = compute_fn(text, lexicon_embeddings, tokenizer, num_categories=num_categories)
    
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

    logger.debug(f"text_id: {text_id}, sent_id: {sent_id}")

    if key in precomputed_dict:
        return precomputed_dict[key]
    else:
        # If no match, fallback
        return [0.0] * num_categories