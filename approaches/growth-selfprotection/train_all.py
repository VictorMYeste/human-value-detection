import argparse
import datasets
import numpy
import os
import numpy as np
import pandas
import sys
import tempfile
import torch
import transformers
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import EarlyStoppingCallback
from transformers import DataCollatorWithPadding
import spacy
import re
from collections import defaultdict

# ========================================================
# IMPORTS AND CONSTANTS
# ========================================================
# Define labels and mappings

labels = [ "Growth Anxiety-Free", "Self-Protection Anxiety-Avoidance" ]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# ========================================================
# MODEL DEFINITIONS
# ========================================================

class EnhancedDebertaModel(nn.Module):
    """Enhanced DeBERTa model with added lexicon feature layer."""
    def __init__(self, pretrained_model, num_labels, id2label, label2id, num_categories):
        super(EnhancedDebertaModel, self).__init__()
        self.transformer = transformers.AutoModel.from_pretrained(pretrained_model)
        #self.lexicon_layer = nn.Linear(num_categories, 128)  # Map categories to 128 dimensions
        self.lexicon_layer = nn.Sequential(
            nn.Linear(num_categories, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.classification_head = nn.Linear(self.transformer.config.hidden_size + 128, num_labels)
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

    def forward(self, input_ids, attention_mask, lexicon_features=None, labels=None):
        """Forward pass for the enhanced model."""
        hidden_state = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        transformer_output = hidden_state[:, 0, :] # CLS token representation

        if lexicon_features is not None:
            lexicon_output = torch.relu(self.lexicon_layer(lexicon_features))
            combined_output = torch.cat([transformer_output, lexicon_output], dim=-1)
        else:
            combined_output = transformer_output
        
        #logits_with_lexicon = self.classification_head(combined_output)
        #logits_without_lexicon = self.classification_head(transformer_output)
        #print(f"Logits with lexicon: {logits_with_lexicon[:5]}")
        #print(f"Logits without lexicon: {logits_without_lexicon[:5]}")

        combined_output = self.dropout(combined_output)
        logits = self.classification_head(combined_output)
        return {"logits": logits}
    
class CustomTrainer(transformers.Trainer):
    """Custom Trainer with modified loss function for multi-label classification."""
    def compute_loss(self, model, inputs, return_outputs=False):
        #print(f"Input IDs Shape: {inputs['input_ids'].shape}")
        #print(f"Attention Mask Shape: {inputs['attention_mask'].shape}")
        #print(f"Lexicon Features Shape: {inputs['lexicon_features'].shape}")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss = binary_cross_entropy_with_logits(logits, labels.float())  # BCE loss
        return (loss, outputs) if return_outputs else loss

# ========================================================
# EMBEDDINGS LOADING FUNCTIONS
# ========================================================

def load_vad_embeddings(path):
    """Load NRC VAD lexicon into a dictionary."""
    embeddings = {}
    with open(path, "r") as f:
        for line in f.readlines()[1:]:  # Skip header
            word, valence, arousal, dominance = line.strip().split("\t")
            embeddings[word] = {
                "valence": float(valence),
                "arousal": float(arousal),
                "dominance": float(dominance)
            }
    return embeddings

def load_emolex_embeddings(path):
    """Load EmoLex lexicon into a dictionary."""
    # Define the emotions in the order they appear in the file
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]
    embeddings = {}
    with open(path, "r") as f:
        for line in f.readlines():
            word, emotion, score = line.strip().split("\t")
            if word not in embeddings:
                embeddings[word] = [0] * len(emotions)
            if emotion in emotions:
                emotion_idx = emotions.index(emotion)
                embeddings[word][emotion_idx] = int(score)
    return embeddings

def load_emotionintensity_embeddings(path):
    """Load NRC Emotion Intensity Lexicon into a dictionary."""
    embeddings = {}
    with open(path, "r") as f:
        for line in f.readlines()[1:]:  # Skip header
            word, emotion, score = line.strip().split("\t")
            score = float(score)  # Intensity scores are continuous
            if word not in embeddings:
                embeddings[word] = {emotion: 0.0 for emotion in ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]}
            embeddings[word][emotion] = score
    return embeddings

def load_worrywords_embeddings(path):
    """Load NRC WorryWords Lexicon into a dictionary."""
    embeddings = {}
    with open(path, "r") as f:
        for line in f.readlines()[1:]:  # Skip header
            parts = line.strip().split("\t")
            term = parts[0]
            mean_score = float(parts[1])  # Mean score column
            embeddings[term.lower()] = mean_score
    return embeddings

def load_liwc_embeddings(path):
    """Load LIWC dictionary and process it into usable categories and scores."""
    embeddings = {}
    category_names = {}

    with open(path, "r") as f:
        lines = f.readlines()
    
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
                print(f"Skipping invalid category line: {line}")

    # Parse Word Section
    word_section = lines[divider_index + 1:]
    # Parse word entries
    for line in word_section:
        parts = line.strip().split("\t")
        if len(parts) < 2:  # Ensure there is at least a word and one category
            print(f"Skipping invalid word line: {line}")
            continue

        word = parts[0].lower()  # Normalize the word to lowercase
        try:
            categories = [int(category) for category in parts[1:] if category.strip().isdigit()]
            if categories:
                embeddings[word] = categories
            else:
                print(f"Skipping line due to non-numeric categories: {line}")
        except ValueError:
            print(f"Skipping invalid category IDs in word line: {line}")
        
    # Debugging output
    #print("Category Names Sample:", list(category_names.items())[:5])
    #print("Embeddings Sample:", list(embeddings.items())[:5])

    return embeddings, category_names

def load_mfd_embeddings(path):
    """Load the Moral Foundations Dictionary into a usable format."""
    embeddings = {}
    current_category = None

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

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

    return embeddings


# ========================================================
# SCORE COMPUTATION FUNCTIONS
# ========================================================

def compute_vad_scores(text, lexicon_embeddings, tokenizer):
    """Compute the average VAD scores for a given text."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [token.lower().lstrip("ġ") for token in tokens]
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
    normalized_tokens = [token.lower().lstrip("ġ") for token in tokens]
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

def compute_emotionintensity_scores(text, lexicon_embeddings, tokenizer):
    """Compute the average Intensity scores for the 8 emotions for a given text."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [token.lower().lstrip("ġ") for token in tokens]

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

def compute_worrywords_scores(text, lexicon_embeddings, tokenizer):
    """Compute the average worry score for a given text."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [token.lower().lstrip("ġ") for token in tokens]
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
    normalized_tokens = [token.lower().lstrip("ġ") for token in tokens]
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

def compute_linguistic_features(text):
    """Compute expanded linguistic and discourse features for moral value prediction."""
    doc = nlp(text)
    
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

def compute_mfd_scores(text, mfd_embeddings, tokenizer):
    """Compute the count of words matching each Moral Foundation dimension."""
    tokens = tokenizer.tokenize(text)
    normalized_tokens = [token.lower().lstrip("ġ") for token in tokens]

    scores = defaultdict(int)
    
    # Compute counts for each category
    for token in normalized_tokens:
        if token.upper() in mfd_embeddings:
            category = mfd_embeddings[token.upper()]
            scores[category] += 1

    return dict(scores)

# ========================================================
# DATASET LOADING FUNCTION
# ========================================================

def load_dataset(directory, tokenizer, lexicon_embeddings = {}, num_categories=1, load_labels=True):
    """Load dataset and add lexicon embeddings if specified."""
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    labels_file_path = os.path.join(directory, "labels-cat.tsv")
    
    data_frame = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0)

    # Fill missing text
    data_frame['Text'] = data_frame['Text'].fillna('')

    # Tokenize the text
    texts = data_frame["Text"]

    if args.previous_sentences:
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

    # Compute Linguistic features
    if args.linguistic_features:
        data_frame['Linguistic_Scores'] = texts.apply(compute_linguistic_features)

    # Compute lexicon embeddings for each sentence
    if args.lexicon != None:
        if args.lexicon == "VAD":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_vad_scores(x, lexicon_embeddings, tokenizer)
            )
        elif args.lexicon == "EmoLex":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_emolex_scores(x, lexicon_embeddings, tokenizer, num_categories)
            )
        elif args.lexicon == "EmotionIntensity":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_emotionintensity_scores(x, lexicon_embeddings, tokenizer)
            )
        elif args.lexicon == "WorryWords":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_worrywords_scores(x, lexicon_embeddings, tokenizer)
            )
        elif args.lexicon == "LIWC":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_liwc_scores(x, lexicon_embeddings, tokenizer, num_categories)
            )
        elif args.lexicon == "MFD":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_mfd_scores(x, lexicon_embeddings, tokenizer)
            )

        data_frame['Lexicon_Scores'] = data_frame['Lexicon_Scores'].apply(
            lambda x: x if isinstance(x, list) and len(x) == num_categories else [0.0] * num_categories
        )

        #print(f"Sample Lexicon Scores: {data_frame['Lexicon_Scores'].head()}")
        #print(f"Lexicon features shape: {len(data_frame['Lexicon_Scores'][0])}")
    
    if args.lexicon != None and args.linguistic_features:
        # Add linguistic features to lexicon embeddings (if any)
        combined_features = [
            np.concatenate((lex, syntactic)) for lex, syntactic in zip(data_frame['Lexicon_Scores'], data_frame['Linguistic_Scores'])
        ]
    elif args.lexicon != None:
        # Add lexicon embeddings to tokenized features
        combined_features = data_frame['Lexicon_Scores']
    elif args.linguistic_features:
        # Add Linguistic embeddings to tokenized features
        combined_features = data_frame['Linguistic_Scores']

    #print(f"Combined features shape: {len(combined_features[0])}")
    
    encoded_sentences["lexicon_features"] = combined_features

    # Load labels if available
    if load_labels and os.path.isfile(labels_file_path):
        labels_frame = pandas.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)
        labels_matrix = numpy.zeros((labels_frame.shape[0], len(labels)))
        for idx, label in enumerate(labels):
            if label in labels_frame.columns:
                labels_matrix[:, idx] = (labels_frame[label] >= 0.5).astype(int)
        encoded_sentences["labels"] = labels_matrix.tolist()

    encoded_sentences = datasets.Dataset.from_dict(encoded_sentences)
    return encoded_sentences

# ========================================================
# TRAINING FUNCTION
# ========================================================

def train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name=None, batch_size=4, num_train_epochs=9, learning_rate=2.07e-05, weight_decay=1.02e-05, gradient_accumulation_steps = 2, num_categories=1, args=None):
    """Train the model and evaluate performance."""
    def compute_metrics(eval_prediction):
        """Compute evaluation metrics like F1-score."""
        prediction_scores, label_scores = eval_prediction
        predictions = torch.sigmoid(torch.tensor(np.array(prediction_scores))) >= 0.5
        labels = torch.tensor(label_scores) >= 0.5

        #print("Predictions:", predictions[:10])
        #print("Labels:", labels[:10])

        f1_scores = {}
        for i in range(predictions.shape[1]):
            predicted = predictions[:, i].sum().item()
            true = labels[:, i].sum().item()
            true_positives = numpy.logical_and(predictions[:,i], labels[:,i]).sum().item()
            precision = 0 if predicted == 0 else true_positives / predicted
            recall = 0 if true == 0 else true_positives / true
            f1_scores[id2label[i]] = round(0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall), 2)
        macro_average_f1_score = round(numpy.mean(list(f1_scores.values())), 2)

        return {'f1-score': f1_scores, 'marco-avg-f1-score': macro_average_f1_score}

    if args.previous_sentences:
        batch_size = 2
        gradient_accumulation_steps = 4

    output_dir = tempfile.TemporaryDirectory()
    training_args = transformers.TrainingArguments(
        output_dir=output_dir.name,
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
        fp16=True,
        ddp_find_unused_parameters=False # Optimized for static models
    )

    if args.lexicon != None:
        #print(f"num_categories passed to model: {num_categories}")
        model = EnhancedDebertaModel(pretrained_model, len(labels), id2label, label2id, num_categories)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, problem_type="multi_label_classification",
        num_labels=len(labels), id2label=id2label, label2id=label2id)
    
    if torch.cuda.is_available():
        print("Using cuda")
        model = model.to('cuda')

    print("TRAINING")
    print("========")

    early_stopping_patience = 3

    early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=0.0)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Log the training arguments
    print("\nArguments:")
    print(f"Pre-trained model: {pretrained_model}")
    print(f"Model name: {model_name if model_name else 'None'}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_train_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Previous sentences used: {'Yes' if args.previous_sentences else 'No'}")
    print(f"Using lexicon: {args.lexicon if args.lexicon else 'No'}")
    print(f"Adding linguistic features: {'Yes' if args.linguistic_features else 'No'}")
    print(f"Number of categories (lexicon): {num_categories}")
    print("\n")

    if args.lexicon != None:
        trainer = CustomTrainer(model, training_args,
        train_dataset=training_dataset, eval_dataset=validation_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer,
        data_collator=data_collator, callbacks=[early_stopping])
    else:
        trainer = transformers.Trainer(model, training_args,
        train_dataset=training_dataset, eval_dataset=validation_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer,
        data_collator=data_collator, callbacks=[early_stopping])

    trainer.train()

    print("\n\nVALIDATION")
    print("==========")
    evaluation = trainer.evaluate()
    for label in labels:
        sys.stdout.write("%-39s %.2f\n" % (label + ":", evaluation["eval_f1-score"][label]))
    sys.stdout.write("\n%-39s %.2f\n" % ("Macro average:", evaluation["eval_marco-avg-f1-score"]))

    return trainer

# ========================================================
# MAIN EXECUTION
# ========================================================

cli = argparse.ArgumentParser(prog="DeBERTa")
cli.add_argument("-t", "--training-dataset", required=True)
cli.add_argument("-v", "--validation-dataset")
cli.add_argument("-p", "--previous-sentences", action='store_true')
cli.add_argument("-s", "--linguistic-features", action='store_true')
cli.add_argument("-l", "--lexicon")
cli.add_argument("-m", "--model-name")
cli.add_argument("-o", "--model-directory")
args = cli.parse_args()

pretrained_model = "microsoft/deberta-base"

if args.previous_sentences:
    tokenizer = transformers.DebertaTokenizer.from_pretrained(pretrained_model, truncation_side = "left")
else:
    tokenizer = transformers.DebertaTokenizer.from_pretrained(pretrained_model)

lexicon_embeddings = {}
num_categories = 1
if args.lexicon != None:
    # Load Lexicon
    if args.lexicon == "VAD":
        path = "../../lexicons/NRC-VAD-Lexicon.txt"
        lexicon_embeddings = load_vad_embeddings(path)
        num_categories = 3
    elif args.lexicon == "EmoLex":
        path = "../../lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        lexicon_embeddings = load_emolex_embeddings(path)
        lexicon_embeddings = {key.lower(): value for key, value in lexicon_embeddings.items()}
        num_categories = 10
    elif args.lexicon == "EmotionIntensity":
        path = "../../lexicons/NRC-Emotion-Intensity-Lexicon-v1.txt"
        lexicon_embeddings = load_emotionintensity_embeddings(path)
        num_categories = 8
    elif args.lexicon == "WorryWords":
        path = "../../lexicons/worrywords-v1.txt"
        lexicon_embeddings = load_worrywords_embeddings(path)
    elif args.lexicon == "LIWC":
        path = "../../lexicons/liwc2015.dic"
        lexicon_embeddings, category_names = load_liwc_embeddings(path)
        num_categories = len(category_names)
        #print(f"num_categories LIWC: {num_categories}")
    elif args.lexicon == "MFD":
        path = "../../lexicons/Moral-Foundations-Dictionary.wmodel"
        lexicon_embeddings = load_mfd_embeddings(path)
        num_categories = 10

    #print("Sample Lexicon Embeddings:", list(lexicon_embeddings.items())[:10])

if args.linguistic_features:
    nlp = spacy.load("en_core_web_sm")

# Load the training dataset
training_dataset = load_dataset(args.training_dataset, tokenizer, lexicon_embeddings, num_categories)
labels_array = np.array(training_dataset['labels'])
class_distribution = labels_array.sum(axis=0)
print(f"Class distribution: {class_distribution}")

# Load the validation dataset
validation_dataset = training_dataset
if args.validation_dataset != None:
    validation_dataset = load_dataset(args.validation_dataset, tokenizer, lexicon_embeddings, num_categories)

# Slicing for testing purposes
#training_dataset = training_dataset.select(range(100))
#validation_dataset = validation_dataset.select(range(100))

# Update number of categories to include linguistic features if activated
if args.linguistic_features:
    num_linguistic_features = 17  # Total number of linguistic features
    if args.lexicon != None:
        num_categories += num_linguistic_features
    else:
        num_categories = num_linguistic_features

# Train and evaluate
trainer = train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name = args.model_name, num_categories=num_categories, args=args)

# Save the model if required
if args.model_name != None:
    print("\n\nUPLOAD to https://huggingface.co/" + args.model_name + " (using HF_TOKEN environment variable)")
    print("======")
    #trainer.push_to_hub()

if args.model_directory != None:
    print("\n\nSAVE to " + args.model_directory)
    print("======")
    trainer.save_model(args.model_directory)
