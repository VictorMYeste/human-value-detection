import argparse
import datasets
import numpy
import os
import pandas
import sys
import tempfile
import torch
import transformers
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import EarlyStoppingCallback

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
        self.lexicon_layer = nn.Linear(num_categories, 128)  # Map categories to 128 dimensions
        self.classification_head = nn.Linear(self.transformer.config.hidden_size + 128, num_labels)
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

    def forward(self, input_ids, attention_mask, lexicon_features=None, labels=None):
        """Forward pass for the enhanced model."""
        transformer_output = self.transformer(input_ids, attention_mask=attention_mask).pooler_output
        lexicon_output = self.lexicon_layer(lexicon_features)
        combined_output = torch.cat([transformer_output, lexicon_output], dim=-1)
        combined_output = self.dropout(combined_output)
        logits = self.classification_head(combined_output)
        return {"logits": logits}
    
class CustomTrainer(transformers.Trainer):
    """Custom Trainer with modified loss function for multi-label classification."""
    def compute_loss(self, model, inputs, return_outputs=False):
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
    embeddings = {}
    with open(path, "r") as f:
        for line in f.readlines()[1:]:  # Skip header
            parts = line.strip().split("\t")
            word = parts[0]
            scores = [int(x) for x in parts[1:]]  # EmoLex uses binary indicators for emotions
            embeddings[word] = scores
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

    # Separate into categories and word entries based on the `%` markers
    divider_index = lines.index("%\n")
    category_section = lines[1:divider_index]
    word_section = lines[divider_index + 1:]

    # Parse category definitions
    for line in category_section:
        parts = line.strip().split("\t")
        category_id = int(parts[0])  # Category ID is an integer
        category_name = parts[1]    # Category name
        category_names[category_id] = category_name

    # Parse word entries
    for line in word_section:
        parts = line.strip().split("\t")
        word = parts[0].lower()  # Normalize word to lowercase
        categories = list(map(int, parts[1:]))  # Convert category IDs to integers
        embeddings[word] = categories

    return embeddings, category_names

# ========================================================
# SCORE COMPUTATION FUNCTIONS
# ========================================================

def compute_vad_scores(text, lexicon_embeddings):
    """Compute the average VAD scores for a given text."""
    tokens = text.split()  # Tokenize by whitespace
    scores = {"valence": 0, "arousal": 0, "dominance": 0}
    count = 0
    for token in tokens:
        if token.lower() in lexicon_embeddings:
            count += 1
            scores["valence"] += lexicon_embeddings[token.lower()]["valence"]
            scores["arousal"] += lexicon_embeddings[token.lower()]["arousal"]
            scores["dominance"] += lexicon_embeddings[token.lower()]["dominance"]
    if count > 0:
        for key in scores:
            scores[key] /= count
    return [scores["valence"], scores["arousal"], scores["dominance"]]

def compute_emolex_scores(text, lexicon_embeddings, num_categories=10):
    """Compute the average EmoLex scores for a given text."""
    tokens = text.split()  # Tokenize by whitespace
    scores = [0] * num_categories  # EmoLex typically has 10 emotions
    count = 0
    for token in tokens:
        if token.lower() in lexicon_embeddings:
            count += 1
            for i in range(num_categories):
                scores[i] += lexicon_embeddings[token.lower()][i]
    if count > 0:
        scores = [score / count for score in scores]  # Average scores
    return scores

def compute_emotionintensity_scores(text, lexicon_embeddings):
    """Compute the average Intensity scores for the 8 emotions for a given text."""
    tokens = text.split()  # Tokenize by whitespace
    scores = {emotion: 0.0 for emotion in ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]}
    count = 0
    for token in tokens:
        if token.lower() in lexicon_embeddings:
            count += 1
            for emotion in scores.keys():
                scores[emotion] += lexicon_embeddings[token].get(emotion, 0.0)
    if count > 0:
        for key in scores:
            scores[key] /= count  # Normalize by token count
    return list(scores.values())  # Return as a list of 8 scores

def compute_worrywords_scores(text, lexicon_embeddings):
    """Compute the average worry score for a given text."""
    tokens = text.split()  # Tokenize by whitespace
    total_score = 0.0
    count = 0
    for token in tokens:
        if token.lower() in lexicon_embeddings:
            count += 1
            total_score += lexicon_embeddings[token.lower()]
    return [total_score / count] if count > 0 else [0.0]  # Return the average worry score as a list

def compute_liwc_scores(text, lexicon_embeddings, num_categories):
    """Compute the average LIWC scores for a given text."""
    tokens = text.split()  # Tokenize by whitespace
    scores = [0.0] * num_categories

    for token in tokens:
        token_lower = token.lower()
        matched = False

        # Exact match
        if token_lower in lexicon_embeddings:
            matched = True
            for category in lexicon_embeddings[token_lower]:
                scores[category - 1] += 1  # Adjust for 0-based indexing

        # Stem match
        if not matched:
            for word, categories in lexicon_embeddings.items():
                if word.endswith("*") and token_lower.startswith(word[:-1]):
                    for category in categories:
                        scores[category - 1] += 1  # Adjust for 0-based indexing
                    break

    # Normalize scores by total words (if count > 0)
    if tokens:
        scores = [score / len(tokens) for score in scores]

    return scores

# ========================================================
# DATASET LOADING FUNCTION
# ========================================================

def load_dataset(directory, tokenizer, lexicon_embeddings = {}, load_labels=True, num_categories=1):
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
                concatenated_texts.append(data_frame.iloc[idx]["Text"])
            elif idx == 1:
                # Second sentence, only one preceding sentence
                concatenated_texts.append(
                    data_frame.iloc[idx - 1]["Text"] + " [SEP] " + data_frame.iloc[idx]["Text"]
                )
            else:
                # Concatenate the two preceding sentences with a separator
                concatenated_texts.append(
                    data_frame.iloc[idx - 2]["Text"] + " " +
                    data_frame.iloc[idx - 1]["Text"] + " [SEP] " +
                    data_frame.iloc[idx]["Text"]
                )
        
        texts = concatenated_texts
    
    encoded_sentences = tokenizer(texts.to_list(), truncation=True, max_length=512)

    # Compute lexicon embeddings for each sentence
    if args.lexicon != None:
        if args.lexicon is "VAD":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_vad_scores(x, lexicon_embeddings)
            )
        elif args.lexicon is "EmoLex":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_emolex_scores(x, lexicon_embeddings, num_categories)
            )
        elif args.lexicon is "EmotionIntensity":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_emotionintensity_scores(x, lexicon_embeddings)
            )
        elif args.lexicon is "WorryWords":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_worrywords_scores(x, lexicon_embeddings)
            )
        elif args.lexicon is "LIWC":
            data_frame['Lexicon_Scores'] = texts.apply(
                lambda x: compute_liwc_scores(x, lexicon_embeddings, num_categories)
            )
        # Add lexicon embeddings to tokenized features
        lexicon_features = numpy.array(texts.to_list())
        encoded_sentences["lexicon_features"] = lexicon_features.tolist()

    # Load labels if available
    if load_labels and os.path.isfile(labels_file_path):
        labels_frame = pandas.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)
        labels_matrix = numpy.zeros((labels_frame.shape[0], len(labels)))
        for idx, label in enumerate(labels):
            if label in labels_frame.columns:
                labels_matrix[:, idx] = (labels_frame[label] >= 0.5).astype(int)
        encoded_sentences["labels"] = labels_matrix.tolist()

    encoded_sentences = datasets.Dataset.from_dict(encoded_sentences)
    return encoded_sentences, data_frame["Text-ID"].to_list(), data_frame["Sentence-ID"].to_list()

# ========================================================
# TRAINING FUNCTION
# ========================================================

def train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name=None, batch_size=4, num_train_epochs=9, learning_rate=2.07e-05, weight_decay=1.02e-05, gradient_accumulation_steps = 2, num_categories=1):
    """Train the model and evaluate performance."""
    def compute_metrics(eval_prediction):
        """Compute evaluation metrics like F1-score."""
        prediction_scores, label_scores = eval_prediction
        predictions = torch.sigmoid(torch.tensor(prediction_scores)) >= 0.5  # Apply sigmoid
        labels = label_scores >= 0.5

        f1_scores = {}
        for i in range(predictions.shape[1]):
            predicted = predictions[:, i].sum()
            true = labels[:, i].sum()
            true_positives = numpy.logical_and(predictions[:,i], labels[:,i]).sum()
            precision = 0 if predicted == 0 else true_positives / predicted
            recall = 0 if true == 0 else true_positives / true
            f1_scores[id2label[i]] = round(0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall), 2)
        macro_average_f1_score = round(numpy.mean(list(f1_scores.values())), 2)

        return {'f1-score': f1_scores, 'marco-avg-f1-score': macro_average_f1_score}

    if args.previous_sentences:
        batch_size = 2
        gradient_accumulation_steps = 4

    output_dir = tempfile.TemporaryDirectory()
    args = transformers.TrainingArguments(
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

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)

    if args.lexicon != None:
        trainer = CustomTrainer(model, args,
        train_dataset=training_dataset, eval_dataset=validation_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer,
        callbacks=[early_stopping])
    else:
        trainer = transformers.Trainer(model, args,
        train_dataset=training_dataset, eval_dataset=validation_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer,
        callbacks=[early_stopping])

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
    if args.lexicon is "VAD":
        path = "../../lexicons/NRC-VAD-Lexicon.txt"
        lexicon_embeddings = load_vad_embeddings(path)
        num_categories = 3
    elif args.lexicon is "EmoLex":
        path = "../../lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
        lexicon_embeddings = load_emolex_embeddings(path)
        num_categories = 10
    elif args.lexicon is "EmotionIntensity":
        path = "../../lexicons/NRC-Emotion-Intensity-Lexicon-v1.txt"
        lexicon_embeddings = load_emotionintensity_embeddings(path)
        num_categories = 8
    elif args.lexicon is "WorryWords":
        path = "../../worrywords-v1.txt"
        lexicon_embeddings = load_worrywords_embeddings(path)
    elif args.lexicon is "LIWC":
        path = "../../lexicons/liwc2015.dic"
        liwc_scores, category_names = load_liwc_embeddings(path)
        num_categories = len(category_names)

# Load the training dataset
training_dataset, training_text_ids, training_sentence_ids = load_dataset(args.training_dataset, tokenizer, lexicon_embeddings, num_categories)

# Load the validation dataset
validation_dataset = training_dataset
if args.validation_dataset != None:
    validation_dataset, validation_text_ids, validation_sentence_ids = load_dataset(args.validation_dataset, tokenizer, lexicon_embeddings, num_categories)

# Train and evaluate
trainer = train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name = args.model_name, num_categories=num_categories)

# Save the model if required
if args.model_name != None:
    print("\n\nUPLOAD to https://huggingface.co/" + args.model_name + " (using HF_TOKEN environment variable)")
    print("======")
    #trainer.push_to_hub()

if args.model_directory != None:
    print("\n\nSAVE to " + args.model_directory)
    print("======")
    trainer.save_model(args.model_directory)