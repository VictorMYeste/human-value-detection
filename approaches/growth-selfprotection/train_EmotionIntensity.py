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

# GENERIC

labels = [ "Growth Anxiety-Free", "Self-Protection Anxiety-Avoidance" ]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

class EnhancedDebertaModel(nn.Module):
    def __init__(self, pretrained_model, num_labels, id2label, label2id, num_emotions):
        super(EnhancedDebertaModel, self).__init__()
        self.transformer = transformers.AutoModel.from_pretrained(pretrained_model)
        self.emotional_layer = nn.Linear(num_emotions, 128)  # Map EmoLex embeddings to 128 dimensions
        self.classification_head = nn.Linear(self.transformer.config.hidden_size + 128, num_labels)
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

    def forward(self, input_ids, attention_mask, emotional_features=None, labels=None):
        transformer_output = self.transformer(input_ids, attention_mask=attention_mask).pooler_output
        emotional_output = self.emotional_layer(emotional_features)
        combined_output = torch.cat([transformer_output, emotional_output], dim=-1)
        combined_output = self.dropout(combined_output)
        logits = self.classification_head(combined_output)
        return {"logits": logits}
    
class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss = binary_cross_entropy_with_logits(logits, labels.float())  # BCE loss
        return (loss, outputs) if return_outputs else loss

def load_emotional_embeddings():
    """Load NRC Emotion Intensity Lexicon into a dictionary."""
    intensity_lexicon_path = "../../lexicons/NRC-Emotion-Intensity-Lexicon-v1.txt"
    intensity_scores = {}
    with open(intensity_lexicon_path, "r") as f:
        for line in f.readlines()[1:]:  # Skip header
            word, emotion, score = line.strip().split("\t")
            score = float(score)  # Intensity scores are continuous
            if word not in intensity_scores:
                intensity_scores[word] = {emotion: 0.0 for emotion in ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]}
            intensity_scores[word][emotion] = score
    return intensity_scores

def compute_emotional_scores(text, intensity_scores, num_emotions=8):
    """Compute the average Intensity scores for the 8 emotions for a given text."""
    tokens = text.split()  # Tokenize by whitespace
    scores = {emotion: 0.0 for emotion in ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]}
    count = 0
    for token in tokens:
        if token.lower() in intensity_scores:
            count += 1
            for emotion in scores.keys():
                scores[emotion] += intensity_scores[token].get(emotion, 0.0)
    if count > 0:
        for key in scores:
            scores[key] /= count  # Normalize by token count
    return list(scores.values())  # Return as a list of 8 scores

def load_dataset_with_emotions(directory, tokenizer, intensity_scores, load_labels=True):
    """Load dataset and add emotional embeddings."""
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    labels_file_path = os.path.join(directory, "labels-cat.tsv")
    
    data_frame = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0)

    # Fill missing text
    data_frame['Text'] = data_frame['Text'].fillna('')

    # Compute emotional embeddings for each sentence
    num_emotions = 8  # Emotion Intensity Lexicon has 8 emotions
    data_frame['Emotional_Scores'] = data_frame['Text'].apply(
        lambda x: compute_emotional_scores(x, intensity_scores, num_emotions)
    )

    # Tokenize the text
    encoded_sentences = tokenizer(data_frame["Text"].to_list(), truncation=True, max_length=512)

    # Add emotional embeddings to tokenized features
    emotional_features = numpy.array(data_frame['Emotional_Scores'].to_list())
    encoded_sentences["emotional_features"] = emotional_features.tolist()

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

# TRAINING

def train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name=None, batch_size=4, num_train_epochs=10, learning_rate=5e-6, weight_decay=0.01, num_emotions=8):
    def compute_metrics(eval_prediction):
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
        gradient_accumulation_steps = 2,
        fp16=True,
        ddp_find_unused_parameters=False # Optimized for static models
    )

    model = EnhancedDebertaModel(pretrained_model, len(labels), id2label, label2id, num_emotions)
    
    if torch.cuda.is_available():
        print("Using cuda")
        model = model.to('cuda')

    print("TRAINING")
    print("========")

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)

    trainer = CustomTrainer(model, args,
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

# COMMAND LINE INTERFACE

cli = argparse.ArgumentParser(prog="DeBERTa")
cli.add_argument("-t", "--training-dataset", required=True)
cli.add_argument("-v", "--validation-dataset")
cli.add_argument("-m", "--model-name")
cli.add_argument("-o", "--model-directory")
args = cli.parse_args()

pretrained_model = "microsoft/deberta-base"
tokenizer = transformers.DebertaTokenizer.from_pretrained(pretrained_model)

# Load EmoLex
intensity_scores = load_emotional_embeddings()

# Number of emotions in the Emotion Intensity Lexicon
num_emotions = 8  # The lexicon includes 8 emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust

# Load the training dataset
training_dataset, training_text_ids, training_sentence_ids = load_dataset_with_emotions(args.training_dataset, tokenizer, intensity_scores)

# Load the validation dataset
validation_dataset = training_dataset
if args.validation_dataset != None:
    validation_dataset, validation_text_ids, validation_sentence_ids = load_dataset_with_emotions(args.validation_dataset, tokenizer, intensity_scores)

# Slicing for testing purposes

#training_dataset = training_dataset.select(range(10))
#validation_dataset = validation_dataset.select(range(10))

# Train and evaluate
trainer = train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name = args.model_name, num_emotions=num_emotions)

# Save the model if required
if args.model_name != None:
    print("\n\nUPLOAD to https://huggingface.co/" + args.model_name + " (using HF_TOKEN environment variable)")
    print("======")
    #trainer.push_to_hub()

if args.model_directory != None:
    print("\n\nSAVE to " + args.model_directory)
    print("======")
    trainer.save_model(args.model_directory)