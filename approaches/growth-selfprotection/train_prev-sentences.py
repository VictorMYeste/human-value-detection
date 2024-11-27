import argparse
import datasets
import numpy
import os
import pandas
import sys
import tempfile
import torch
import transformers
from transformers import EarlyStoppingCallback

# GENERIC

labels = [ "Growth Anxiety-Free", "Self-Protection Anxiety-Avoidance" ]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)} 

def load_dataset(directory, tokenizer, load_labels=True):
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    labels_file_path = os.path.join(directory, "labels-cat.tsv")
    
    data_frame = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0)

    # Fix TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
    data_frame['Text'] = data_frame['Text'].fillna('')

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
    
    data_frame["ConcatenatedText"] = concatenated_texts

    encoded_sentences = tokenizer(data_frame["ConcatenatedText"].to_list(), truncation=True, max_length=512)

    if load_labels and os.path.isfile(labels_file_path):
        labels_frame = pandas.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)
        # Extract only the new label columns
        labels_matrix = numpy.zeros((labels_frame.shape[0], len(labels)))
        for idx, label in enumerate(labels):
            if label in labels_frame.columns:
                labels_matrix[:, idx] = (labels_frame[label] >= 0.5).astype(int)
        encoded_sentences["labels"] = labels_matrix.tolist()

    encoded_sentences = datasets.Dataset.from_dict(encoded_sentences)
    return encoded_sentences, data_frame["Text-ID"].to_list(), data_frame["Sentence-ID"].to_list()


# TRAINING

def train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name=None, batch_size=2, num_train_epochs=9, learning_rate=2.07e-05, weight_decay=1.02e-05):
    def compute_metrics(eval_prediction):
        prediction_scores, label_scores = eval_prediction
        predictions = prediction_scores >= 0.0 # sigmoid
        labels = label_scores >= 0.5

        f1_scores = {}
        for i in range(predictions.shape[1]):
            predicted = predictions[:,i].sum()
            true = labels[:,i].sum()
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
        gradient_accumulation_steps = 4,
        fp16=True,
        ddp_find_unused_parameters=False # Optimized for static models
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, problem_type="multi_label_classification",
        num_labels=len(labels), id2label=id2label, label2id=label2id)
    if torch.cuda.is_available():
        print("Using cuda")
        model = model.to('cuda')

    print("TRAINING")
    print("========")

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)

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

# COMMAND LINE INTERFACE

cli = argparse.ArgumentParser(prog="DeBERTa")
cli.add_argument("-t", "--training-dataset", required=True)
cli.add_argument("-v", "--validation-dataset")
cli.add_argument("-m", "--model-name")
cli.add_argument("-o", "--model-directory")
args = cli.parse_args()

pretrained_model = "microsoft/deberta-base"
tokenizer = transformers.DebertaTokenizer.from_pretrained(pretrained_model, truncation_side = "left")

# Load the training dataset
training_dataset, training_text_ids, training_sentence_ids = load_dataset(args.training_dataset, tokenizer)

# Load the validation dataset
validation_dataset = training_dataset
if args.validation_dataset != None:
    validation_dataset, validation_text_ids, validation_sentence_ids = load_dataset(args.validation_dataset, tokenizer)

# Slicing for testing purposes

#training_dataset = training_dataset.select(range(10))
#validation_dataset = validation_dataset.select(range(10))

# Train and evaluate
trainer = train(training_dataset, validation_dataset, pretrained_model, tokenizer, model_name = args.model_name)

# Save the model if required
if args.model_name != None:
    print("\n\nUPLOAD to https://huggingface.co/" + args.model_name + " (using HF_TOKEN environment variable)")
    print("======")
    #trainer.push_to_hub()

if args.model_directory != None:
    print("\n\nSAVE to " + args.model_directory)
    print("======")
    trainer.save_model(args.model_directory)