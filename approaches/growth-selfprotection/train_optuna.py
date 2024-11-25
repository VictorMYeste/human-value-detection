import argparse
import datasets
import numpy
import os
import pandas
import sys
import tempfile
import torch
import transformers
import optuna

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

    encoded_sentences = tokenizer(data_frame["Text"].to_list(), truncation=True, max_length=512)

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
# OPTUNA OBJECTIVE FUNCTION

def objective(trial, training_dataset, validation_dataset, pretrained_model, tokenizer):
    
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

    # Hyperparameter suggestions from Optuna
    batch_size = 4
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    weight_decay = 0.01
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
    gradient_accumulation_steps = 2

    # TrainingArguments with suggested hyperparameters
    args = transformers.TrainingArguments(
        output_dir="./results",
        save_strategy="no", # Speed up optimization by not saving models
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=False,
        metric_for_best_model='marco-avg-f1-score',
        gradient_accumulation_steps = gradient_accumulation_steps,
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
    trainer = transformers.Trainer(model, args,
        train_dataset=training_dataset, eval_dataset=validation_dataset,
        compute_metrics=compute_metrics, tokenizer=tokenizer)

    trainer.train()

    print("\n\nVALIDATION")
    print("==========")
    evaluation = trainer.evaluate()
    
    return evaluation["eval_marco-avg-f1-score"]  # Metric to maximize

# COMMAND LINE INTERFACE

cli = argparse.ArgumentParser(prog="DeBERTa")
cli.add_argument("-t", "--training-dataset", required=True)
cli.add_argument("-v", "--validation-dataset")
cli.add_argument("-m", "--model-name")
cli.add_argument("-o", "--model-directory")
args = cli.parse_args()

pretrained_model = "microsoft/deberta-base"
tokenizer = transformers.DebertaTokenizer.from_pretrained(pretrained_model)

# Load the training dataset
training_dataset, training_text_ids, training_sentence_ids = load_dataset(args.training_dataset, tokenizer)

# Load the validation dataset
validation_dataset = training_dataset
if args.validation_dataset != None:
    validation_dataset, validation_text_ids, validation_sentence_ids = load_dataset(args.validation_dataset, tokenizer)

# Slicing for testing purposes

#training_dataset = training_dataset.select(range(10))
#validation_dataset = validation_dataset.select(range(10))

# OPTIMIZE WITH OPTUNA
study = optuna.create_study(direction="maximize")  # Maximize F1-score
study.optimize(lambda trial: objective(trial, training_dataset, validation_dataset, pretrained_model, tokenizer), n_trials=50)  # Number of trials

# Print the best hyperparameters
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
print(f"  Params: {study.best_trial.params}")

# Save the model if required
if args.model_name != None:
    print("\n\nUPLOAD to https://huggingface.co/" + args.model_name + " (using HF_TOKEN environment variable)")
    print("======")
    #trainer.push_to_hub()

if args.model_directory != None:
    print("\n\nSAVE to " + args.model_directory)
    print("======")
    trainer.save_model(args.model_directory)