import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # Add the project root to sys.path
import pandas
import torch
import spacy

from core.models import EnhancedDebertaModel
from core.dataset_utils import compute_ner_features, compute_linguistic_features, compute_lexicon_scores

import sys

def predict(text_instances, nlp, tokenizer, model, sigmoid, labels, id2label, device):
    """ Predicts the label probabilities for each sentence """
    # "text" contains all sentences (plain strings) of a single text in order (same Text-ID in the input file)
    encoded_sentences = tokenizer(text_instances, truncation=True, padding=True, return_tensors="pt", max_length=512)

    # Move the encoded sentences to the GPU (if available)
    encoded_sentences = {key: value.to(device) for key, value in encoded_sentences.items()}

    # Compute linguistic and NER features for each sentence separately
    #linguistic_features = torch.tensor([compute_linguistic_features(sentence, nlp) for sentence in text_instances], dtype=torch.float32)
    #ner_features = torch.tensor([compute_ner_features(sentence, nlp) for sentence in text_instances], dtype=torch.float32)

    # Add features to encoded sentences
    #encoded_sentences["linguistic_features"] = linguistic_features
    #encoded_sentences["ner_features"] = ner_features

    # Remove 'token_type_ids' if not required by the model
    encoded_sentences.pop("token_type_ids", None)

    # Forward pass through the model (also moved to GPU)
    model = model.to(device)
    model_output = model(**encoded_sentences)
    
    sentences_predictions = sigmoid(model_output["logits"])

    labels = []
    for predictions in sentences_predictions:
        pred_dict = {}
        for idx, prediction in enumerate(predictions.tolist()):
            value_label = id2label[idx]  # Using original id2label
            pred_dict[value_label] = prediction
        labels.append(pred_dict)

    del model_output
    torch.cuda.empty_cache()
        
    return labels

def label(instances, nlp, tokenizer, model, sigmoid, labels, id2label, device):
    """ Predicts the label probabilities for each instance and adds them to it """
    text_instances = [instance["Text"] for instance in instances]
    return [{
            "Text-ID": instance["Text-ID"],
            "Sentence-ID": instance["Sentence-ID"],
            **labels
        } for instance, labels in zip(instances, predict(text_instances, nlp, tokenizer, model, sigmoid, labels, id2label, device))]

def writeRun(labeled_instances, output_dir, model_name):
    """ Writes all (labeled) instances to the predictions.tsv in the output directory """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, model_name + ".tsv")
    pandas.DataFrame.from_dict(labeled_instances).to_csv(output_file, header=True, index=False, sep='\t')