import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.config import MODEL_CONFIG

import datasets
import pandas
import numpy
import torch
import transformers
from safetensors.torch import load_file
import spacy
from tqdm import tqdm  # Import tqdm for progress bar

from core.models import EnhancedDebertaModel
from core.dataset_utils import compute_ner_features, compute_linguistic_features, compute_lexicon_scores

# Load model-specific configuration
model_group = "presence"
model_config = MODEL_CONFIG[model_group]

# GENERIC

labels = model_config["labels"]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)} 

# SETUP

model_path = "models/Text-NER" # load from directory
#model_path = "JohannesKiesel/valueeval24-bert-baseline-en" # load from huggingface hub

# Load the configuration first
config = transformers.AutoConfig.from_pretrained(model_path)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
# model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

# Instantiate the custom model with the loaded config
model = EnhancedDebertaModel(
    pretrained_model=model_config["pretrained_model"],
    config=config,
    num_labels=len(config.id2label),
    id2label=config.id2label,
    label2id=config.label2id,
    ner_feature_dim=18,
)

# Load model weights manually
state_dict = load_file(model_path+"/model.safetensors")
model.load_state_dict(state_dict)
model.eval()

sigmoid = torch.nn.Sigmoid()

# PREDICTION

def predict(text_instances, nlp):
    """ Predicts the label probabilities for each sentence """
    # "text" contains all sentences (plain strings) of a single text in order (same Text-ID in the input file)
    encoded_sentences = tokenizer(text_instances, truncation=True, padding=True, return_tensors="pt")

    # Compute linguistic and NER features for each sentence separately
    #linguistic_features = torch.tensor([compute_linguistic_features(sentence, nlp) for sentence in text_instances], dtype=torch.float32)
    ner_features = torch.tensor([compute_ner_features(sentence, nlp) for sentence in text_instances], dtype=torch.float32)

    # Add features to encoded sentences
    #encoded_sentences["linguistic_features"] = linguistic_features
    encoded_sentences["ner_features"] = ner_features

    # Forward pass through the model
    model_output = model(**encoded_sentences)
    
    sentences_predictions = sigmoid(model_output["logits"])

    labels = []
    for predictions in sentences_predictions:
        pred_dict = {}
        for idx, prediction in enumerate(predictions.tolist()):
            value_label = id2label[idx]  # Using original id2label
            pred_dict[value_label] = prediction
        labels.append(pred_dict)
        
    return labels

# EXECUTION

def label(instances):
    """ Predicts the label probabilities for each instance and adds them to it """
    text_instances = [instance["Text"] for instance in instances]
    nlp = spacy.load("en_core_web_sm")
    return [{
            "Text-ID": instance["Text-ID"],
            "Sentence-ID": instance["Sentence-ID"],
            **labels
        } for instance, labels in zip(instances, predict(text_instances, nlp))]

def writeRun(labeled_instances, output_dir):
    """ Writes all (labeled) instances to the predictions.tsv in the output directory """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "Text-NER-predictions.tsv")
    pandas.DataFrame.from_dict(labeled_instances).to_csv(output_file, header=True, index=False, sep='\t')

# code not executed by tira-run-inference-server (which directly calls 'predict(text)')
if "TIRA_INFERENCE_SERVER" not in os.environ:

    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]
    labeled_instances = []
    input_file = os.path.join(dataset_dir, "sentences.tsv")
    grouped_texts =  pandas.read_csv(input_file, sep='\t', header=0, index_col=None).groupby("Text-ID")

    with tqdm(total=len(grouped_texts), desc="Processing Texts") as pbar:
        
        for text_instances in grouped_texts:
            # label the instances of each text separately
            labeled_instances.extend(label(text_instances[1].sort_values("Sentence-ID").to_dict("records")))
            pbar.update(1)  # Update progress bar

        writeRun(labeled_instances, output_dir)