import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"  # ↓ fights fragmentation :contentReference[oaicite:0]{index=0}
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # Add the project root to sys.path
from safetensors.torch import load_file
import pandas
import torch
import spacy
import transformers
from tqdm import tqdm  # Import tqdm for progress bar
import random
import numpy as np

from core.config import MODEL_CONFIG
from core.cli import parse_args
from core.models import EnhancedDebertaModel
from core.dataset_utils import compute_ner_features, compute_linguistic_features, compute_lexicon_scores

import sys
import logging
from core.log import logger

def predict(text_instances, nlp, tokenizer, model, sigmoid, labels, id2label, device):
    """
    Predicts the label probabilities for each sentence
    Memory friendly: FP16 model, small batches, inference_mode, eager clean-up.
    """

    BATCH = 16
    results = []

    with torch.inference_mode():           # lighter than no_grad
        for i in range(0, len(text_instances), BATCH):
            batch = text_instances[i:i + BATCH]

            # "text" contains all sentences (plain strings) of a single text in order (same Text-ID in the input file)
            encoded_sentences = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=512)

            encoded_sentences.pop("token_type_ids", None) # not used by DeBERTa

            # Move the encoded sentences to the GPU (if available)
            encoded_sentences = {key: value.to(device) for key, value in encoded_sentences.items()}

            # Compute linguistic and NER features for each sentence separately
            #linguistic_features = torch.tensor([compute_linguistic_features(sentence, nlp) for sentence in text_instances], dtype=torch.float32)
            #ner_features = torch.tensor([compute_ner_features(sentence, nlp) for sentence in text_instances], dtype=torch.float32)

            # Add features to encoded sentences
            #encoded_sentences["linguistic_features"] = linguistic_features
            #encoded_sentences["ner_features"] = ner_features

            logits = model(**encoded_sentences)["logits"]             # FP16 weights, FP16 activations
            probs  = sigmoid(logits).cpu()               # move only tiny tensor to host

            for row in probs:
                results.append({id2label[idx]: score for idx, score in enumerate(row.tolist())})

            # ---- free GPU buffers ASAP
            del encoded_sentences, logits, probs
            torch.cuda.empty_cache()
        
    return results

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

def run(model_group: str = "presence") -> None:
    """
    End-to-end inference entry point.

    Parameters
    ----------
    model_group : str, optional
        Key in `core.config.MODEL_CONFIG`.  Change this when you copy the tiny
        `predict.py` wrapper into another model folder.
    """

    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    logger.info(f"Using device: {device}")

    # Load model-specific configuration
    model_config = MODEL_CONFIG[model_group]

    # Define CLI arguments for training script
    args = parse_args(prog_name=model_group)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Optionally set the seed if provided
    if args.seed is not None:
        logger.info(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # GENERIC

    labels = model_config["labels"]
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)} 

    # SETUP

    model_name = args.model_name
    model_path = "models/" + model_name # load from directory
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
        #ner_feature_dim=18,
    )

    # Load model weights manually
    state_dict = load_file(model_path+"/model.safetensors")
    model.load_state_dict(state_dict)
    model.half().eval().to(device)
    sigmoid = torch.nn.Sigmoid()

    # code not executed by tira-run-inference-server (which directly calls 'predict(text)')
    if "TIRA_INFERENCE_SERVER" not in os.environ:

        dataset_dir = args.test_dataset
        output_dir = args.output_directory

        labeled_instances = []
        input_file = os.path.join(dataset_dir, "sentences.tsv")
        grouped_texts =  pandas.read_csv(input_file, sep='\t', header=0, index_col=None).groupby("Text-ID")

        nlp = spacy.load("en_core_web_sm")

        with tqdm(total=len(grouped_texts), desc="Processing Texts") as pbar:
            
            for text_instances in grouped_texts:
                # label the instances of each text separately
                labeled_instances.extend(label(text_instances[1].sort_values("Sentence-ID").to_dict("records"), nlp, tokenizer, model, sigmoid, labels, id2label, device))
                pbar.update(1)  # Update progress bar

            writeRun(labeled_instances, output_dir, model_name)