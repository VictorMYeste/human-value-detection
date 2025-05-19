import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas
import torch
import transformers
from safetensors.torch import load_file
from tqdm import tqdm  # Import tqdm for progress bar
import random
import numpy as np
import spacy

from core.config import MODEL_CONFIG
from core.cli import parse_args
from core.prediction import label, writeRun
from core.models import EnhancedDebertaModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.parallel")
import logging
from core.log import logger

def main() -> None:
    # Check if CUDA is available and set the device
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    logger.info(f"Using device: {device}")

    # Load model-specific configuration
    model_group = "presence"
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
    model.eval()

    sigmoid = torch.nn.Sigmoid()

    # Move the model to the selected device (GPU or CPU)
    model.to(device)

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
            
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))