import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"  # ↓ fights fragmentation :contentReference[oaicite:0]{index=0}
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # Add the project root to sys.path
from safetensors.torch import load_file
import pandas as pd
import torch
import spacy
import transformers
from tqdm import tqdm  # Import tqdm for progress bar
import random
import numpy as np

from core.config import MODEL_CONFIG
from core.cli import parse_args
from core.models import EnhancedDebertaModel
from core.dataset_utils import add_previous_label_features, compute_ner_embeddings, LEXICON_COMPUTATION_FUNCTIONS, compute_lexicon_scores, compute_precomputed_scores, concatenate_text_with_prev_labels, compute_linguistic_features
from core.topic_detection import TopicModeling
from core.config import LEXICON_PATHS
from core.lexicon_utils import load_lexicon, load_embeddings

import sys
import logging
from core.log import logger

# ------------------------------------------------------------------
# Utilities reused from dataset_utils / models for test-time decoding
# ------------------------------------------------------------------
def build_prev_label_and_text(df, labels, model, tokenizer, device,
                              lexicon_feats=None, ling_feats=None,
                              ner_feats=None, topic_feats=None):
    """
    Sequentially predicts every sentence in df and returns two lists:
      • prev_label_feats  – list[list[float]]  (len(df) × 2·|labels|)
      • concat_texts      – list[str]          (len(df))
    The logic mirrors what DynamicPrevLabelCallback does during validation.
    """
    # 1. run auto-regressive pass to get predicted labels for every row
    prev_label_feats = add_previous_label_features(
        df=df,
        labels=labels,
        is_training=False,
        model=model,
        tokenizer_for_dynamic=tokenizer,
        lexicon_features=lexicon_feats,
        linguistic_features=ling_feats,
        ner_features=ner_feats,
        topic_features=topic_feats,
        device=device
    )

    # 2. rebuild the concatenated text that contains those labels as tags
    df = df.copy()
    df["Text"] = df["Original_Text"] if "Original_Text" in df else df["Text"]
    concat_texts = [
        concatenate_text_with_prev_labels(df, i, labels, prev_label_feats)
        for i in range(len(df))
    ]

    return prev_label_feats, concat_texts

# ------------------------------------------------------------------
# Other utilities
# ------------------------------------------------------------------
def _zero_pred(labels):
    """Return dict  {label: 0.0  for label in labels}."""
    return {lab: 0.0 for lab in labels}

# ------------------------------------------------------------------
# Main prediction functions
# ------------------------------------------------------------------
def predict(group_df, lexicon_embeddings, num_categories, nlp, tokenizer, model, sigmoid, labels, id2label, device, args):
    """
    Auto-regressive inference for one Text-ID (group_df is already sorted).
    Returns list[dict] aligned with group_df rows.
    """
    
    # ---------- (A) optional extra feature vectors ----------

    ling_feat_vecs = None
    if args.linguistic_features:
        ling_feat_vecs = [compute_linguistic_features(txt, nlp) for txt in group_df["Text"]]

    ner_vecs = None
    if args.ner_features:
        ner_vecs = [compute_ner_embeddings(txt, nlp) for txt in group_df["Text"]]

    lex_vecs = None
    if args.lexicon:
        # 1) Is this lexicon in the dict of token-based scorers?
        if args.lexicon in LEXICON_COMPUTATION_FUNCTIONS:
            # Token-based approach
            lex_vecs = [
                compute_lexicon_scores(txt, args.lexicon, lexicon_embeddings, tokenizer, num_categories)
                for txt in group_df["Text"]
            ]
        else:
            # 2) This must be a precomputed (row-level) lexicon (e.g. LIWC-22 software generated)
            lex_vecs = [
                compute_precomputed_scores(row, lexicon_embeddings, num_categories)
                for _, row in group_df.iterrows()
            ]
        # If the above can produce NaNs, fix them
        lex_vecs = [
            [0.0 if (isinstance(x, float) and np.isnan(x)) else x for x in vec]
            for vec in lex_vecs
        ]
    
    topic_vecs = None
    if args.topic_detection:
        topic_vecs = group_df["topic_vectors"].tolist()

    # ---------- (B) Build prev-label tensor + concat text ----------
    if args.previous_sentences:
        prev_feats, text_instances = build_prev_label_and_text(
            group_df,
            labels=labels,
            model=model,
            tokenizer=tokenizer,
            device=device,
            lexicon_feats=lex_vecs,
            ling_feats=ling_feat_vecs,
            ner_feats=ner_vecs,
            topic_feats=topic_vecs
        )
    else:
        text_instances = group_df["Text"].tolist()

    # ---------- (C) single forward pass in batches ----------
    BATCH = 16
    results = []
    for i in range(0, len(group_df), BATCH):
        batch = text_instances[i:i + BATCH]

        # "text" contains all sentences (plain strings) of a single text in order (same Text-ID in the input file)
        encoded_sentences = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=512)

        encoded_sentences.pop("token_type_ids", None) # not used by DeBERTa

        # Move the encoded sentences to the GPU (if available)
        encoded_sentences = {key: value.to(device) for key, value in encoded_sentences.items()}

        # Compute features for each sentence separately

        if args.previous_sentences:
            encoded_sentences["prev_label_features"] = torch.tensor(prev_feats[i:i + BATCH], dtype=torch.float32, device=device)

        if args.linguistic_features:
            encoded_sentences['linguistic_features'] = torch.tensor(ling_feat_vecs[i : i+BATCH], dtype=torch.float32, device=device)

        if args.ner_features:
            encoded_sentences["ner_features"] = torch.tensor(ner_vecs[i : i+BATCH], dtype=torch.float32, device=device)

        if args.lexicon:
            encoded_sentences["lexicon_features"] = torch.tensor(lex_vecs[i : i+BATCH], dtype=torch.float32, device=device)
        
        if args.topic_detection:
            encoded_sentences["topic_features"] = torch.tensor(topic_vecs[i : i+BATCH], dtype=torch.float32, device=device)

        with torch.inference_mode():
            logits = model(**encoded_sentences)["logits"]
            probs  = sigmoid(logits).cpu().tolist() # move only tiny tensor to host

        results.extend({id2label[j]: p for j, p in enumerate(row)} for row in probs)

        # ---- free GPU buffers ASAP
        del encoded_sentences, logits, probs
        torch.cuda.empty_cache()
    
    return results

def label(instances, lexicon_embeddings, num_categories, nlp, tokenizer, model, sigmoid, labels, id2label, device, args):
    """ Predicts the label probabilities for each instance and adds them to it """
    
    df = pd.DataFrame(instances)              # ← build DF so helper can sort
    df = df.sort_values(["Text-ID", "Sentence-ID"]).reset_index(drop=True)

    preds = predict(df, lexicon_embeddings, num_categories, nlp, tokenizer, model, sigmoid, labels, id2label, device, args)

    return [{
            "Text-ID": instance["Text-ID"],
            "Sentence-ID": instance["Sentence-ID"],
            **labels
        } for instance, labels in zip(instances, preds)]

def writeRun(labeled_instances, output_dir, pred_type, args):
    """ Writes all (labeled) instances to the predictions.tsv in the output directory """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = ""
    if args.filter_1_model:
        filename += "1_" + args.filter_1_model + "_" + str(args.filter_1_th) + "_"
    filename += args.model_name + "-" + pred_type + ".tsv"
    output_file = os.path.join(output_dir, filename)
    pd.DataFrame.from_dict(labeled_instances).to_csv(output_file, header=True, index=False, sep='\t')

def run(model_group: str = "presence", filter_label: str = "Presence") -> None:
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

    if args.residualblock:
        args.multilayer = True

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

    # SETUP

    model_name = args.model_name
    model_path = "models/" + model_name # load from directory
    #model_path = "JohannesKiesel/valueeval24-bert-baseline-en" # load from huggingface hub

    # Load the configuration first
    config = transformers.AutoConfig.from_pretrained(model_path)

    file_labels = [v for k, v in sorted(config.id2label.items(),
                                        key=lambda kv: int(kv[0]))]
    if file_labels:                      # the checkpoint defines its own labels
        labels = file_labels
    else:                                # fall back to the static table
        labels = model_config["labels"]

    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)} 

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    # model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

    # code not executed by tira-run-inference-server (which directly calls 'predict(text)')
    if "TIRA_INFERENCE_SERVER" not in os.environ:

        if args.test_dataset:
            dataset_dir = args.test_dataset
            pred_type = "test"
        elif args.validation_dataset:
            dataset_dir = args.validation_dataset
            pred_type = "val"
        else:
            raise ValueError("No validation or test dataset specified.")
        output_dir = args.output_directory

        labeled_instances = []
        input_file = os.path.join(dataset_dir, "sentences.tsv")
        sent_df =  pd.read_csv(input_file, sep='\t', header=0, index_col=None)

        # ------------------------------------------------------------------
        # Attach Filter scores if the user asked for hierarchical
        # --------------------------------------------------------------------
        if args.filter_1_model:
            if args.test_dataset:
                filemode = "test"
            elif args.validation_dataset:
                filemode = "val"
            pres_df = pd.read_csv(args.filter_1_dir + args.filter_1_model + "-" + filemode + ".tsv", sep="\t")
            pres_df = pres_df.rename(columns={filter_label: "_" + filter_label + "_"})
            sent_df = sent_df.merge(pres_df,
                                    on=["Text-ID", "Sentence-ID"],
                                    how="left")
            sent_df["_" + filter_label + "_"] = sent_df["_" + filter_label + "_"].fillna(0.0)
            sent_df["keep"] = sent_df["_" + filter_label + "_"] >= args.filter_1_th
            logger.info(f"Hierarchical run ➜ "
                        f"{sent_df['keep'].sum()} / {len(sent_df)} sentences "
                        f"will be processed by the value model "
                        f"(threshold={args.filter_1_th})")
        else:
            sent_df["keep"] = True # flat mode

        if args.lexicon in ["LIWC-22", "eMFD", "MFD-20", "MJD"]:
            # We handle training and validation differently
            if args.test_dataset:
                lexicon_embeddings, num_categories = load_lexicon(args.lexicon, LEXICON_PATHS[args.lexicon+"-test"])
            elif args.validation_dataset:
                lexicon_embeddings, num_categories = load_lexicon(args.lexicon, LEXICON_PATHS[args.lexicon+"-validation"])
        else:
            # Fallback: single lexicon for everything (the old approach)
            lexicon_embeddings, num_categories = load_embeddings(args.lexicon)

        discovered_topics = 0
        if args.topic_detection:
            if args.topic_detection == "nmf":
                discovered_topics = model_config.get("nmf_topic_feature_dim", None)
            else:
                discovered_topics = None
            topic_model = TopicModeling(method=args.topic_detection, num_topics=discovered_topics)
            topic_vectors = topic_model.fit_transform(sent_df["Text"].tolist())
            sent_df["topic_vectors"] = list(topic_vectors)  # store as a column or array
            discovered_topics = topic_vectors.shape[1]
            logger.info(f"Using topic_feature_dim = {discovered_topics}")

        grouped_texts = sent_df.groupby("Text-ID")

        # Instantiate the custom model with the loaded config
        model = EnhancedDebertaModel(
            pretrained_model=model_config["pretrained_model"],
            config=config,
            num_labels=len(config.id2label),
            id2label=config.id2label,
            label2id=config.label2id,
            num_categories=num_categories,
            ling_feature_dim=17 if args.linguistic_features else 0,
            ner_feature_dim=768 if args.ner_features else 0,
            multilayer=args.multilayer,
            residualblock=args.residualblock,
            topic_feature_dim=discovered_topics,
            previous_sentences=args.previous_sentences
        )

        # Load model weights manually
        state_dict = load_file(model_path+"/model.safetensors")
        model.load_state_dict(state_dict)
        model.eval().to(device)
        sigmoid = torch.nn.Sigmoid()

        nlp = spacy.load("en_core_web_sm")

        with tqdm(total=len(grouped_texts), desc="Processing Texts") as pbar:
            for text_id, text_df in grouped_texts:
                # (a) rows that pass the Presence gate
                kept_rows     = text_df[text_df["keep"]]
                # (b) rows filtered out by the gate
                dropped_rows  = text_df[~text_df["keep"]]

                # ---- value model only sees the kept sentences ----
                if not kept_rows.empty:
                    labeled_instances.extend(
                        label(kept_rows.sort_values("Sentence-ID").to_dict("records"),
                              lexicon_embeddings, num_categories, nlp, tokenizer,
                              model, sigmoid, labels, id2label, device, args)
                    )

                # ---- for every dropped row, write zeros straight away ----
                for _, r in dropped_rows.iterrows():
                    labeled_instances.append({
                        "Text-ID":     r["Text-ID"],
                        "Sentence-ID": r["Sentence-ID"],
                        **_zero_pred(labels)
                    })

                pbar.update(1)  # Update progress bar

            writeRun(labeled_instances, output_dir, pred_type, args)