import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from transformers import pipeline, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

import pandas as pd
from datasets import Dataset
import torch
#from core.utils import slice_for_testing

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HVD")

SENTENCES_PATH = "../data/training-english/sentences.tsv"
LABELS_PATH = "../data/training-english/labels-cat.tsv"

AUG_SENTENCES_PATH = "../data/training-english/sentences-aug-2.tsv"
AUG_LABELS_PATH = "../data/training-english/labels-cat-aug-2.tsv"

AUGMENTATION_CONFIG = {
    "use_paraphrasing": True,
    "paraphrasing_models": [
        #"ramsrigouthamg/t5_paraphraser",
        "google/flan-t5-large",
        "humarin/chatgpt_paraphraser_on_T5_base"
    ],
    "num_augmented_variations": 1,  # One paraphrase per model
    "num_beams": 1,
    "batch_size": 8,
    "device": 0 if torch.cuda.is_available() else -1,
    "max_length": 512,
    "temperature": 1.5,
    "top_k": 30,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "similarity_threshold": 0.85
}

# Load a strong sentence embedding model
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def filter_paraphrases(original, paraphrases, threshold=0.85):
    """
    Removes paraphrases that are too similar to the original sentence.
    """
    original_embedding = embedder.encode(original, convert_to_tensor=True)
    filtered_paraphrases = []
    
    for para in paraphrases:
        para_embedding = embedder.encode(para, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(original_embedding, para_embedding).item()
        if similarity < threshold:
            filtered_paraphrases.append(para)

    return filtered_paraphrases or paraphrases[0]  # Ensure at least one paraphrase is kept

def back_translate(sentence, lang="es"):
    """
    Translates the sentence to another language and back to English.
    """
    translated = GoogleTranslator(source="en", target=lang).translate(sentence)
    return GoogleTranslator(source=lang, target="en").translate(translated)

def generate_augmented_dataset():
    """
    Generates paraphrased sentences using multiple models and assigns original labels.
    Saves new files: sentences-aug.tsv and labels-cat-aug.tsv.
    """
    logger.info("Loading original dataset...")

    # Load sentences and labels
    df_sentences = pd.read_csv(SENTENCES_PATH, sep="\t", encoding="utf-8")
    df_labels = pd.read_csv(LABELS_PATH, sep="\t", encoding="utf-8")

    #df_sentences = slice_for_testing(df_sentences, 10)
    #df_labels = slice_for_testing(df_labels, 10)

    texts = df_sentences["Text"].fillna("").tolist()
    dataset = Dataset.from_dict({"Text": texts})

    logger.info(f"Starting paraphrasing of {len(texts)} sentences using batch size {AUGMENTATION_CONFIG['batch_size']}...")

    # Load multiple paraphrasing models
    paraphrasers = [
        pipeline(
            "text2text-generation",
            model=model,
            tokenizer=T5Tokenizer.from_pretrained(model, legacy=False),
            device=AUGMENTATION_CONFIG["device"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        ) for model in AUGMENTATION_CONFIG["paraphrasing_models"]
    ]

    def paraphrase_batch(batch):
        """
        Paraphrases a batch of texts using multiple models while ensuring proper structure.
        """
        paraphrased_sentences = [[] for _ in batch["Text"]]  # Empty lists for each sentence

        for model in paraphrasers:
            outputs = model(
                batch["Text"],
                max_length=AUGMENTATION_CONFIG["max_length"],
                num_return_sequences=1,  # Get one variation per model
                num_beams=AUGMENTATION_CONFIG["num_beams"],
                do_sample=True,
                temperature=AUGMENTATION_CONFIG["temperature"],
                top_k=AUGMENTATION_CONFIG["top_k"],
                top_p=AUGMENTATION_CONFIG["top_p"],
                truncation=True
            )

            for idx, output in enumerate(outputs):
                # Ensure output is a list and extract the first element
                if isinstance(output, list) and len(output) > 0:
                    paraphrased_sentences[idx].append(output[0]["generated_text"])
                elif isinstance(output, dict) and "generated_text" in output:
                    paraphrased_sentences[idx].append(output["generated_text"])
                else:
                    logger.warning(f"Unexpected output format at index {idx}: {output}")
                    paraphrased_sentences[idx].append(batch["Text"][idx])  # Fallback to original text if output fails

        # Apply filtering to remove redundant paraphrases
        for idx, paraphrase_list in enumerate(paraphrased_sentences):
            paraphrased_sentences[idx] = filter_paraphrases(batch["Text"][idx], paraphrase_list, threshold=AUGMENTATION_CONFIG["similarity_threshold"])
        
        return {"Paraphrased_Text": paraphrased_sentences}

    # Apply paraphrasing with batch processing
    paraphrased_dataset = dataset.map(
        paraphrase_batch,
        batched=True,
        batch_size=AUGMENTATION_CONFIG["batch_size"]
    )

    # Create new dataset with augmented data
    augmented_sentences = []
    augmented_labels = []

    for i, row in df_sentences.iterrows():
        text_id, sentence_id, text = row["Text-ID"], row["Sentence-ID"], row["Text"]

        # Append original sentence
        augmented_sentences.append([text_id, sentence_id, text])
        augmented_labels.append([text_id, sentence_id] + df_labels.iloc[i, 2:].tolist())

        # Append filtered paraphrased versions from both models
        paraphrased_versions = paraphrased_dataset["Paraphrased_Text"][i]

        for j, paraphrased_text in enumerate(paraphrased_versions):
            new_sentence_id = f"{sentence_id}_aug{j+1}"  # Ensure unique ID
            augmented_sentences.append([text_id, new_sentence_id, paraphrased_text])
            augmented_labels.append([text_id, new_sentence_id] + df_labels.iloc[i, 2:].tolist())

    # Convert to DataFrames
    df_aug_sentences = pd.DataFrame(augmented_sentences, columns=df_sentences.columns)
    df_aug_labels = pd.DataFrame(augmented_labels, columns=df_labels.columns)

    # Save new augmented files
    df_aug_sentences.to_csv(AUG_SENTENCES_PATH, sep="\t", index=False, encoding="utf-8")
    df_aug_labels.to_csv(AUG_LABELS_PATH, sep="\t", index=False, encoding="utf-8")

    logger.info(f"Successfully saved {len(df_aug_sentences)} augmented sentences to {AUG_SENTENCES_PATH}.")
    logger.info(f"Successfully saved {len(df_aug_labels)} labels to {AUG_LABELS_PATH}.")

# Run the function
generate_augmented_dataset()