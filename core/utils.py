import os
import shutil

import pandas as pd
import nltk

from core.log import logger

def validate_args(labels, training_dataset, validation_dataset):
    assert len(labels) > 0, "Labels cannot be empty."
    assert training_dataset is not None, "Training dataset cannot be None."
    assert validation_dataset is not None, "Validation dataset cannot be None."
    logger.info("Arguments validated successfully.")

def slice_for_testing(dataset, size=1000):
    if hasattr(dataset, 'select'):  # Assuming it's a dataset with a select method
        return dataset.select(range(size))
    elif isinstance(dataset, pd.DataFrame):  # Check if it's a pandas DataFrame
        return dataset.iloc[:size]  # Use iloc to slice DataFrame rows
    else:
        raise TypeError("Unsupported dataset type. Expected Dataset or DataFrame.")

def validate_file(path):
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    if not os.path.isfile(path):
        logger.error(f"Path is not a file: {path}")
        raise ValueError(f"Path is not a valid file: {path}")

def normalize_token(token):
    return token.lower().lstrip("ġ")

def skip_invalid_line(line, reason):
    logger.warning(f"Skipping line due to {reason}: {line}")

def read_file_lines(path, skip_header=False, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()
    return lines[1:] if skip_header else lines

def download_nltk_resources():
    """Ensure necessary NLTK resources are available."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def clear_directory(directory):
    """Remove all files and subdirectories in the given directory."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")