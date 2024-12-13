import os
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HVD")

def validate_args(labels, training_dataset, validation_dataset):
    assert len(labels) > 0, "Labels cannot be empty."
    assert training_dataset is not None, "Training dataset cannot be None."
    assert validation_dataset is not None, "Validation dataset cannot be None."
    logger.info("Arguments validated successfully.")

def slice_for_testing(dataset, size=100):
    return dataset.select(range(size))

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