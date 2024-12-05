import argparse

def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(prog="Growth-SelfProtection")
    cli.add_argument("-t", "--training-dataset", required=True, help="Path to training dataset")
    cli.add_argument("-v", "--validation-dataset", default=None, help="Path to validation dataset")
    cli.add_argument("-p", "--previous-sentences", action='store_true', help="If to add the two previous sentences of every sentence to the model")
    cli.add_argument("-f", "--linguistic-features", action='store_true', help="If to add linguistic features to the model")
    cli.add_argument("-l", "--lexicon", default=None, help="Lexicon to be added on top of the model")
    cli.add_argument("-m", "--model-name", help="Name of the model if being uploaded to HuggingFace")
    cli.add_argument("-o", "--model-directory", default="models", help="Directory to save the trained model")
    cli.add_argument("-s", "--slice", action='store_true', help="Slice for testing with size = 100")
    return cli.parse_args()