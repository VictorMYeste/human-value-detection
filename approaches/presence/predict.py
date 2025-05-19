import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.parallel")

from core.prediction import run

MODEL_GROUP = "presence"

if __name__ == "__main__":
    run(model_group=MODEL_GROUP)