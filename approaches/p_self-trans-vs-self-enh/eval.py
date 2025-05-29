from pathlib import Path
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = Path(__file__).resolve().parents[2]   # three levels up
sys.path.insert(0, str(project_root))                # put it at the front

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.parallel")

from core.evaluation import run

MODEL_GROUP = "self-trans_self-enh"

if __name__ == "__main__":
    run(model_group=MODEL_GROUP)