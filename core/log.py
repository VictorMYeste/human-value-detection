import sys
import logging
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# Remove any existing log handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Force stdout to be unbuffered
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Custom handler that ensures immediate flushing
class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # Ensure logs are flushed immediately
    
# Configure logging to always use stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[FlushStreamHandler(sys.stdout)]  # Force logs to stdout
)

logger = logging.getLogger("HVD")

# Suppress duplicate logs on multi-GPU runs (only rank 0 logs)
if not accelerator.is_main_process:
    logger.setLevel(logging.WARNING)  # Reduce logging for non-primary ranks