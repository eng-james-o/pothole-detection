import os
from pathlib import Path

APP_NAME = "Pothole Detection API"
APP_VERSION = "1.0.0"

MODEL_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", "deployment/model_config.yaml"))
INFERENCE_CONF = float(os.getenv("INFERENCE_CONF", "0.25"))
INFERENCE_IOU = float(os.getenv("INFERENCE_IOU", "0.45"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "8000000"))
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "").strip()
