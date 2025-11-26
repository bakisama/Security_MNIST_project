from pathlib import Path

# Paths
BASE_DIR = Path.cwd().resolve()
OUT_DIR = BASE_DIR / "secure_ai_outputs"
IMG_DIR = OUT_DIR / "images"
MODEL_DIR = OUT_DIR / "models"
METRIC_DIR = OUT_DIR / "metrics"

# Training / attack hyperparams
USE_EXISTING_MODELS = False
BASELINE_EPOCHS = 5
ADV_TRAIN_EPOCHS = 5
POISON_EPOCHS = 5
BATCH_SIZE = 128

FGSM_EPS = 0.25

# Poisoning
PATCH_SIZE = 4
PATCH_VALUE = 1.0
TARGET_CLASS = 7  # digit “7”
