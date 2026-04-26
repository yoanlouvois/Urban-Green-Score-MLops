# ========================
# DATA (RAW)
# ========================

RAW_DATA_DIR = "data/raw"
SPLITS = ["train", "val", "test"]
DOMAINS = ["Urban", "Rural"]

# ========================
# DATA (PROCESSED)
# ========================

PROCESSED_DATA_DIR = "data/processed"

# ========================
# MODEL OUTPUT
# ========================

MODEL_OUTPUT_DIR = "artifacts/models"
BEST_MODEL_PATH = "artifacts/models/best_model.pth"
LAST_MODEL_PATH = "artifacts/models/last_model.pth"

# ========================
# TRAINING
# ========================

SUBSET_SIZES = {
    "train": {
        "Urban": 20,
        "Rural": 5,
    },
    "val": {
        "Urban": 4,
        "Rural": 1,
    },
    "eval": {
        "Urban": 4,
        "Rural": 1,
    },
}

TRAIN_IMAGES_DIR = "data/processed/train/images"
TRAIN_MASKS_DIR = "data/processed/train/masks"

VAL_IMAGES_DIR = "data/processed/val/images"
VAL_MASKS_DIR = "data/processed/val/masks"

NUM_CLASSES = 8
IGNORE_INDEX = 0

BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0