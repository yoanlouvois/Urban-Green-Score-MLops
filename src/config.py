# ========================
# DATA (RAW)
# ========================

DATASETS = {
    "urban": {
        "img_dir": "data/raw/train/Urban/images_png",
        "mask_dir": "data/raw/train/Urban/masks_png"
    },
    "rural": {
        "img_dir": "data/raw/train/Rural/images_png",
        "mask_dir": "data/raw/train/Rural/masks_png"
    }
}

SUBSET_SIZES = {
    "urban": 20,
    "rural": 5
}

SEED = 42

# ========================
# DATA (PROCESSED)
# ========================

TRAIN_IMAGES_DIR = "data/processed/train/images"
TRAIN_MASKS_DIR = "data/processed/train/masks"

# ========================
# MODEL OUTPUT
# ========================

MODEL_OUTPUT_DIR = "artifacts/models"
BEST_MODEL_PATH = "artifacts/models/best_model.pth"
LAST_MODEL_PATH = "artifacts/models/last_model.pth"

# ========================
# TRAINING
# ========================

NUM_CLASSES = 8
IGNORE_INDEX = 0

BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0