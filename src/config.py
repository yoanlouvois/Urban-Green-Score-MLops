# ========================
# DATA (RAW)
# ========================

RAW_DATA_DIR = "data/raw"
SPLITS = ["train", "val", "test"]
DOMAINS = ["Urban", "Rural"]

SEED = 42

# Optionnel : à utiliser plus tard côté train, pas dans preprocess
SUBSET_SIZES = {
    "Urban": 20,
    "Rural": 5,
}

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

NUM_CLASSES = 8
IGNORE_INDEX = 0

BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0