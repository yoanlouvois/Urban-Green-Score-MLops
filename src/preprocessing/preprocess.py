import os
import random
from PIL import Image
from tqdm import tqdm
import shutil
import sys

from utils import resize_image, resize_mask, random_rotation

# Allows imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATASETS, SUBSET_SIZES, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, SEED


# clean output folders
if os.path.exists(TRAIN_IMAGES_DIR):
    shutil.rmtree(TRAIN_IMAGES_DIR)

if os.path.exists(TRAIN_MASKS_DIR):
    shutil.rmtree(TRAIN_MASKS_DIR)

os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_MASKS_DIR, exist_ok=True)

random.seed(SEED)

samples = []

for domain, paths in DATASETS.items():
    images = [f for f in os.listdir(paths["img_dir"]) if f.endswith(".png")]

    subset_size = SUBSET_SIZES.get(domain, len(images))  # fallback si non défini
    selected = random.sample(images, min(subset_size, len(images)))

    print(f"{domain}: {len(selected)} images selected")

    for img_name in selected:
        samples.append({
            "domain": domain,
            "img_name": img_name,
            "img_path": os.path.join(paths["img_dir"], img_name),
            "mask_path": os.path.join(paths["mask_dir"], img_name)
        })

random.shuffle(samples)

print(f"\nTotal images to process: {len(samples)}")

for sample in tqdm(samples):
    domain = sample["domain"]
    img_name = sample["img_name"]

    if not os.path.exists(sample["mask_path"]):
        print(f"Mask not found for {domain}/{img_name}, skipping...")
        continue

    img = Image.open(sample["img_path"]).convert("RGB")
    mask = Image.open(sample["mask_path"])

    img = resize_image(img)
    mask = resize_mask(mask)

    img, mask = random_rotation(img, mask)

    output_name = f"{domain}_{img_name}"

    img.save(os.path.join(TRAIN_IMAGES_DIR, output_name))  
    mask.save(os.path.join(TRAIN_MASKS_DIR, output_name))