import os
import random
import shutil
import sys

from PIL import Image
from tqdm import tqdm

# Allows imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS, DOMAINS, SEED
from utils import resize_image, resize_mask, random_rotation


random.seed(SEED)


def clean_processed_dir():
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)

    for split in SPLITS:
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, split, "masks"), exist_ok=True)


def collect_samples():
    samples = []

    for split in SPLITS:
        for domain in DOMAINS:
            img_dir = os.path.join(RAW_DATA_DIR, split, domain, "images_png")
            mask_dir = os.path.join(RAW_DATA_DIR, split, domain, "masks_png")

            if not os.path.exists(img_dir):
                print(f"Image folder not found: {img_dir}, skipping...")
                continue

            mask_exists = os.path.exists(mask_dir)

            if not mask_exists:
                print(f"No masks for {split}/{domain} (ok for test)")

            images = [f for f in os.listdir(img_dir) if f.endswith(".png")]

            print(f"{split}/{domain}: {len(images)} images found")

            for img_name in images:
                samples.append({
                    "split": split,
                    "domain": domain,
                    "img_name": img_name,
                    "img_path": os.path.join(img_dir, img_name),
                    "mask_path": os.path.join(mask_dir, img_name) if mask_exists else None,
                })

    random.shuffle(samples)
    return samples


def preprocess_sample(sample):
    split = sample["split"]
    domain = sample["domain"]
    img_name = sample["img_name"]

    img = Image.open(sample["img_path"]).convert("RGB")
    img = resize_image(img)

    mask = None
    if sample["mask_path"] is not None and os.path.exists(sample["mask_path"]):
        mask = Image.open(sample["mask_path"])
        mask = resize_mask(mask)

    # Data augmentation uniquement sur train ET si mask existe
    if split == "train" and mask is not None:
        img, mask = random_rotation(img, mask)

    output_name = f"{domain}_{img_name}"

    output_img_dir = os.path.join(PROCESSED_DATA_DIR, split, "images")
    output_mask_dir = os.path.join(PROCESSED_DATA_DIR, split, "masks")

    img.save(os.path.join(output_img_dir, output_name))

    if mask is not None:
        mask.save(os.path.join(output_mask_dir, output_name))


def main():
    clean_processed_dir()

    samples = collect_samples()

    print(f"\nTotal images to process: {len(samples)}")

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(preprocess_sample, samples), total=len(samples)))

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()