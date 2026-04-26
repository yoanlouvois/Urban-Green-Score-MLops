import os
import random
import shutil
import sys
import argparse

from PIL import Image
from tqdm import tqdm

# Allows imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SPLITS, DOMAINS
from utils import resize_image, resize_mask, random_rotation

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def clean_processed_dir(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in SPLITS:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)


def collect_samples(raw_data_dir):
    samples = []

    for split in SPLITS:
        for domain in DOMAINS:
            img_dir = os.path.join(raw_data_dir, split, domain, "images_png")
            mask_dir = os.path.join(raw_data_dir, split, domain, "masks_png")

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


def preprocess_sample(sample, output_dir):
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

    output_img_dir = os.path.join(output_dir, split, "images")
    output_mask_dir = os.path.join(output_dir, split, "masks")

    img.save(os.path.join(output_img_dir, output_name))

    if mask is not None:
        mask.save(os.path.join(output_mask_dir, output_name))


def main():
    args = parse_args()

    random.seed(args.seed)

    clean_processed_dir(args.output_dir)

    samples = collect_samples(args.raw_data_dir)

    print(f"\nTotal images to process: {len(samples)}")

    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    worker_fn = partial(preprocess_sample, output_dir=args.output_dir)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(worker_fn, samples), total=len(samples)))

    print("\nPreprocessing completed.")
    #to run : python src/preprocessing/preprocess.py --raw-data-dir data/raw --output-dir data/processed --num-workers 4


if __name__ == "__main__":
    main()