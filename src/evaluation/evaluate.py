import os
import sys
import json

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    TRAIN_IMAGES_DIR,
    TRAIN_MASKS_DIR,
    BEST_MODEL_PATH,
    NUM_CLASSES,
    IGNORE_INDEX,
    BATCH_SIZE,
    NUM_WORKERS,
)

from training.dataset import UrbanGreenDataset
from training.model import UNet
from evaluation.metrics import (
    pixel_accuracy,
    intersection_over_union,
    mean_iou,
)


OUTPUT_METRICS_PATH = "artifacts/evaluation/metrics.json"


def evaluate_model(model, dataloader, device):
    model.eval()

    total_pixel_accuracy = 0.0
    num_batches = 0

    iou_sums = {class_id: 0.0 for class_id in range(NUM_CLASSES) if class_id != IGNORE_INDEX}
    iou_counts = {class_id: 0 for class_id in range(NUM_CLASSES) if class_id != IGNORE_INDEX}

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            acc = pixel_accuracy(preds, masks, ignore_index=IGNORE_INDEX)
            ious = intersection_over_union(
                preds,
                masks,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
            )

            total_pixel_accuracy += acc
            num_batches += 1

            for class_id, iou in ious.items():
                if iou is not None:
                    iou_sums[class_id] += iou
                    iou_counts[class_id] += 1

    avg_pixel_accuracy = total_pixel_accuracy / num_batches if num_batches > 0 else 0.0

    avg_iou_per_class = {}

    for class_id in iou_sums:
        if iou_counts[class_id] > 0:
            avg_iou_per_class[class_id] = iou_sums[class_id] / iou_counts[class_id]
        else:
            avg_iou_per_class[class_id] = None

    avg_mean_iou = mean_iou(avg_iou_per_class)

    return {
        "pixel_accuracy": avg_pixel_accuracy,
        "mean_iou": avg_mean_iou,
        "iou_per_class": avg_iou_per_class,
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = UrbanGreenDataset(
        images_dir=TRAIN_IMAGES_DIR,
        masks_dir=TRAIN_MASKS_DIR,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    metrics = evaluate_model(model, val_loader, device)

    print("\nEvaluation metrics:")
    print(json.dumps(metrics, indent=4))

    with open(OUTPUT_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to: {OUTPUT_METRICS_PATH}")


if __name__ == "__main__":
    main()