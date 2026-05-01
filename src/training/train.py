import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Allows imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    NUM_CLASSES,
    IGNORE_INDEX,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    SUBSET_SIZES,
)

from training.dataset import UrbanGreenDataset
from training.model import UNet

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed/train"),
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_VAL", "data/processed/val"),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "artifacts/models"),
    )

    parser.add_argument(
        "--use-subset",
        action="store_true",
        default=False,
        help="Use subset sizes from config.py",
    )

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)

    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

    return running_loss / len(dataloader)

def make_domain_subset(dataset, subset_sizes, split_name):
    selected_indices = []
    counts = {domain: 0 for domain in subset_sizes}

    for idx, img_name in enumerate(dataset.image_files):
        for domain, max_count in subset_sizes.items():
            if img_name.startswith(domain + "_") and counts[domain] < max_count:
                selected_indices.append(idx)
                counts[domain] += 1
                break

    print(f"{split_name.capitalize()} subset used:")
    for domain, count in counts.items():
        print(f"  {domain}: {count}/{subset_sizes[domain]}")

    return Subset(dataset, selected_indices)

def main():

    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Train dir: {args.train_dir}")
    print(f"Val dir:   {args.val_dir}")
    print(f"Model dir: {args.model_dir}")

    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Train directory not found: {args.train_dir}")

    if not os.path.exists(args.val_dir):
        raise FileNotFoundError(f"Val directory not found: {args.val_dir}")

    best_model_path = os.path.join(args.model_dir, "best_model.pth")
    last_model_path = os.path.join(args.model_dir, "last_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_train_dataset = UrbanGreenDataset(
        images_dir=os.path.join(args.train_dir, "images"),
        masks_dir=os.path.join(args.train_dir, "masks"),
    )

    if args.use_subset:
        train_dataset = make_domain_subset(
            dataset=full_train_dataset,
            subset_sizes=SUBSET_SIZES["train"],
            split_name="train",
        )
    else:
        train_dataset = full_train_dataset
        print("Train subset disabled: using all training images.")

    full_val_dataset = UrbanGreenDataset(
        images_dir=os.path.join(args.val_dir, "images"),
        masks_dir=os.path.join(args.val_dir, "masks"),
    )

    if args.use_subset:
        val_dataset = make_domain_subset(
            dataset=full_val_dataset,
            subset_sizes=SUBSET_SIZES["val"],
            split_name="val",
        )
    else:
        val_dataset = full_val_dataset
        print("Val subset disabled: using all validation images.")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty.")

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), last_model_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved.")

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model path: {best_model_path}")

    '''run : python src/training/train.py \
    --train-dir data/processed/train \
    --val-dir data/processed/val \
    --model-dir artifacts/models \
    --batch-size 2 \
    --epochs 5 \
    --learning-rate 1e-4 \
    --num-workers 0'''


if __name__ == "__main__":
    main()