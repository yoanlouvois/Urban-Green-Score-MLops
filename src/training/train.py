import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Allows imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    TRAIN_IMAGES_DIR,
    TRAIN_MASKS_DIR,
    MODEL_OUTPUT_DIR,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    NUM_CLASSES,
    IGNORE_INDEX,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    VAL_IMAGES_DIR,
    VAL_MASKS_DIR,
    SUBSET_SIZES,
)

from training.dataset import UrbanGreenDataset
from training.model import UNet


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
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_train_dataset = UrbanGreenDataset(
    images_dir=TRAIN_IMAGES_DIR,
    masks_dir=TRAIN_MASKS_DIR,
    )

    train_dataset = make_domain_subset(
    dataset=full_train_dataset,
    subset_sizes=SUBSET_SIZES["train"],
    split_name="train",
    )

    full_val_dataset = UrbanGreenDataset(
    images_dir=VAL_IMAGES_DIR,
    masks_dir=VAL_MASKS_DIR,
    )

    val_dataset = make_domain_subset(
        dataset=full_val_dataset,
        subset_sizes=SUBSET_SIZES["val"],
        split_name="val",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

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

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")

        torch.save(model.state_dict(), LAST_MODEL_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Best model saved.")

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model path: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()