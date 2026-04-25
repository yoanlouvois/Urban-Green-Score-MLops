import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class UrbanGreenDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.image_files = [
            f for f in os.listdir(images_dir)
            if f.endswith(".png")
        ]

        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.int64)

        # image: H, W, C -> C, H, W
        image = torch.from_numpy(image).permute(2, 0, 1)

        # mask: H, W
        mask = torch.from_numpy(mask)

        return image, mask
    
if __name__ == "__main__":
    dataset = UrbanGreenDataset(
        images_dir="data/processed/train/images",
        masks_dir="data/processed/train/masks"
    )

    image, mask = dataset[0]

    print("Dataset size:", len(dataset))
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Mask labels:", torch.unique(mask))

    all_labels = set()

    for i in range(len(dataset)):
        _, mask = dataset[i]
        labels = torch.unique(mask).tolist()
        all_labels.update(labels)

    print("All labels in dataset:", sorted(all_labels))