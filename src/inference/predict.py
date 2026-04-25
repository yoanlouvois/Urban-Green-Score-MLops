import os
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.utils import resize_image

from config import (
    BEST_MODEL_PATH,
    NUM_CLASSES,
    IGNORE_INDEX,
)

from training.model import UNet
from scoring.green_score import compute_green_score


IMAGE_PATH = "data/raw/train/Urban/images_png/1410.png"
OUTPUT_DIR = "artifacts/predictions"


def get_mask_colormap():
    colors = [
        (0, 0, 0),        # 0 no-data
        (200, 200, 200),  # 1 background
        (128, 128, 128),  # 2 building
        (255, 255, 0),    # 3 road
        (0, 0, 255),      # 4 water
        (139, 69, 19),    # 5 barren
        (0, 255, 0),      # 6 forest
        (0, 128, 0),      # 7 agriculture
    ]

    return ListedColormap(np.array(colors) / 255.0)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # IMPORTANT : même preprocessing que training
    image = resize_image(image)

    image_np = np.array(image).astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)

    return image, image_tensor


def predict_mask(model, image_tensor, device):
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1)

    return pred_mask.squeeze(0).cpu()


def save_prediction_visualization(image, pred_mask, green_score, output_path):
    cmap = get_mask_colormap()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(image)
    axs[0].set_title("Satellite image")
    axs[0].axis("off")

    axs[1].imshow(pred_mask, cmap=cmap, vmin=0, vmax=7)
    axs[1].set_title(f"Predicted mask | GS: {green_score:.2f}/100")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image, image_tensor = load_image(IMAGE_PATH)

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    pred_mask = predict_mask(model, image_tensor, device)

    green_score, proportions = compute_green_score(
        pred_mask,
        ignore_index=IGNORE_INDEX,
    )

    print("\nPrediction result:")
    print(f"Image: {IMAGE_PATH}")
    print(f"Green Score: {green_score:.2f}/100")
    print("\nClass proportions:")
    print(json.dumps(proportions, indent=4))

    output_img_path = os.path.join(OUTPUT_DIR, "prediction.png")
    save_prediction_visualization(
        image=image,
        pred_mask=pred_mask,
        green_score=green_score,
        output_path=output_img_path,
    )

    output_json_path = os.path.join(OUTPUT_DIR, "prediction.json")

    with open(output_json_path, "w") as f:
        json.dump(
            {
                "image_path": IMAGE_PATH,
                "green_score": green_score,
                "class_proportions": proportions,
            },
            f,
            indent=4,
        )

    print(f"\nPrediction image saved to: {output_img_path}")
    print(f"Prediction JSON saved to: {output_json_path}")


if __name__ == "__main__":
    main()