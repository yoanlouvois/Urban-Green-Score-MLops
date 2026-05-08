import os
import json
import base64
import io

import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv(
    "API_GATEWAY_URL",
    "https://5jhgyaenq3.execute-api.eu-west-3.amazonaws.com/predict"
)

IMAGE_PATH = "data/raw/test/Rural/images_png/4196.png"
OUTPUT_DIR = "artifacts/predictions"

OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "api_response.json")
OUTPUT_MASK_RAW_PATH = os.path.join(OUTPUT_DIR, "api_mask_raw.png")
OUTPUT_VISUALIZATION_PATH = os.path.join(OUTPUT_DIR, "api_prediction.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def save_prediction_visualization(image_path, pred_mask, green_score, output_path):
    image = Image.open(image_path).convert("RGB")

    mask_h, mask_w = pred_mask.shape
    image = image.resize((mask_w, mask_h))

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
    plt.close()


with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

response = requests.post(
    API_URL,
    headers={"Content-Type": "image/png"},
    data=image_bytes,
    timeout=60,
)

print("Status code:", response.status_code)

if response.status_code != 200:
    print(response.text)
    raise RuntimeError("API request failed.")

result = response.json()

green_score = result.get("green_score")
mask_base64 = result.get("mask_png_base64")

print("Green score:", green_score)
print("Mask shape:", result.get("mask_shape"))
print("Class proportions:")
print(json.dumps(result.get("class_proportions"), indent=4))

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(result, f, indent=4)

if mask_base64:
    mask_bytes = base64.b64decode(mask_base64)

    with open(OUTPUT_MASK_RAW_PATH, "wb") as f:
        f.write(mask_bytes)

    mask_img = Image.open(io.BytesIO(mask_bytes))
    pred_mask = np.array(mask_img).astype(np.uint8)

    save_prediction_visualization(
        image_path=IMAGE_PATH,
        pred_mask=pred_mask,
        green_score=float(green_score),
        output_path=OUTPUT_VISUALIZATION_PATH,
    )

    print(f"\nRaw mask saved to: {OUTPUT_MASK_RAW_PATH}")
    print(f"Visualization saved to: {OUTPUT_VISUALIZATION_PATH}")
else:
    print("\nNo mask_png_base64 found in API response.")