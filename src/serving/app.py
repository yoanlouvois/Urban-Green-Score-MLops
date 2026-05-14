import os
import sys
import io
import base64

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import NUM_CLASSES, IGNORE_INDEX
from preprocessing.utils import resize_image
from training.model import UNet
from scoring.green_score import compute_green_score


MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

app = FastAPI(title="Urban Green Score Inference API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    loaded_model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    loaded_model.to(device)
    loaded_model.eval()

    model = loaded_model
    print(f"Model loaded from {MODEL_PATH} on {device}", flush=True)


@app.on_event("startup")
def startup_event():
    load_model()


def load_image_from_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = resize_image(image)

    image_np = np.array(image).astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)

    return image, image_tensor


def predict_mask(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1)

    return pred_mask.squeeze(0).cpu()


def mask_to_base64_png(mask_tensor):
    mask_np = mask_tensor.numpy().astype(np.uint8)

    mask_image = Image.fromarray(mask_np, mode="L")

    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/ping")
def ping():
    if model is None:
        return Response(status_code=500)

    return Response(content="OK", status_code=200)


@app.post("/invocations")
async def invocations(request: Request):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model is not loaded"},
        )

    content_type = request.headers.get("content-type", "")

    try:
        body = await request.body()

        if "image/png" in content_type or "image/jpeg" in content_type or "application/octet-stream" in content_type:
            image, image_tensor = load_image_from_bytes(body)
        else:
            return JSONResponse(
                status_code=415,
                content={
                    "error": f"Unsupported content type: {content_type}. Use image/png or image/jpeg."
                },
            )

        pred_mask = predict_mask(image_tensor)

        green_score, proportions = compute_green_score(
            pred_mask,
            ignore_index=IGNORE_INDEX,
        )

        mask_png_base64 = mask_to_base64_png(pred_mask)

        response = {
            "green_score": green_score,
            "class_proportions": proportions,
            "mask_shape": list(pred_mask.shape),
            "mask_png_base64": mask_png_base64,
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )