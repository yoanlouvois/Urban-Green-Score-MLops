import os
import sys
import io
import base64

import requests
import streamlit as st
import numpy as np
import torch

from PIL import Image
from matplotlib.colors import ListedColormap

# Allow imports from project root/src
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.append(ROOT_DIR)
sys.path.append(SRC_DIR)

from config import NUM_CLASSES, IGNORE_INDEX  # noqa: E402
from preprocessing.utils import resize_image  # noqa: E402
from training.model import UNet  # noqa: E402
from scoring.green_score import compute_green_score  # noqa: E402


DEFAULT_API_URL = os.getenv(
    "API_GATEWAY_URL",
    "https://5jhgyaenq3.execute-api.eu-west-3.amazonaws.com/predict",
)

DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "artifacts", "models", "best_model.pth")


st.set_page_config(
    page_title="Urban Green Score",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
            max-width: 1300px;
        }

        h1 {
            font-size: 2rem !important;
            margin-bottom: 0.2rem !important;
        }

        h2, h3 {
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
        }

        [data-testid="stImage"] img {
            max-height: 430px;
            object-fit: contain;
        }

        [data-testid="stSidebar"] {
            min-width: 260px;
            max-width: 260px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 Urban Green Score")
st.write("Analyse une image satellite et estime un score de végétation urbaine.")


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


def mask_to_colored_image(mask_np):
    cmap = get_mask_colormap()
    colored = cmap(mask_np, bytes=True)[:, :, :3]
    return Image.fromarray(colored)


def load_image_for_local_inference(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = resize_image(image)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)

    return image, image_tensor


@st.cache_resource
def load_local_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device


def predict_local(uploaded_file, model_path):
    image, image_tensor = load_image_for_local_inference(uploaded_file)

    model, device = load_local_model(model_path)

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()

    green_score, proportions = compute_green_score(
        pred_mask,
        ignore_index=IGNORE_INDEX,
    )

    mask_np = pred_mask.numpy().astype(np.uint8)
    mask_img = mask_to_colored_image(mask_np)

    return {
        "image": image,
        "mask_img": mask_img,
        "green_score": green_score,
        "class_proportions": proportions,
        "mask_shape": list(mask_np.shape),
    }


def predict_api(uploaded_file, api_url):
    image_bytes = uploaded_file.getvalue()

    response = requests.post(
        api_url,
        headers={"Content-Type": "image/png"},
        data=image_bytes,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    result = response.json()

    mask_bytes = base64.b64decode(result["mask_png_base64"])
    raw_mask = Image.open(io.BytesIO(mask_bytes))
    mask_np = np.array(raw_mask).astype(np.uint8)
    mask_img = mask_to_colored_image(mask_np)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(mask_img.size)

    return {
        "image": image,
        "mask_img": mask_img,
        "green_score": result["green_score"],
        "class_proportions": result["class_proportions"],
        "mask_shape": result["mask_shape"],
    }


with st.sidebar:
    st.header("Configuration")

    mode = st.radio(
        "Mode d'inférence",
        ["Cloud API Gateway", "Local model"],
    )

    if mode == "Cloud API Gateway":
        api_url = st.text_input("API Gateway URL", value=DEFAULT_API_URL)
        st.warning(
            "Le SageMaker Endpoint doit être actif/InService avant d’utiliser ce mode."
        )
    else:
        model_path = st.text_input("Local model path", value=DEFAULT_MODEL_PATH)
        st.info("Ce mode utilise directement le modèle local sur ta machine.")

uploaded_file = st.file_uploader(
    "Choisir une image satellite",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    top_col1, top_col2, top_col3 = st.columns([2, 1, 2])

    with top_col1:
        run = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    result_placeholder = top_col2.empty()
    status_placeholder = top_col3.empty()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image source")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.subheader("Masque de segmentation")
        mask_placeholder = st.empty()

    if run:
        try:
            with status_placeholder:
                with st.spinner("Analyse en cours..."):
                    if mode == "Cloud API Gateway":
                        result = predict_api(uploaded_file, api_url)
                    else:
                        result = predict_local(uploaded_file, model_path)

            result_placeholder.metric(
                label="Green Score",
                value=f"{result['green_score']:.2f}/100",
            )

            mask_placeholder.image(
                result["mask_img"],
                use_container_width=True,
            )

            st.write("### Proportions des classes")
            st.bar_chart(result["class_proportions"])

            with st.expander("Détails JSON"):
                st.json(
                    {
                        "green_score": result["green_score"],
                        "class_proportions": result["class_proportions"],
                        "mask_shape": result["mask_shape"],
                    }
                )

        except Exception as e:
            st.error(str(e))