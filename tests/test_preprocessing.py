import numpy as np
from PIL import Image
from src.preprocessing.utils import resize_mask


def test_resize_mask_preserves_class_labels():
    mask = Image.fromarray(
        np.array([[0, 1], [6, 7]], dtype=np.uint8)
    )

    resized = resize_mask(mask)
    values = set(np.array(resized).flatten().tolist())

    assert values.issubset({0, 1, 6, 7})