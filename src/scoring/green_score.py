import numpy as np
import torch


CLASS_LABELS = {
    "background": 1,
    "building": 2,
    "road": 3,
    "water": 4,
    "barren": 5,
    "forest": 6,
    "agriculture": 7,
}

GREEN_SCORE_WEIGHTS = {
    "forest": 1.0,
    "agriculture": 0.7,
    "water": 0.5,
    "barren": 0.2,
    "background": 0.1,
}


def compute_class_proportions(mask, ignore_index=0):
    """
    Computes class proportions from a segmentation mask.

    mask can be:
    - numpy array [H, W]
    - torch tensor [H, W]
    """

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    valid_pixels = mask != ignore_index
    total_pixels = np.sum(valid_pixels)

    if total_pixels == 0:
        return {class_name: 0.0 for class_name in CLASS_LABELS}

    proportions = {}

    for class_name, label in CLASS_LABELS.items():
        class_pixels = np.sum(mask == label)
        proportions[class_name] = class_pixels / total_pixels

    return proportions


def compute_green_score(mask, ignore_index=0):
    """
    Computes the Urban Green Score from a segmentation mask.
    """

    proportions = compute_class_proportions(mask, ignore_index=ignore_index)

    green_score = 100 * sum(
        GREEN_SCORE_WEIGHTS[class_name] * proportions[class_name]
        for class_name in GREEN_SCORE_WEIGHTS
    )

    green_score = max(0.0, min(100.0, green_score))

    return green_score, proportions


def compare_green_scores(true_mask, pred_mask, ignore_index=0):
    """
    Computes Green Score for true and predicted masks.
    """

    true_score, true_proportions = compute_green_score(
        true_mask,
        ignore_index=ignore_index,
    )

    pred_score, pred_proportions = compute_green_score(
        pred_mask,
        ignore_index=ignore_index,
    )

    return {
        "green_score_true": true_score,
        "green_score_pred": pred_score,
        "green_score_error": abs(true_score - pred_score),
        "true_proportions": true_proportions,
        "pred_proportions": pred_proportions,
    }