import torch
from src.config import IGNORE_INDEX
from src.scoring.green_score import compute_green_score


def test_green_score_full_forest():
    mask = torch.full((10, 10), 6)

    score, proportions = compute_green_score(mask, ignore_index=IGNORE_INDEX)

    assert score > 0
    assert proportions["forest"] == 1.0