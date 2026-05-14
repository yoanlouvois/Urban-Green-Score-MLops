import torch
from src.evaluation.metrics import pixel_accuracy


def test_pixel_accuracy_perfect_prediction():
    preds = torch.tensor([[1, 2], [3, 4]])
    masks = torch.tensor([[1, 2], [3, 4]])

    acc = pixel_accuracy(preds, masks, ignore_index=0)

    assert acc == 1.0