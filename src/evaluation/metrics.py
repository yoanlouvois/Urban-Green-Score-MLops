

def pixel_accuracy(preds, targets, ignore_index=0):
    """
    Computes pixel accuracy.

    preds:   [B, H, W] predicted class labels
    targets: [B, H, W] ground-truth class labels
    """

    valid_mask = targets != ignore_index

    correct = (preds == targets) & valid_mask
    total = valid_mask.sum()

    if total == 0:
        return 0.0

    return (correct.sum().float() / total.float()).item()


def intersection_over_union(preds, targets, num_classes=8, ignore_index=0):
    """
    Computes IoU for each class.

    preds:   [B, H, W]
    targets: [B, H, W]
    """

    iou_per_class = {}

    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_mask = preds == class_id
        target_mask = targets == class_id

        valid_mask = targets != ignore_index

        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            iou_per_class[class_id] = None
        else:
            iou_per_class[class_id] = (intersection / union).item()

    return iou_per_class


def mean_iou(iou_per_class):
    """
    Computes mean IoU, ignoring classes that are not present.
    """

    valid_ious = [
        iou for iou in iou_per_class.values()
        if iou is not None
    ]

    if len(valid_ious) == 0:
        return 0.0

    return sum(valid_ious) / len(valid_ious)