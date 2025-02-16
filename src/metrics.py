from pathlib import Path
from utils.interpolate.markup_utils import (
    load_markup, 
    is_border_object, 
    Mask
)
from utils.integrate.metrics import compute_mask_metrics


def compute_precision_recall(
    gt_paths: list[Path], 
    pred_paths: list[Path], 
    image_shape: tuple[int, int],
    iou_threshold: float = 0.7,
    gt_mask_filter: callable = None,
    pred_mask_filter: callable = None
) -> dict:
    """Compute precision, recall.

    This function calculates the precision and recall
    across multiple pairs of ground truth and predicted segmentation masks.

    Args:
        gt_paths: A list of paths to ground truth markup files.
        pred_paths: A list of paths to predicted markup files.
        image_size: Size of the image (assumes square images).
        iou_threshold: The IoU threshold to use for calculating metrics.
        gt_mask_filter: An optional function to filter ground truth masks.
        pred_mask_filter: An optional function to filter predicted masks.

    Returns:
        A dictionary containing the precision and recall.
    """
    assert len(gt_paths) == len(pred_paths), "Ground truth and predicted paths lists must have the same length."

    total_matched = 0.0
    gt_count = 0
    pred_count = 0

    for gt_path, pred_path in zip(gt_paths, pred_paths):
        gt_data = load_markup(gt_path, image_shape)
        pred_data = load_markup(pred_path, image_shape)

        # Filter masks if filters are provided
        if gt_mask_filter:
            gt_data = gt_mask_filter(gt_data)
        if pred_mask_filter:
            pred_data = pred_mask_filter(pred_data)

        # Add metrics to sum
        metrics = compute_mask_metrics(gt_data, pred_data, iou_threshold)
        total_matched += metrics['conf_matrix'][0, 0]

        # Add counts to sum
        gt_count += len(gt_data)
        pred_count += len(pred_data)

    # Calculate precision and recall
    precision = total_matched / pred_count
    recall = total_matched / gt_count
    return {
        "precision": precision,
        "recall": recall,
        "conf_matrix": [
            [total_matched, pred_count - total_matched],
            [gt_count - total_matched, 0]
        ]
    }
    

def find_border_masks(
    masks: list[dict], 
    image_shape: tuple[int, int], 
    border_tolerance: float = 0.02
) -> list[dict]:
    """Find masks that are too close to the borders of the image.

    Args:
        masks: A list of masks to filter.
        image_shape: The shape of the image (height, width).
        border_tolerance: The minimum distance from the border to keep a mask.

    Returns:
        1) A list of masks that are too close to the borders.
        2) A list of masks that are not too close to the borders.
    """
    border_masks = []
    non_border_masks = []
    for mask in masks:
        if is_border_object(Mask(mask), image_shape, border_tolerance):
            border_masks.append(mask)
        else:
            non_border_masks.append(mask)
    return border_masks, non_border_masks


def compute_border_metrics(
    gt_paths: list[Path], 
    pred_paths: list[Path], 
    image_shape: tuple[int, int],
    iou_threshold: float = 0.7
) -> dict:
    """Compute metrics for border objects only.

    This function computes precision and recall for border objects.
    """
    return compute_precision_recall(
        gt_paths, 
        pred_paths, 
        image_shape, 
        iou_threshold, 
        gt_mask_filter=lambda x: find_border_masks(x, image_shape)[0], 
        pred_mask_filter=lambda x: find_border_masks(x, image_shape)[0]
    )


def compute_non_border_metrics(
    gt_paths: list[Path], 
    pred_paths: list[Path], 
    image_shape: tuple[int, int],
    iou_threshold: float = 0.7
) -> dict:
    """Compute metrics for non-border objects only.

    This function computes precision and recall for non-border objects.
    """
    return compute_precision_recall(
        gt_paths, 
        pred_paths, 
        image_shape, 
        iou_threshold, 
        gt_mask_filter=lambda x: find_border_masks(x, image_shape)[1], 
        pred_mask_filter=lambda x: find_border_masks(x, image_shape)[1]
    )
    