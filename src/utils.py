import numpy as np

class Mask:
    label: int
    points: list[tuple[float]]
    
    def __init__(self, other : dict):
        self.label = other['cls']
        self.points = other['points']
        
    def to_dict(self):
        return {
            'cls': self.label,
            'points': self.points
        }


def is_border_object(
    mask: Mask, 
    image_shape: tuple, 
    border_threshold: float = 0.022
) -> bool:
    """
    Checks if the mask is a border object by determining if any of its points 
    are close to the image edges by the threshold.

    Args:
        mask (Mask): The mask object with label and points.
        image_shape (tuple): The shape of the image (height, width, ...).
        border_threshold (float): Ratio of image dimensions for border proximity.

    Returns:
        bool: True if mask represents the border object, False otherwise.
    """
    height, width = image_shape[:2]
    points = np.array(mask.points)

    # Check if any point is within the border threshold of the image edges
    is_touching_border = np.any(
        (points[:, 0] < border_threshold * height) | 
        (points[:, 0] > (1 - border_threshold) * height) | 
        (points[:, 1] < border_threshold * width) | 
        (points[:, 1] > (1 - border_threshold) * width)
    )
    
    return is_touching_border