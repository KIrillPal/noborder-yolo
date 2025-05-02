import numpy as np

from src.aggregate.base import BaseClassifier
from src.utils import Mask, is_border_object

class ConfidenceClassifier(BaseClassifier):
    def __init__(self, n_classes : int, threshold : int = 16):
        super().__init__(n_classes)
        self.threshold = threshold
        
    def classify(self, groups : list[list[dict]]) -> tuple[list[list[dict]], list[int]]:
        mean_confidences = np.stack([
            np.array([(obj['conf'] if 'conf' in obj else 1.0) for obj in group]).mean()
            for group in groups
        ])
        groups = [
            g for g, c in zip(groups, mean_confidences)
            if c > self.threshold
        ]
        return groups, [
            max(
                list(range(self.n_classes)), 
                key=lambda x: len([1 for obj in group if obj['cls'] == x])
            )
            for group in groups
        ]