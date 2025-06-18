import numpy as np

from src.aggregate.base import BaseClassifier
from src.utils import Mask, is_border_object

class ConfidenceClassifier(BaseClassifier):
    def __init__(self, n_classes : int, threshold : float = 0.5):
        super().__init__(n_classes)
        self.threshold = threshold
        
    def classify(self, groups : list[list[dict]]) -> tuple[list[list[dict]], list[int]]:
        groups = [
            [o for o in g if (o['conf'] if 'conf' in o else 1.0) > self.threshold] for g in groups
        ]
        groups = [g for g in groups if len(g) > 0]
        return groups, [
            max(
                list(range(self.n_classes)), 
                key=lambda x: len([1 for obj in group if obj['cls'] == x])
            )
            for group in groups
        ]


class MaxConfidenceClassifier(BaseClassifier):
    def __init__(self, n_classes : int, threshold : float = 0.7):
        super().__init__(n_classes)
        self.threshold = threshold
        
    def classify(self, groups : list[list[dict]]) -> tuple[list[list[dict]], list[int]]:
        max_confs = [
            max(
                [o['conf'] if 'conf' in o else 1.0 for o in g],
                default=0.0
            )
            for g in groups
        ]
        groups = [g for g, mc in zip(groups, max_confs) if mc > self.threshold]
        return groups, [
            max(
                list(range(self.n_classes)), 
                key=lambda x: len([1 for obj in group if obj['cls'] == x])
            )
            for group in groups
        ]