from src.aggregate.base import BaseClassifier
from src.utils import Mask, is_border_object

class NoBorderClassifier(BaseClassifier):
    def __init__(self, n_classes : int, image_shape : tuple):
        super().__init__(n_classes)
        self.image_shape = image_shape
        
    def classify(self, groups : list[list[dict]]) -> tuple[list[list[dict]], list[int]]:
        groups = [
            [obj for obj in g if not obj['is_border']] 
            for g in groups
        ]
        groups = [g for g in groups if len(g) > 0]
        return groups, [
            max(
                list(range(self.n_classes)), 
                key=lambda x: len([1 for obj in group if obj['cls'] == x])
            )
            for group in groups
        ]