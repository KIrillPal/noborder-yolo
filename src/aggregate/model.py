from src.aggregate.base import BaseClassifier, NO_OBJECT
from typing import Callable

class ModelClassifier(BaseClassifier):
    def __init__(self, n_classes : int, model : Callable, feat_factory : Callable):
        super().__init__(n_classes)
        self.model = model
        self.feat_factory = feat_factory
        
    def classify(self, groups : list[list[dict]]) -> tuple[list[list[dict]], list[int]]:

        classes = self.model([self.feat_factory(g) for g in groups])
        groups = [
            g for g, c in zip(groups, classes)
            if 0 <= c < self.n_classes
        ]
        return groups, [
            c for c in classes
            if 0 <= c < self.n_classes
        ]