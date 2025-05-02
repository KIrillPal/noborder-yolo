from src.aggregate.base import BaseClassifier, NO_OBJECT

class ThresholdClassifier(BaseClassifier):
    def __init__(self, n_classes : int, threshold : int):
        super().__init__(n_classes)
        self.threshold = threshold
        
    def classify(self, groups : list[list[dict]]) -> tuple[list[list[dict]], list[int]]:
        groups = [
            g for g in groups
            if len(g) >= self.threshold
        ]
        return groups, [
            max(
                list(range(self.n_classes)), 
                key=lambda x: len([1 for obj in group if obj['cls'] == x])
            )
            for group in groups
        ]