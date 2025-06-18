NO_OBJECT = 1_000_000_007


class BaseClassifier:
    def __init__(self, n_classes : int):
        self.n_classes = n_classes
        
    def classify(self, groups : list[list[dict]]) -> tuple[list[list[dict]], list[int]]:
        pass


class BaseMerger:
    def __init__(self, n_classes : int):
        self.n_classes = n_classes
        
    def merge(self, groups : list[list[dict]], classes: list[int]):
        pass


class Aggregator:
    def __init__(self, classifier : BaseClassifier, merger: BaseMerger):
        self.classifier_ = classifier
        self.merger_ = merger
        
    def aggregate(self, groups : list[list[dict]]) -> list[dict]:
        groups, classes = self.classifier_.classify(groups)
        merged = self.merger_.merge(groups, classes)
        return [m for m in merged if len(m['points']) > 0]


class ComposeClassifier(BaseClassifier):
    def __init__(self, classifiers : list[BaseClassifier]):
        super().__init__(classifiers[0].n_classes)
        self.classifiers_ = classifiers

    def classify(self, groups : list[list[dict]]) -> tuple[list[list[dict]], list[int]]:
        for classifier in self.classifiers_:
            groups, classes = classifier.classify(groups)
        return groups, classes