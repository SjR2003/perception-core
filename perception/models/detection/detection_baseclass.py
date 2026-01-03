from abc import abstractmethod


class DetectionBackend:
    @abstractmethod
    def __init__(self, model) -> None:
        pass

    @abstractmethod
    def infer(self, X):
        pass
