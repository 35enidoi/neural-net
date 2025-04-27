from abc import ABC, abstractmethod

__all__ = [
    # Activation algorithms
    "AbstractActivationAlgorithm", "AbstractActivationAlgorithmNoStatic",
    # Loss functions
    "AbstractLossAlgorithm", "AbstractLossAlgorithmNoStatic"
]


# Activate algorithm

class AbstractActivationAlgorithm(ABC):
    @staticmethod
    @abstractmethod
    def execute(x: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def execute_derivative(x: float) -> float:
        pass


class AbstractActivationAlgorithmNoStatic(ABC):
    @abstractmethod
    def execute(self, x: float) -> float:
        pass

    @abstractmethod
    def execute_derivative(self, x: float) -> float:
        pass


# Loss algorithm

class AbstractLossAlgorithm(ABC):
    @staticmethod
    @abstractmethod
    def execute(predict: list[float], answer: list[float]) -> float:
        pass

    @staticmethod
    @abstractmethod
    def execute_derivative(predict: float, answer: float) -> float:
        pass


class AbstractLossAlgorithmNoStatic(ABC):
    @abstractmethod
    def execute(self, predict: list[float], answer: list[float]) -> float:
        pass

    @abstractmethod
    def execute_derivative(self, predict: float, answer: float) -> float:
        pass
