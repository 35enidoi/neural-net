from abc import ABC, abstractmethod


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


class MeanSquaredError(AbstractLossAlgorithm):
    """平均二乗誤差"""
    @staticmethod
    def execute(predict, answer):
        error = 0
        for a, b in zip(predict, answer):
            error += (a - b) ** 2

        return error

    @staticmethod
    def execute_derivative(predict, answer):
        return 2 * (predict - answer)


class MeanAbsoluteError(AbstractLossAlgorithm):
    """平均絶対誤差"""
    @staticmethod
    def execute(predict, answer):
        error = sum(abs(p - a) for p, a in zip(predict, answer))
        return error

    @staticmethod
    def execute_derivative(predict, answer):
        if (error := (predict - answer)) > 0:
            return 1
        elif error < 0:
            return -1
        else:
            # 本当は0で微分不可能だけど0ということにしておく
            return 0


class HuberLoss(AbstractLossAlgorithmNoStatic):
    """フーバー損失"""
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def execute(self, predict, answer):
        error = 0

        for p, a in zip(predict, answer):
            absolute_error = abs(a - p)
            if (absolute_error := abs(a - p)) <= self.delta:
                error += 0.5 * (absolute_error ** 2)
            else:
                error += self.delta * (absolute_error - (0.5 * self.delta))

        return error

    def execute_derivative(self, predict, answer):
        if abs(answer - predict) <= self.delta:
            return predict - answer
        else:
            return self.delta * self.__sign(predict - answer)

    def __sign(self, x: float):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
