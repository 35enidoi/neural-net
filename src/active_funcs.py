from abc import ABC, abstractmethod
from math import exp, tanh


class AbstractAcitivationAlgorithm(ABC):
    @staticmethod
    @abstractmethod
    def execute(x: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def execute_derivative(x: float) -> float:
        pass


class Sigmoid(AbstractAcitivationAlgorithm):
    @staticmethod
    def execute(x):
        """
        シグモイド関数！！！

        Parameters
        ----------
        x : float
            引数！

        Returns
        -------
        float
            返り値！

        Note
        ----
        - シグモイド関数の計算式は`1 / (1 + e ^ -x)`だぞ！
        """
        if x >= 0:
            z = exp(-x)
            return 1 / (1 + z)
        else:
            z = exp(x)
            return z / (1 + z)

    @staticmethod
    def execute_derivative(x):
        """
        シグモイド関数を微分したやつ！

        Parameters
        ----------
        x : float
            引数！

        Returns
        -------
        float
            返り値！

        Note
        ----
        - シグモイド関数を微分したやつの計算式はシグモイド関数をsigとして`sig(x) * (1 - sig(x))`だぞ！
        """
        s = Sigmoid.execute(x)
        return s * (1 - s)


class Tanh(AbstractAcitivationAlgorithm):
    @staticmethod
    def execute(x):
        return tanh(x)

    @staticmethod
    def execute_derivative(x):
        return 1 - tanh(x) ** 2


class ReLU(AbstractAcitivationAlgorithm):
    @staticmethod
    def execute(x):
        return max(0, x)

    @staticmethod
    def execute_derivative(x):
        if x > 0:
            return 1
        else:
            # x=0において導関数ないけど気にしない気にしない...
            return 0
