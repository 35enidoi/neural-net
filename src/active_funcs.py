from abc import ABC, abstractmethod
from math import exp, tanh, sin, cos, log


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


# Linear-like functions
# 線形的(1次関数的)な関数

class Identity(AbstractActivationAlgorithm):
    @staticmethod
    def execute(x):
        return x

    @staticmethod
    def execute_derivative(x):
        return 1.0


class Absolute(AbstractActivationAlgorithm):
    @staticmethod
    def execute(x):
        return abs(x)

    @staticmethod
    def execute_derivative(x):
        if x > 0.0:
            return 1.0
        elif x < 0.0:
            return -1.0
        else:
            # x=0の時微分不可能だけどとりあえず1.0ということにしておく...
            return 1.0


# Logistic-like functions
# 滑らかに変化して一定の値に収束する非線形性を持つ関数

class Sigmoid(AbstractActivationAlgorithm):
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
        if x >= 0.0:
            z = exp(-x)
            return 1 / (1.0 + z)
        else:
            z = exp(x)
            return z / (1.0 + z)

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
        return s * (1.0 - s)


class Tanh(AbstractActivationAlgorithm):
    @staticmethod
    def execute(x):
        return tanh(x)

    @staticmethod
    def execute_derivative(x):
        return 1.0 - tanh(x) ** 2


# Periodric functions
# 周期的な関数

class Sin(AbstractActivationAlgorithm):
    @staticmethod
    def execute(x):
        return sin(x)

    @staticmethod
    def execute_derivative(x):
        return cos(x)


# ReLU-like functions
# ReLU関数(f(x) = max(0, x))とそれに類似する関数

class ReLU(AbstractActivationAlgorithm):
    @staticmethod
    def execute(x):
        return max(0, x)

    @staticmethod
    def execute_derivative(x):
        if x > 0.0:
            return 1.0
        elif x < 0.0:
            return 0.0
        else:
            # x=0において導関数ないけど気にしない気にしない...
            return 0.0


class LReLU(AbstractActivationAlgorithmNoStatic):
    def __init__(self, alfa: float = 0.01):
        self.alfa = alfa

    def execute(self, x):
        if x >= 0.0:
            return x
        else:
            return self.alfa * x

    def execute_derivative(self, x):
        if x >= 0.0:
            return 1.0
        else:
            return self.alfa


class Swish(AbstractActivationAlgorithm):
    @staticmethod
    def execute(x):
        # シグモイド関数実装めんどくさいのでsigmoidから借りる
        return x * Sigmoid.execute(x)

    @staticmethod
    def execute_derivative(x):
        r = Sigmoid.execute(x)
        s = x * r
        return s + r * (1.0 - s)


class ELU(AbstractActivationAlgorithmNoStatic):
    def __init__(self, alfa: float = 1.0):
        self.alfa = alfa

    def execute(self, x):
        if x > 0.0:
            return x
        else:
            return self.alfa * (exp(x) - 1)

    def execute_derivative(self, x):
        if x <= 0.0:
            return self.alfa * exp(x)
        else:
            return 1.0


class SoftPlus(AbstractActivationAlgorithm):
    @staticmethod
    def execute(x):
        return log(1.0 + exp(x))

    @staticmethod
    def execute_derivative(x):
        # 微分した形はシグモイド関数と同じ
        return Sigmoid.execute(x)
