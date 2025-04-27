from random import uniform
from functools import cached_property

from neuralnet.abstracts import AbstractActivationAlgorithm, AbstractActivationAlgorithmNoStatic
from neuralnet.active_funcs import ReLU
from neuralnet.nn.util import active_func_check


class Filter:
    def __init__(self, input_size: tuple[int, int], kernel_size: tuple[int, int], stride: int) -> None:
        if input_size[0] < kernel_size[0] or input_size[1] < kernel_size[1]:
            raise ValueError("Kernel size must be less than or equal to input size.")

        if not all(i >= stride for i in (input_size + kernel_size)):
            raise ValueError("Stride must be less than or equal to input size.")

        if not all(i % stride == 0 for i in (input_size + kernel_size)):
            raise ValueError("Stride must be a multiple of input size.")

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self._filter = [[uniform(-1, 1) for _ in range(kernel_size[0])] for _ in range(kernel_size[1])]

    @cached_property
    def output_size(self) -> int:
        return (
            (self.input_size[0] - self.kernel_size[0]) // self.stride + 1,
            (self.input_size[1] - self.kernel_size[1]) // self.stride + 1
        )

    def convilution(self, input_data: list[list[float]]) -> list[list[float]]:
        if len(input_data) != self.input_size[0] or len(input_data[0]) != self.input_size[1]:
            raise ValueError("Input data size does not match filter input size.")

        output = [[0 for _ in range(self.output_size[1])] for _ in range(self.output_size[0])]

        for x_pos in range(self.output_size[0]):
            for y_pos in range(self.output_size[1]):
                for ker_x in range(self.kernel_size[0]):
                    for ker_y in range(self.kernel_size[1]):
                        op = input_data[x_pos * self.stride + ker_x][y_pos * self.stride + ker_y] * self._filter[ker_x][ker_y]
                        output[x_pos][y_pos] += op

        return output


class MaxPooling:
    @staticmethod
    def pooling(input_data: list[list[float]], kernel_size: tuple[int, int], stride: int) -> list[list[float]]:
        ...  # Todo 後で作る


class ConvolutionalNeuralNetwork:
    def __init__(
            self,
            input_size: tuple[int, int],
            kernel_size: tuple[int, int] = (2, 2),
            stride: int = 1,
            filter_num: int = 4,
            activate_func: AbstractActivationAlgorithm | AbstractActivationAlgorithmNoStatic = ReLU) -> None:
        if input_size[0] < kernel_size[0] or input_size[1] < kernel_size[1]:
            raise ValueError("Kernel size must be less than or equal to input size.")

        if not all(i >= stride for i in (input_size + kernel_size)):
            raise ValueError("Stride must be less than or equal to input size.")

        if not all(i % stride == 0 for i in (input_size + kernel_size)):
            raise ValueError("Stride must be a multiple of input size.")

        self.filters = [Filter(input_size, kernel_size, stride) for _ in range(filter_num)]
        self.__activate_func = activate_func

    @property
    def activate_func(self) -> AbstractActivationAlgorithm:
        return self.__activate_func

    @activate_func.setter
    def activate_func(self, func: AbstractActivationAlgorithm) -> None:
        active_func_check(func)
        self.__activate_func = func

    def predict(self, input_data: list[list[float]]) -> list[list[list[float]]]:
        if len(input_data) != self.filters[0].input_size[0] or len(input_data[0]) != self.filters[0].input_size[1]:
            raise ValueError("Input data size does not match filter input size.")

        output = []

        for filter in self.filters:
            conv = filter.convilution(input_data)
            output.append(self._activate(conv))

        return output

    def _activate(self, input_data: list[list[float]]) -> list[list[float]]:
        output = [[self.__activate_func.execute(i) for i in row] for row in input_data]

        return output
