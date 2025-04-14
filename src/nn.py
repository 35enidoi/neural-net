from __future__ import annotations

from inspect import isclass
from random import uniform
from typing import Callable

from src.active_funcs import Sigmoid, AbstractAcitivationAlgorithm, AbstractAcitivationAlgorithmNoStatic


class NeuralNode:
    def __init__(self, layer: int) -> None:
        """
        Initializes a NeuralNode instance.

        Parameters
        ----------
        layer : int
            The layer index of the neural node in the neural network.

        Attributes
        ----------
        layer : int
            The layer index of the neural node in the neural network.
        bias : float
            The bias value of the neural node, initialized randomly between -1 and 1.
        delta_value : float
            The delta value used for backpropagation, initialized to 0.
        value : float
            The current value of the neural node, initialized to 0.
        pairent : list of list of [NeuralNode, float]
            A list of parent nodes and their associated weights.
        """
        self.layer = layer
        self.bias = uniform(-1, 1)
        self.delta_value: float = 0
        self.value: float = 0
        self.pairent: list[list[NeuralNode, float]] = []

    def pairlink(self, pair_node: NeuralNode) -> None:
        """
        Establishes a connection between the current neural node and a given pair node.

        Parameters
        ----------
        pair_node : NeuralNode
            The neural node to be linked with the current node.

        Returns
        -------
        None
        """
        self.pairent.append([pair_node, uniform(-1, 1)])

    def value_reset(self) -> None:
        """initialize the value to zero."""
        self.value = 0

    def delta_value_reset(self) -> None:
        """initialize the delta value to zero."""
        self.delta_value = 0


class NeuralNetwork:
    """
    NeuralNetwork class for creating and training a simple feedforward neural network.
    This class implements a neural network with fully connected layers, supporting
    forward propagation for predictions and backpropagation for training. The network
    supports customizable learning rates and customizable activation function.

    Methods
    -------
    predict(*inputs: int)
        Performs a forward pass through the neural network to generate predictions.
    train(data: list[list[int]], error_func: Callable[[list[int], list[float]], tuple[list[float], float]]) -> list[float]

    Notes
    -----
    - The network requires at least two layers: an input layer and an output layer.
    - Each node in a layer is connected to every node in the next layer.
    - The `train` method uses backpropagation to adjust weights and biases based on
      the error calculated by the provided error function.

    """

    def __init__(self, node_nums: list[int], learn_rate: float = 0.1) -> None:
        """
        Initialize a neural network with the specified number of nodes in each layer and learning rate.

        Parameters
        ----------
        node_nums : list of int
            A list where each element represents the number of nodes in a layer.
            The length of the list determines the number of layers in the network.
            Must have at least two elements (input and output layers).
        learn_rate : float, optional
            The learning rate for the neural network. Defaults to 0.1.

        Raises
        ------
        ValueError
            If the number of layers (length of `node_nums`) is less than 2.

        Attributes
        ----------
        eta : float
            The learning rate for the neural network.
        layer : int
            The number of layers in the neural network.
        nodes : list of list of NeuralNode
            A nested list where each sublist contains the nodes of a layer.
            Each node is an instance of the `NeuralNode` class.

        Notes
        -----
        Each node in a layer is connected to every node in the next layer using the `pairlink` method.
        """
        self.__activate_func: AbstractAcitivationAlgorithm = Sigmoid
        self.eta = learn_rate

        self.layer = len(node_nums)
        if self.layer < 2:
            raise ValueError("At least an input layer and an output layer are required.")

        self.nodes = [[NeuralNode(layer) for _ in range(num)] for layer, num in enumerate(node_nums)]
        for layer_pos in range(self.layer - 1):
            for node in self.nodes[layer_pos]:
                for next_node in self.nodes[layer_pos + 1]:
                    node.pairlink(next_node)

    @property
    def activate_func(self):
        return self.__activate_func

    @activate_func.setter
    def activate_func(self, cls):
        if isclass(cls):
            if issubclass(cls, AbstractAcitivationAlgorithm):
                self.__activate_func = cls
                return
        else:
            if issubclass(type(cls), AbstractAcitivationAlgorithmNoStatic):
                self.__activate_func = cls
                return

        raise ValueError("The activation functions (class) must be a subclass of AbstractAcitivationAlgorithm.")

    def predict(self, *inputs: int) -> list[float]:
        """
        Perform a forward pass through the neural network to generate predictions.

        Parameters
        ----------
        *inputs : int
            Variable-length input arguments representing the input values to the
            neural network. The number of inputs must match the number of input
            nodes in the network.

        Returns
        -------
        list of float
            A list of output values from the neural network, corresponding to the
            values of the output nodes after the forward pass.

        Raises
        ------
        ValueError
            If the number of inputs does not match the number of input nodes.

        Notes
        -----
        - This method resets the values of all nodes in the network before
          performing the forward pass.
        - The forward pass involves propagating the input values through the
          network layers, applying the activation function to each node,
          and updating the values of connected nodes based on weights and biases.
        """
        if len(inputs) != len(self.nodes[0]):
            raise ValueError("The number of inputs must match the number of input nodes.")
        else:
            # 初期化
            for nodes in self.nodes:
                for node in nodes:
                    node.value_reset()

            # 入力層に入力値代入&次のノードへ伝搬
            for node, value in zip(self.nodes[0], inputs):
                node.value = value
                for next_node, weight in node.pairent:
                    next_node.value += node.value * weight

            # 入力層以降へ順番に伝搬
            for nodes in self.nodes[1:]:
                for node in nodes:
                    node.value = self.__activate_func.execute(node.value + node.bias)
                    for next_node, weight in node.pairent:
                        next_node.value += node.value * weight

            # 出力層の値をリスト化して返す
            return [node.value for node in self.nodes[-1]]

    def train(self,
              data: list[list[int]],
              error_func: Callable[[list[int], list[float]], tuple[list[float], float]]
              ) -> list[float]:
        """
        Trains the neural network using the provided data and error function.

        Parameters
        ----------
        data : list of list of int
            A list of input data where each element is a list of integers representing the input features.
        error_func : Callable[[list[int], list[float]], tuple[list[float], float]]
            A function that calculates the error and provides the expected output.
            It takes the input data and the predicted output as arguments and returns
            a tuple containing the expected output and the error value.

        Returns
        -------
        list of float
            A list of error values for each training iteration.

        Notes
        -----
        - This method performs forward propagation to predict the output,
          and backpropagation to adjust the weights and biases of the network.
        - The learning rate `eta` is used to scale the adjustments to weights and biases.
        """

        error_list: list[float] = []

        for inputs in data:
            # 順伝搬(予測)
            ans = self.predict(*inputs)

            # 誤差逆伝搬(反映)
            # 初期化デルタ値を初期化
            for nodes in self.nodes:
                for node in nodes:
                    node.delta_value_reset()

            # 教師による答えとエラーの大きさを取得
            real_answer, error = error_func(inputs, ans)
            error_list.append(error)

            # 出力層のバイアス、それにつながっているエッヂの重さ調整
            for pos, output_node in enumerate(self.nodes[-1]):
                diff = (output_node.value - real_answer[pos])
                output_node.delta_value = diff * self.__activate_func.execute_derivative(output_node.value)

                output_node.bias -= self.eta * output_node.delta_value
                for node in self.nodes[-2]:
                    node.pairent[pos][1] -= self.eta * output_node.delta_value * node.value

            # 入力層と出力層を除いた中間層についてバイアスとエッヂの重さ調整
            for nodes in reversed(self.nodes[1:-1]):
                for pos, node in enumerate(nodes):
                    error_sums = sum(next_node.delta_value * weight for next_node, weight in node.pairent)
                    node.delta_value = error_sums * self.__activate_func.execute_derivative(node.value)

                    node.bias -= self.eta * node.delta_value
                    for before_node in self.nodes[node.layer - 1]:
                        before_node.pairent[pos][1] -= self.eta * node.delta_value * node.value

        return error_list

    def __str__(self) -> str:
        def _node_name(pos, layer):
            if layer == 0:
                return f"in{pos}"
            elif layer == self.layer - 1:
                return f"ou{pos}"
            else:
                return f"h{pos}"

        return_str = ""

        for layer, nodes in enumerate(self.nodes):
            return_str += f"{layer} layer:"

            if layer == 0:
                return_str += "  # Input Layer"
            elif layer == self.layer - 1:
                return_str += "  # Output Layer"

            return_str += "\n"

            for pos, node in enumerate(nodes):
                if layer == 0:
                    node_str_format = "({name}; value: {value:.3f}, {pairs})\n"
                elif layer == self.layer - 1:
                    node_str_format = "({name}; bias: {bias:.3f}, value: {value:.3f})\n"
                else:
                    node_str_format = "({name}; bias: {bias:.3f}, value: {value:.3f}, {pairs})\n"

                pairs = []
                for pari_pos, pairent in enumerate(node.pairent):
                    pairs.append(f"{_node_name(pari_pos, pairent[0].layer)}={pairent[1]:.3f}")

                return_str += node_str_format.format(
                    name=_node_name(pos, layer),
                    bias=node.bias,
                    value=node.value,
                    pairs=", ".join(pairs)
                )

            return_str += "\n"

        return return_str
