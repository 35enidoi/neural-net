
from inspect import isclass
from typing import Callable, Optional

from neuralnet.abstracts import (
    AbstractActivationAlgorithm, AbstractActivationAlgorithmNoStatic,
    AbstractLossAlgorithm, AbstractLossAlgorithmNoStatic
)
from neuralnet.active_funcs import Sigmoid, Identity
from neuralnet.loss_funcs import MeanSquaredError
from neuralnet._exception_messages import NNExceptionMessages
from neuralnet.nn.neuralnode import NeuralNode


__all__ = ["FullyConnectedNeuralNetwork",]


class FullyConnectedNeuralNetwork:
    def __init__(self,
                 node_nums: list[int],
                 learn_rate: float = 0.1,
                 activate_funcs: Optional[list[AbstractActivationAlgorithmNoStatic | AbstractActivationAlgorithm]] = None,
                 loss_func: AbstractLossAlgorithm | AbstractLossAlgorithmNoStatic = MeanSquaredError
                 ) -> None:
        # 学習率(eta)
        self.eta = learn_rate

        # レイヤー数の確認
        self.layer = len(node_nums)
        if self.layer < 2:
            raise ValueError(NNExceptionMessages.NN_INIT_LAYER_NOM)

        # 活性化関数のチェック
        if activate_funcs:
            # 実装めんどくさいのでactivate_funcsのpropertyに丸投げする(おい)
            self.activate_funcs = activate_funcs
        else:
            if self.layer == 2:
                # レイヤー数2の時、出力層のみ活性化になるのでidentityじゃなくてsigmoidの方がいい
                self.__activate_funcs = [Sigmoid]
            else:
                self.__activate_funcs = []

                # 中間層にシグモイド関数を追加
                for _ in range(self.layer - 2):
                    self.__activate_funcs.append(Sigmoid)
                # 出力層は恒等関数
                self.__activate_funcs.append(Identity)

        # 損失関数のチェック
        self.__loss_function = loss_func

        self.nodes = [[NeuralNode(layer) for _ in range(num)] for layer, num in enumerate(node_nums)]
        for layer_pos in range(self.layer - 1):
            for node in self.nodes[layer_pos]:
                for next_node in self.nodes[layer_pos + 1]:
                    node.pairlink(next_node)

    @property
    def activate_funcs(self):
        return self.__activate_funcs.copy()

    @activate_funcs.setter
    def activate_funcs(self, x):
        if len(x) != self.layer - 1:
            raise ValueError(NNExceptionMessages.NN_ACV_LAY_NOM)

        for activate_func in x:
            if isclass(activate_func):
                # クラスの時(staticな方じゃないとだめ)
                if not issubclass(activate_func, AbstractActivationAlgorithm):
                    raise ValueError(NNExceptionMessages.NN_ACV_FUNC_NOM)
            else:
                # インスタンスの時(どっちでもいい)
                if not issubclass(type(activate_func), AbstractActivationAlgorithm | AbstractActivationAlgorithmNoStatic):
                    raise ValueError(NNExceptionMessages.NN_ACV_FUNC_INSTANCE_NOM)

        # ここまでこれたなら確認完了
        self.__activate_funcs = x

    @property
    def loss_function(self):
        return self.__loss_function

    @loss_function.setter
    def loss_function(self, x):
        if isclass(x):
            if not issubclass(x, AbstractLossAlgorithm):
                raise ValueError()
        else:
            if not issubclass(type(x), AbstractLossAlgorithm | AbstractLossAlgorithmNoStatic):
                raise ValueError()

        # ここまでこれたなら確認完了
        self.__loss_function: AbstractLossAlgorithm | AbstractLossAlgorithmNoStatic = x

    def predict(self, *inputs: int) -> list[float]:
        if len(inputs) != len(self.nodes[0]):
            raise ValueError(NNExceptionMessages.NN_PREDICT_INPUT_NOM)
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
                    # __activate_funcsでnode.layer - 1なのはactivate_funcsは入力層を含めないから(1レイヤー分ずれている)
                    node.value = self.__activate_funcs[node.layer - 1].execute(node.value + node.bias)
                    for next_node, weight in node.pairent:
                        next_node.value += node.value * weight

            # 出力層の値をリスト化して返す
            return [node.value for node in self.nodes[-1]]

    def _back_propagation(self, real_answer: list[float]) -> None:
        # デルタ値を初期化
        for nodes in self.nodes:
            for node in nodes:
                node.delta_value_reset()

        # 出力層のバイアス、それにつながっているエッヂの重さ調整
        for pos, output_node in enumerate(self.nodes[-1]):
            diff = self.__loss_function.execute_derivative(self.nodes[-1][pos].value, real_answer[pos])
            # __activate_funcsでnode.layer - 1なのはactivate_funcsは入力層を含めないから(1レイヤー分ずれている)
            output_node.delta_value = diff * self.__activate_funcs[-1].execute_derivative(output_node.value)

            output_node.bias -= self.eta * output_node.delta_value
            for node in self.nodes[-2]:
                node.pairent[pos][1] -= self.eta * output_node.delta_value * node.value

        # 入力層と出力層を除いた中間層についてバイアスとエッヂの重さ調整
        for nodes in reversed(self.nodes[1:-1]):
            for pos, node in enumerate(nodes):
                error_sums = sum(next_node.delta_value * weight for next_node, weight in node.pairent)
                # node.layer - 1の部分は上を参照
                node.delta_value = error_sums * self.__activate_funcs[node.layer - 1].execute_derivative(node.value)

                node.bias -= self.eta * node.delta_value
                for before_node in self.nodes[node.layer - 1]:
                    before_node.pairent[pos][1] -= self.eta * node.delta_value * node.value

    def train(self,
              data: list[list[int]],
              answer_func: Callable[[list[int]], list[float]],
              ) -> list[float]:

        error_list: list[float] = []

        for inputs in data:
            # 順伝搬(予測)
            ans = self.predict(*inputs)

            # 教師の答えを持ってくる
            real_answer = answer_func(inputs)

            # エラー値追加
            error_list.append(self.__loss_function.execute(ans, real_answer))

            # 誤差逆伝搬
            self._back_propagation(real_answer)

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
