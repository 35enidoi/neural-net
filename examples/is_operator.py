from random import randint

from _util import show_error, add_src_to_path
add_src_to_path()

from neuralnet.nn import NeuralNetwork
from neuralnet.active_funcs import Tanh, Identity  # tanhを活性化関数として使う


def is_error_funtion(input: list[bool]) -> list[float]:
    a, b = input
    real_ans = a is b
    return [real_ans,]


def create_training_input(length: int) -> tuple[bool, bool]:
    return [(randint(0, 1), randint(0, 1)) for _ in range(length)]


if __name__ == "__main__":
    # 隠れ層の活性化にtanhを使う
    # 出力層は恒等関数(f(x)=x、すなわちy=x)(Identity)
    # input: 2 -> 1 middle -> output: 1
    # 1 middle layer (8,)
    nn = NeuralNetwork([2, 8, 1], activate_funcs=[Tanh, Identity])

    # training
    train_num = 10000
    errors = nn.train(create_training_input(train_num), is_error_funtion)

    # check
    print(f"first error late is {errors[0]:.5f}")
    print(f"final error late is {errors[-1]:.5f}")
    print()  # 改行
    for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)):
        print(f"{a} and {b}, predict: {nn.predict(a, b)[0]:.5f}, answer: {a is b}")

    show_error(errors)
