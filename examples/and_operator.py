from random import randint

from _util import show_error, add_src_to_path
add_src_to_path()

from neuralnet.nn import NeuralNetwork


def and_error_funtion(input: list[bool]) -> list[float]:
    a, b = input
    real_ans = a and b
    return [real_ans,]


def create_training_input(length: int) -> tuple[bool, bool]:
    return [(randint(0, 1), randint(0, 1)) for _ in range(length)]


if __name__ == "__main__":
    # input: 2 -> output: 1
    # no middle layer
    nn = NeuralNetwork([2, 1])

    # training
    train_num = 50000
    errors = nn.train(create_training_input(train_num), and_error_funtion)

    # check
    print(f"first error late is {errors[0]:.5f}")
    print(f"final error late is {errors[-1]:.5f}")
    print(nn)  # 改行
    for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)):
        print(f"{a} and {b}, predict: {nn.predict(a, b)[0]:.5f}, answer: {a and b}")

    show_error(errors, step=500)
