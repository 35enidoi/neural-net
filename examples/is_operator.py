from random import randint

from _add_src_to_path import add_src_to_path
add_src_to_path()

from src.nn import NeuralNetwork


def and_error_funtion(input: list[bool], predict_ans: list[float]) -> tuple[list[float], float]:
    a, b = input
    predict = predict_ans[0]
    real_ans = a is b
    error = 0.5 * (predict - real_ans) ** 2
    return [real_ans,], error


def create_training_input(length: int) -> tuple[bool, bool]:
    return [(randint(0, 1), randint(0, 1)) for _ in range(length)]


if __name__ == "__main__":
    # input: 2 -> middle -> output: 1
    # 1 middle layer (4,)
    nn = NeuralNetwork([2, 8, 1])

    # training
    train_num = 100000
    errors = nn.train(create_training_input(train_num), and_error_funtion)

    # check
    print(f"first error late is {errors[0]:.5f}")
    print(f"final error late is {errors[-1]:.5f}")
    print()  # 改行
    for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)):
        print(f"{a} and {b}, predict: {nn.predict(a, b)[0]:.5f}, answer: {a is b}")
