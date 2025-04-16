from random import uniform

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from _add_src_to_path import add_src_to_path
add_src_to_path()

from src.nn import NeuralNetwork
from src.active_funcs import ReLU, Identity


# このファイル全部copilotに書いてもらった
# すごくない？

def regression_error_function(input: list[float], predict_ans: list[float]) -> tuple[list[float], float]:
    x = input[0]
    real_ans = 2.0 * x + 3.0  # Ground truth: y = 2x + 3
    predict = predict_ans[0]
    error = 0.5 * (predict - real_ans) ** 2  # Mean squared error
    return [real_ans], error


def create_training_input(length: int) -> list[list[float]]:
    return [[uniform(-10, 10)] for _ in range(length)]  # Random x values in range [-10, 10]


if __name__ == "__main__":
    # ReLU系はなんか学習率をかなり下げないとバグりやすい
    # 遅い学習率のためにレイヤーのノード数爆上げしておく
    # input: 2 -> 1 middle -> output: 1
    # 1 middle layer (16,)
    nn = NeuralNetwork([1, 16, 1], 0.00001, [ReLU, Identity])

    # Training
    train_num = 10000
    errors = nn.train(create_training_input(train_num), regression_error_function)

    # Check training results
    print(f"First error rate: {errors[0]:.5f}")
    print(f"Final error rate: {errors[-1]:.5f}")
    print()  # Line break

    # Test the trained model
    for x in [-10, -5, 0, 5, 10]:
        predicted = nn.predict(x)[0]
        actual = 2.0 * x + 3.0
        print(f"x: {x}, Predicted: {predicted:.5f}, Actual: {actual:.5f}")

    if plt:
        # matplotlibがある場合表示する
        # Plot the error rates
        step = 100
        avg_errors = [sum(errors[i:i+step]) / step for i in range(0, len(errors), step)]
        plt.plot(range(len(avg_errors)), avg_errors)
        plt.xlabel("Step (x10)")
        plt.ylabel("Average Error")
        plt.title(f"Average Error per {step} Steps")
        plt.show()

        # 実際に表示してテストする
        # Plot the actual function and the neural network's approximation
        x_values = [x / 10.0 for x in range(-100, 101)]  # Generate x values from -10 to 10
        actual_values = [2.0 * x + 3.0 for x in x_values]  # Actual function: y = 2x + 3
        predicted_values = [nn.predict(x)[0] for x in x_values]  # Predicted values from the neural network

        plt.figure()
        plt.plot(x_values, actual_values, label="Actual Function (y = 2x + 3)", color="blue")
        plt.plot(x_values, predicted_values, label="NN Approximation", color="red", linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Actual Function vs Neural Network Approximation")
        plt.legend()
        plt.show()
