from random import uniform

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from _util import show_error, add_src_to_path
add_src_to_path()

from neuralnet.nn import FullyConnectedNeuralNetwork as FCNN
from neuralnet.active_funcs import Identity
from neuralnet.loss_funcs import HuberLoss


# このファイル全部copilotに書いてもらった
# すごくない？

def regression_error_function(input: list[float]) -> tuple[list[float], float]:
    x = input[0]
    real_ans = 2.0 * x + 3.0  # Ground truth: y = 2x + 3
    return [real_ans,]


def create_training_input(length: int) -> list[list[float]]:
    return [[uniform(-10, 10)] for _ in range(length)]  # Random x values in range [-10, 10]


if __name__ == "__main__":
    # input: 2 -> output: 1
    # no middle layer
    # activate function: Identity(恒等関数)
    nn = FCNN([1, 1], learn_rate=0.01, activate_funcs=[Identity], loss_func=HuberLoss())

    # Training
    train_num = 1000
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

    #  エラーの表示
    show_error(errors, step=10)

    if plt:
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
