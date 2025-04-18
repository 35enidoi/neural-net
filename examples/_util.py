from sys import path as sys_path
from os import path as os_path

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None


def add_src_to_path() -> None:
    sys_path.append(os_path.abspath(os_path.join(os_path.dirname(__file__), "../")))


def show_error(errors: list[float], step: int = 100) -> None:
    if not errors:
        return
    elif step < 1:
        raise ValueError("Step must be greater than or equal to 1")
    if plt:
        avg_error = [sum(errors[i:i+step]) / step for i in range(0, len(errors), step)]
        plt.plot(range(len(errors[::step])), avg_error)
        plt.xlabel(f"Step (x{step})")
        plt.ylabel("Average Error")
        plt.title(f"Average Error per {step} Steps")
        plt.show()
