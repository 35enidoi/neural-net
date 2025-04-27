from __future__ import annotations

from random import uniform


class NeuralNode:
    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.bias = uniform(-1, 1)
        self.delta_value: float = 0
        self.value: float = 0
        self.pairent: list[list[NeuralNode, float]] = []

    def pairlink(self, pair_node: NeuralNode) -> None:
        self.pairent.append([pair_node, uniform(-1, 1)])

    def value_reset(self) -> None:
        """initialize the value to zero."""
        self.value = 0

    def delta_value_reset(self) -> None:
        """initialize the delta value to zero."""
        self.delta_value = 0
