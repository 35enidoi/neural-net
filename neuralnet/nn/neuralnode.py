from __future__ import annotations

from random import uniform
from dataclasses import dataclass


@dataclass()
class Connection:
    node: NeuralNode
    weight: float = uniform(-1, 1)


class NeuralNode:
    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.bias = uniform(-1, 1)
        self.delta_value: float = 0
        self.value: float = 0
        self.pairent: list[Connection] = []

    def pairlink(self, pair_node: NeuralNode) -> None:
        self.pairent.append(Connection(pair_node, uniform(-1, 1)))

    def value_reset(self) -> None:
        """initialize the value to zero."""
        self.value = 0

    def delta_value_reset(self) -> None:
        """initialize the delta value to zero."""
        self.delta_value = 0
