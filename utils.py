from typing import List, Optional, Tuple

from dataclasses import dataclass


INF = int(1e12)


@dataclass
class AlgorithmResult:
    pareto_set: List[Tuple[int, int]]
    iterations_count: int
    time_elapsed: float


class Node:
    """
    Represents a search node.

    Attributes
    ----------
    s : int
        Number of the state

    g1, g2 : int
        g-value of the node

    f1, f2 : int
        f-value of the node

    parent : Node
        Pointer to the parent node

    """

    def __init__(
        self,
        s: int,
        g1: int = INF, g2: int = INF,
        h1: int = 0, h2: int = 0,
        parent: Optional["Node"] = None,
    ):
        self.s = s
        self.g1 = g1
        self.g2 = g2
        self.h1 = h1
        self.h2 = h2
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0

    @property
    def f1(self):
        return self.g1 + self.h1

    @property
    def f2(self):
        return self.g2 + self.h2

    def __eq__(self, other):
        return self.s == other.s

    def __str__(self) -> str:
        return f's={self.s}, g=({self.g1}, {self.g2}), h=({self.h1}, {self.h2})'


class ForwardNode(Node):

    def __lt__(self, other):
        if self.f1 != other.f1:
            return self.f1 < other.f1
        if self.f2 != other.f2:
            return self.f2 < other.f2
        # todo: tie-breaking
        return False


class BackwardNode(Node):

    def __lt__(self, other):
        if self.f2 != other.f2:
            return self.f2 < other.f2
        if self.f1 != other.f1:
            return self.f1 < other.f1
        # todo: tie-breaking
        return False
