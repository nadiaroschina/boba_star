from collections import defaultdict
from heapq import heappop, heappush

from typing import Dict, List, Optional, Tuple, DefaultDict

from map_and_scenarios import Map, compute_cost

INF = int(1e12)


class DijkstraNode:
    """
    Represents a search node.

    Attributes
    ----------
    s : int
        Number of state

    g1, g2 : int
        g-value of the node

    parent : Node
        Pointer to the parent node

    """

    def __init__(
        self,
        s: int,
        g1: int = INF,
        g2: int = INF,
        parent: Optional["DijkstraNode"] = None,
    ):
        self.s = s
        self.g1 = g1
        self.g2 = g2
        self.parent = parent

    def __eq__(self, other):
        return self.s == other.s

    def __hash__(self):
        return hash(self.s)


class ForwardDijkstraNode(DijkstraNode):
    """
    This version of the DijkstraNode uses cost1 as the cost function
    """

    def __lt__(self, other):
        if self.g1 < other.g1:
            return True
        if self.g1 == other.g1:
            return self.g2 < other.g2


class BackwardDijkstraNode(DijkstraNode):
    """
    This version of the DijkstraNode uses cost2 as the cost function
    """

    def __lt__(self, other):
        if self.g2 < other.g2:
            return True
        if self.g2 == other.g2:
            return self.g1 < other.g1


class DijkstraSearchTree:
    """
    SearchTree using a priority queue for OPEN and a dictionary for CLOSED
    """

    def __init__(self):
        self._open: List[DijkstraNode] = list()
        self._closed: Dict[int, DijkstraNode] = dict()
        self.min_g1: DefaultDict[int, int] = defaultdict(lambda: INF)
        self.min_g2: DefaultDict[int, int] = defaultdict(lambda: INF)

    def __len__(self) -> int:
        return len(self._open) + len(self._closed)

    def open_is_empty(self) -> bool:
        return len(self._open) == 0

    def add_to_open(self, item: DijkstraNode):
        heappush(self._open, item)

    def get_best_node_from_open(self) -> Optional[DijkstraNode]:
        while self._open:
            best_node = heappop(self._open)
            if not self.was_expanded(best_node):
                return best_node
        return None

    def add_to_closed(self, item: DijkstraNode):
        self._closed[(item.s)] = item
        self.min_g1[item.s] = item.g1
        self.min_g2[item.s] = item.g2

    def was_expanded(self, item: DijkstraNode) -> bool:
        return (item.s) in self._closed

    @property
    def opened(self):
        return self._open

    @property
    def expanded(self):
        return self._closed.values()


def dijkstra_for_heuristics(
    task_map: Map,
    s_start: int,
    optimize_for: str
) -> Tuple[DefaultDict[int, int], DefaultDict[int, int]]:
    """
    Implementation of Dijkstra algorithm

    Attributes
    ----------
    task_map: Map
        representation of the graph

    s_start: int
        number of the state to start search from

    optimize_for: str
        'c1' or 'c2'
        which cost function to optimize ('c1' - direction, 'c2' - time)

    Returns
    ----------
    defaultdict[int, int]
        computed g-values for vertices
    """
    dst = DijkstraSearchTree()
    start_node = (
        ForwardDijkstraNode(s_start, g1=0, g2=0, parent=None)
        if optimize_for == 'c1'
        else BackwardDijkstraNode(s_start, g1=0, g2=0, parent=None)
    )
    dst.add_to_open(start_node)

    while not dst.open_is_empty():

        x = dst.get_best_node_from_open()
        if x is None:
            break

        dst.add_to_closed(x)

        for t in task_map.get_neighbors(x.s):
            y = (
                ForwardDijkstraNode(t)
                if optimize_for == 'c1'
                else BackwardDijkstraNode(t)
            )

            if not dst.was_expanded(y):
                y.g1 = compute_cost(task_map, x.s, y.s, 1) + x.g1
                y.g2 = compute_cost(task_map, x.s, y.s, 2) + x.g2
                y.parent = x

            dst.add_to_open(y)

    return dst.min_g1, dst.min_g2
