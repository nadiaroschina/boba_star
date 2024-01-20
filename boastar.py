import time

from collections import defaultdict
from heapq import heappop, heappush

from typing import Dict, List, DefaultDict

from map_and_scenarios import Map, build_reversed_map, compute_cost
from dijkstra_heuristics import dijkstra_for_heuristics
from utils import AlgorithmResult, Node, ForwardNode


INF = int(1e12)


class SearchTreeBOA:
    """
    Search tree used in BOA* algorithm
    """

    def __init__(self):
        self._open: List[ForwardNode] = list()
        self._closed: Dict[int, ForwardNode] = dict()
        self._min_g1: DefaultDict[int, int] = defaultdict(lambda: INF)
        self._min_g2: DefaultDict[int, int] = defaultdict(lambda: INF)

    def __len__(self) -> int:
        return len(self._open) + len(self._closed)

    def open_is_empty(self) -> bool:
        return len(self._open) == 0

    def add_to_open(self, item: ForwardNode):
        heappush(self._open, item)

    def update_min_g2(self, item: ForwardNode):
        self._min_g2[item.s] = min(self._min_g2[item.s], item.g2)

    def get_best_node_from_open(self) -> ForwardNode:
        return heappop(self._open)

    def get_min_g2(self, s: int):
        return self._min_g2[s]

    @property
    def opened(self):
        return self._open

    @property
    def expanded(self):
        return self._closed.values()


def boa_star(
    task_map: Map,
    s_start: int,
    s_goal: int,
) -> AlgorithmResult:
    """
    BOA* search algorithm
    """

    reversed_map = build_reversed_map(task_map)

    # estimeted path length to s_goal
    h1, _ = dijkstra_for_heuristics(reversed_map, s_goal, 'c1')
    _, h2 = dijkstra_for_heuristics(reversed_map, s_goal, 'c2')

    start_time = time.time()

    boa = SearchTreeBOA()
    start_node = ForwardNode(
        s_start,
        g1=0, g2=0,
        h1=h1[s_start], h2=h2[s_start],
        parent=None
    )
    boa.add_to_open(start_node)

    solutions: List[Node] = list()

    steps = 0

    while not boa.open_is_empty():

        steps += 1

        x = boa.get_best_node_from_open()

        if (
            x.g2 >= boa.get_min_g2(x.s)
            or x.f2 >= boa.get_min_g2(s_goal)
        ):
            continue

        boa.update_min_g2(x)

        if x.s == s_goal:
            solutions.append(x)

        for t in task_map.get_neighbors(x.s):

            y = ForwardNode(
                s=t,
                g1=compute_cost(task_map, x.s, t, 1) + x.g1,
                g2=compute_cost(task_map, x.s, t, 2) + x.g2,
                h1=h1[t], h2=h2[t],
                parent=x
            )

            if not (
                y.g2 >= boa.get_min_g2(y.s)
                or y.f2 >= boa.get_min_g2(s_goal)
            ):
                boa.add_to_open(y)

    execution_time = time.time() - start_time

    return AlgorithmResult(
        pareto_set=[(node.g1, node.g2) for node in solutions],
        iterations_count=steps,
        time_elapsed=execution_time
    )
