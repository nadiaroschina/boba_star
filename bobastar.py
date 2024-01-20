import time

from collections import defaultdict
from heapq import heappop, heappush
from dataclasses import dataclass

from typing import List, Tuple, DefaultDict

from map_and_scenarios import Map, build_reversed_map, compute_cost
from dijkstra_heuristics import dijkstra_for_heuristics
from utils import AlgorithmResult, Node, ForwardNode, BackwardNode

INF = int(1e12)


@dataclass
class BobaAlgorithmResult:
    pareto_set: List[Tuple[int, int]]
    total_iterations_count: int
    estimated_iterations_count_in_parallel: int
    total_time_elapsed: float
    estimated_time_elapsed_in_parallel: float


class BOBA:

    def __init__(
        self,
        task_map: Map,
        reversed_map: Map,
        s_start: int,
        s_goal: int,

        h1: DefaultDict[int, int],
        h2: DefaultDict[int, int],
        h1_dash: DefaultDict[int, int],
        h2_dash: DefaultDict[int, int],

        ub1: DefaultDict[int, int],
        ub2: DefaultDict[int, int],
        ub1_dash: DefaultDict[int, int],
        ub2_dash: DefaultDict[int, int]
    ):
        self.task_map = task_map
        self.reversed_map = reversed_map
        self.s_start = s_start
        self.s_goal = s_goal

        self.h1 = h1
        self.h2 = h2
        self.h1_dash = h1_dash
        self.h2_dash = h2_dash

        self.ub1 = ub1
        self.ub2 = ub2
        self.ub1_dash = ub1_dash
        self.ub2_dash = ub2_dash

        self.min_g1 = defaultdict(lambda: INF)
        self.min_g2 = defaultdict(lambda: INF)

        self.solutions_forward: List[Tuple[int, int]] = list()
        self.solutions_backward: List[Tuple[int, int]] = list()

        self._open_forward: List[Node] = list()
        self._open_backward: List[Node] = list()

        self.time_elapsed_forward: float = 0
        self.time_elapsed_backward: float = 0

        start_node_forward = ForwardNode(
            s=s_start,
            g1=0, g2=0,
            h1=h1[s_start], h2=h2[s_start],
            parent=None
        )
        self._open_forward.append(start_node_forward)
        self._finished_forward = False

        start_node_backward = BackwardNode(
            s=s_goal,
            g1=0, g2=0,
            h1=h1_dash[s_goal], h2=h2_dash[s_goal],
            parent=None
        )
        self._open_backward.append(start_node_backward)
        self._finished_backward = False

        self.preferred_next_step = 'forward'

    def step_forward(self) -> None:

        start_time = time.time()

        x = heappop(self._open_forward)

        if x.f1 >= self.min_g1[self.s_start]:
            self._finished_forward = True
            return

        if (
            x.g2 >= self.min_g2[x.s]
            or x.f2 >= self.min_g2[self.s_goal]
        ):
            return

        if self.min_g2[x.s] == INF:
            self.h1_dash[x.s] = x.g1

        self.min_g2[x.s] = x.g2

        if x.s == self.s_goal:
            z = self.solutions_forward[-1] if self.solutions_forward else None
            if z and z[0] == x.f1:
                self.solutions_forward.pop()
            self.solutions_forward.append((x.g1, x.g2))
            return

        if x.g2 + self.ub2[x.s] < self.min_g2[self.s_goal]:
            self.min_g2[self.s_goal] = x.g2 + self.ub2[x.s]
            z = self.solutions_forward[-1] if self.solutions_forward else None
            if z and z[0] == x.f1:
                self.solutions_forward.pop()
            self.solutions_forward.append(
                (x.g1 + self.h1[x.s], x.g2 + self.ub2[x.s]))
            if self.h1[x.s] == self.ub1[x.s]:
                return

        for t in self.task_map.get_neighbors(x.s):
            y = ForwardNode(
                s=t,
                g1=compute_cost(self.task_map, x.s, t, 1) + x.g1,
                g2=compute_cost(self.task_map, x.s, t, 2) + x.g2,
                h1=self.h1[t], h2=self.h2[t],
                parent=x
            )

            if (
                y.g2 >= self.min_g2[t]
                or y.f1 >= self.min_g1[self.s_start]
                or y.f2 >= self.min_g2[self.s_goal]
            ):
                continue

            heappush(self._open_forward, y)

            self.time_elapsed_forward += time.time() - start_time

    def step_backward(self) -> None:

        start_time = time.time()

        x = heappop(self._open_backward)

        if x.f2 >= self.min_g2[self.s_goal]:
            self._finished_backward = True
            return

        if (
            x.g1 >= self.min_g1[x.s]
            or x.f1 >= self.min_g1[self.s_start]
        ):
            return

        if self.min_g1[x.s] == INF:
            self.h2[x.s] = x.g2

        self.min_g1[x.s] = x.g1

        if x.s == self.s_start:
            z = self.solutions_backward[-1] if self.solutions_backward else None
            if z and z[1] == x.f2:
                self.solutions_backward.pop()
            self.solutions_backward.append((x.g1, x.g2))
            return

        if x.g1 + self.ub1_dash[x.s] < self.min_g1[self.s_start]:
            self.min_g1[self.s_start] = x.g1 + self.ub1_dash[x.s]
            z = self.solutions_backward[-1] if self.solutions_backward else None
            if z and z[1] == x.f2:
                self.solutions_backward.pop()
            self.solutions_backward.append(
                (x.g1 + self.ub1_dash[x.s], x.g2 + self.h2_dash[x.s])
            )
            if self.h2_dash[x.s] == self.ub2_dash[x.s]:
                return

        for t in self.reversed_map.get_neighbors(x.s):
            y = BackwardNode(
                s=t,
                g1=compute_cost(self.reversed_map, x.s, t, 1) + x.g1,
                g2=compute_cost(self.reversed_map, x.s, t, 2) + x.g2,
                h1=self.h1_dash[t], h2=self.h2_dash[t],
                parent=x
            )

            if (
                y.g1 >= self.min_g1[t]
                or y.f1 >= self.min_g1[self.s_start]
                or y.f2 >= self.min_g2[self.s_goal]
            ):
                continue

            heappush(self._open_backward, y)

            self.time_elapsed_backward += time.time() - start_time

    def finished(self) -> bool:
        return (
            (self._finished_forward or len(self._open_forward) == 0)
            and (self._finished_backward or len(self._open_backward) == 0)
        )

    def step(self) -> str:

        assert not self.finished()

        # only one type of step is left

        if self._finished_forward or len(self._open_forward) == 0:
            self.step_backward()
            return 'backward'

        if self._finished_backward or len(self._open_backward) == 0:
            self.step_forward()
            return 'forward'

        # both forrward and backward steps are possible

        if self.preferred_next_step == 'forward':
            self.step_forward()
            self.preferred_next_step = 'backward'
            return 'forward'

        else:
            self.step_backward()
            self.preferred_next_step = 'forward'
            return 'backward'

    @property
    def solutions(self) -> List[Tuple[int, int]]:
        return self.solutions_forward + self.solutions_backward


def boba_star(
    task_map: Map,
    s_start: int,
    s_goal: int
) -> BobaAlgorithmResult:

    forward_steps = 0
    backward_steps = 0

    reversed_map = build_reversed_map(task_map)

    h1_dash, ub2_dash = dijkstra_for_heuristics(task_map, s_start, 'c1')
    ub1, h2 = dijkstra_for_heuristics(reversed_map, s_goal, 'c2')

    ub1_dash, h2_dash = dijkstra_for_heuristics(task_map, s_start, 'c2')
    h1, ub2 = dijkstra_for_heuristics(reversed_map, s_goal, 'c1')

    boba = BOBA(
        task_map, reversed_map,
        s_start, s_goal,
        h1, h2,
        h1_dash, h2_dash,
        ub1, ub2,
        ub1_dash, ub2_dash,
    )

    while not boba.finished():
        made_step = boba.step()
        if made_step == 'forward':
            forward_steps += 1
        else:
            backward_steps += 1

    return BobaAlgorithmResult(
        pareto_set=boba.solutions,
        total_iterations_count=forward_steps + backward_steps,
        estimated_iterations_count_in_parallel=max(
            forward_steps, backward_steps),
        total_time_elapsed=boba.time_elapsed_forward + boba.time_elapsed_backward,
        estimated_time_elapsed_in_parallel=max(
            boba.time_elapsed_forward, boba.time_elapsed_backward)
    )


def boba_star_real(
    task_map: Map,
    s_start: int,
    s_goal: int
) -> AlgorithmResult:
    boba_alg_res = boba_star(task_map, s_start, s_goal)
    return AlgorithmResult(
        pareto_set=boba_alg_res.pareto_set,
        iterations_count=boba_alg_res.total_iterations_count,
        time_elapsed=boba_alg_res.total_time_elapsed
    )


def boba_star_emulated_parallel(
    task_map: Map,
    s_start: int,
    s_goal: int
) -> AlgorithmResult:
    boba_alg_res = boba_star(task_map, s_start, s_goal)
    return AlgorithmResult(
        pareto_set=boba_alg_res.pareto_set,
        iterations_count=boba_alg_res.estimated_iterations_count_in_parallel,
        time_elapsed=boba_alg_res.estimated_time_elapsed_in_parallel
    )


class BOAEnh(BOBA):

    def __init__(
        self,
        task_map: Map,
        reversed_map: Map,
        s_start: int,
        s_goal: int,

        h1: DefaultDict[int, int],
        h2: DefaultDict[int, int],
        h1_dash: DefaultDict[int, int],
        h2_dash: DefaultDict[int, int],

        ub1: DefaultDict[int, int],
        ub2: DefaultDict[int, int],
        ub1_dash: DefaultDict[int, int],
        ub2_dash: DefaultDict[int, int]
    ):
        self.task_map = task_map
        self.reversed_map = reversed_map
        self.s_start = s_start
        self.s_goal = s_goal

        self.h1 = h1
        self.h2 = h2
        self.h1_dash = h1_dash
        self.h2_dash = h2_dash

        self.ub1 = ub1
        self.ub2 = ub2
        self.ub1_dash = ub1_dash
        self.ub2_dash = ub2_dash

        self.min_g1 = defaultdict(lambda: INF)
        self.min_g2 = defaultdict(lambda: INF)

        self.solutions_forward: List[Tuple[int, int]] = list()
        self._open_forward: List[Node] = list()

        self.time_elapsed_forward = 0

        start_node_forward = ForwardNode(
            s=s_start,
            g1=0, g2=0,
            h1=h1[s_start], h2=h2[s_start],
            parent=None
        )
        self._open_forward.append(start_node_forward)
        self._finished_forward = False

    def step_backward(self):
        pass

    def finished(self) -> bool:
        return (self._finished_forward or len(self._open_forward) == 0)

    def step(self) -> None:
        self.step_forward()

    @property
    def solutions(self) -> List[Tuple[int, int]]:
        return self.solutions_forward


def boa_star_enh(
    task_map: Map,
    s_start: int,
    s_goal: int
) -> AlgorithmResult:

    reversed_map = build_reversed_map(task_map)

    h1_dash, ub2_dash = dijkstra_for_heuristics(task_map, s_start, 'c1')
    ub1, h2 = dijkstra_for_heuristics(reversed_map, s_goal, 'c2')

    ub1_dash, h2_dash = dijkstra_for_heuristics(task_map, s_start, 'c2')
    h1, ub2 = dijkstra_for_heuristics(reversed_map, s_goal, 'c1')

    steps = 0

    boaenh = BOAEnh(
        task_map, reversed_map,
        s_start, s_goal,
        h1, h2,
        h1_dash, h2_dash,
        ub1, ub2,
        ub1_dash, ub2_dash,
    )

    while not boaenh.finished():
        steps += 1
        boaenh.step()

    return AlgorithmResult(boaenh.solutions, steps, boaenh.time_elapsed_forward)
