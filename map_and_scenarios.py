from dataclasses import dataclass
from collections import defaultdict

from typing import Dict, List, Tuple, DefaultDict

import matplotlib.pyplot as plt
from tqdm import tqdm

INF = int(1e12)


class Map:
    """
    Represents the problem instance:
    a graph with n states (points on a plane)
    and m edges (possible moves) between them

    Attributes
    ----------
    n : int
        Number of states

    m : int
        Number of edges

    _coords : defaultdict[int, tuple[int, int]]
        Dictionary containing coordinates of each state of the graph

    _edges : defaultdict[int, list[int]]
        Adjacency list

    _edges_cost_1 : defaultdict[tuple[int, int], int]
        Costs of the edges according to the first cost function (travel distance)

    _edges_cost_2 : defaultdict[tuple[int, int], int]
        Costs of the edges according to the second cost function (travel time)
    """

    def __init__(
        self,
        name: str,
        edges:  DefaultDict[int, List[int]],
        cost_1: DefaultDict[Tuple[int, int], int],
        cost_2: DefaultDict[Tuple[int, int], int],
        coords: Dict[int, Tuple[int, int]],
        n: int,
        m: int
    ):
        """
        Initializes the map
        """
        self._name = name[0:2]
        self._edges_cost_1 = cost_1
        self._edges_cost_2 = cost_2
        self._edges = edges
        self._coords = coords
        self._n = n
        self._m = m

    def get_neighbors(self, s: int) -> List[int]:
        """
        Gets a list of neighboring states for a state with this number
        """
        return self._edges[s]

    def get_parameters(self) -> Tuple[int, int]:
        """
        Returns the parameters of the graph
        """
        return self._n, self._m

    def get_coords(self, s) -> Tuple[int, int]:
        """
        Returns the coordinates of a graph state by its number
        """
        return self._coords[s]

    def get_cost_1(self, s1,  s2) -> int:
        """
        Returns the cost of the edge between two states if it exists
        and INF otherwise
        """
        return self._edges_cost_1[(s1,  s2)]

    def get_cost_2(self, s1,  s2) -> int:
        """
        Returns the cost of the edge between two states if it exists
        and INF otherwise
        """
        return self._edges_cost_2[(s1,  s2)]


def build_reversed_map(task_map: Map) -> Map:
    """
    Builds a reversed problem instance: states remain the same,
    edges change their direction
    """

    reversed_edges = defaultdict(list)
    for v, adj in task_map._edges.items():
        for w in adj:
            reversed_edges[w].append(v)

    reversed_cost_1 = defaultdict(lambda: INF)
    for (v, w), c in task_map._edges_cost_1.items():
        reversed_cost_1[(w, v)] = c

    reversed_cost_2 = defaultdict(lambda: INF)
    for (v, w), c in task_map._edges_cost_2.items():
        reversed_cost_2[(w, v)] = c

    return Map(
        task_map._name,
        reversed_edges,
        reversed_cost_1,
        reversed_cost_2,
        task_map._coords,
        task_map._n,
        task_map._m
    )


def read_map_from_file(
    path_cost_1: str,
    path_cost_2: str,
    path_coords: str,
) -> Map:
    """
    Reads map (problem instance) from txt file

    Parameters
    ----------
      path_cost_1 : str
        Path to file with values of the first cost function (distance)

      path_cost_2 : str
        Path to file with values of the second cost function (time)

      path_coords : str
        Path to file with states coordinates

    Returns
    -------
      map : Map
        An object of type Map containing information about the map (problem instance)
    """
    n, m = None, None
    coords = dict()
    cost_1 = defaultdict(lambda: INF)
    cost_2 = defaultdict(lambda: INF)
    edges = defaultdict(list)
    name = path_cost_1.replace('data/', '')

    # states coordinates
    with open(path_coords) as coords_file:
        for line in coords_file:
            if line.startswith("v"):
                v_args = line.split()
                coords[int(v_args[1])] = (int(v_args[2]), int(v_args[3]))

    # first cost function (distance), graph parametrs, edges
    with open(path_cost_1) as cost_1_file:
        for line in cost_1_file:
            # graph parametrs
            if line.startswith("p"):
                graph_args = line.split()
                n, m = int(graph_args[2]), int(graph_args[3])
            # edges and first cost function
            if line.startswith("a"):
                e_args = line.split()
                edges[int(e_args[1])].append(int(e_args[2]))
                cost_1[(int(e_args[1]), int(e_args[2]))] = int(e_args[3])

    # second cost function
    with open(path_cost_2) as cost_2_file:
        for line in cost_2_file:
            if line.startswith("a"):
                e_args = line.split()
                cost_2[(int(e_args[1]), int(e_args[2]))] = int(e_args[3])

    assert n
    assert m

    return Map(name, edges, cost_1, cost_2, coords, n, m)


def draw_map(task_map: Map) -> None:
    coords = task_map._coords
    edges = task_map._edges
    fig, ax = plt.subplots()

    for state, (x, y) in tqdm(coords.items()):
        ax.plot(x, y, 'bo')
        ax.annotate(state, (x, y), xytext=(x, y + 0.1))

    for state, neighbors in tqdm(edges.items()):
        for neighbor in neighbors:
            x1, y1 = coords[state]
            x2, y2 = coords[neighbor]
            ax.plot([x1, x2], [y1, y2], 'k-')

    ax.set_xlim(min(x for x, _ in coords.values()) - 1,
                max(x for x, _ in coords.values()) + 1)
    ax.set_ylim(min(y for _, y in coords.values()) - 1,
                max(y for _, y in coords.values()) + 1)

    plt.show()


def compute_cost(
    task_map: Map,
    s1: int,
    s2: int,
    cost_num: int
) -> int:
    """
    Get the cost of a move between states s1 and s2
    by using cost_num cost function (1 for distance, 2 for time)

    Parameters
    ----------
    s1 : int
        Number of the first state

    s2 : int
        Number of the second state

    cos_num : int
        Which of the two cost functions to use

    Returns
    ----------
    int
        Cost of the move between cells using corresponding criterion
    """
    if cost_num == 1:
        return task_map.get_cost_1(s1, s2)
    elif cost_num == 2:
        return task_map.get_cost_2(s1, s2)
    else:
        raise ValueError(
            f'cost_num is expected to be 1 or 2, but got {cost_num}'
        )


@dataclass
class TestScen:
    """
    Represents a testing scenario

    Attributes
    ----------
    s_start : int
        Number of the start state

    s_goal : int
        Number of the goal state

    pareto_set: List[Tuple[int, int]]
        Set of pareto-optimal solutions,
        sorted in ascending order of the value of the first cost
        (and descending order of the value of the second cost)
    """
    s_start: int
    s_goal: int
    pareto_set: list


def read_scen_from_file(
    file_name: str
) -> TestScen:
    """
    Reads test scenario from txt file
    """

    s_start, s_goal = None, None
    pareto_set: List[Tuple[int, int]] = list()

    with open(file_name, 'r') as f:
        for line in f:
            args = line.split()
            # start and goal vertices
            if line.startswith('st'):
                s_start, s_goal = int(args[1]), int(args[2])
            # solutions
            if line.startswith('sol'):
                pareto_set.append((int(args[1]), int(args[2])))

    assert s_start
    assert s_goal
    assert pareto_set
    return TestScen(s_start, s_goal, sorted(pareto_set))
