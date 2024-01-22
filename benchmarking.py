import numpy as np

from typing import Callable, List, Optional, Tuple
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from tqdm import tqdm

from map_and_scenarios import Map


def plot_algorithm_result(
    res: list[Tuple[int, int]],
    map_name: str,
    alg_name: str,
) -> None:
    actual = np.array(sorted(res))
    x, y = actual[:, 0], actual[:, 1]
    plt.plot(x, y, color='pink')
    plt.scatter(x, y, marker='o', c='blue')
    plt.xlabel('distance')
    plt.ylabel('time')
    plt.title(f'Pareto-optimal curve for {alg_name} on {map_name} map')
    # plt.savefig(f'pareto_set_{map_name}_{alg_name}.png')
    plt.show()


class Tester:

    def __init__(
        self,
        number_of_problems: int,
        task_map: Map,
        seed: Optional[int] = None  # for reproducibility
    ):
        self.number_of_problems = number_of_problems
        self.task_map = task_map
        self.problems = None
        self.res_times = dict()
        self.res_iters = dict()
        self.seed = seed
        self.my_colors = iter([
            'tab:pink', 'tab:orange', 'tab:purple', 'tab:blue',
            'tab:pink', 'tab:orange', 'tab:purple', 'tab:blue'
        ])
        self.plot_dots = False

    def generate_problems(self) -> None:
        if self.seed:
            np.random.seed(self.seed)
        n = self.task_map._n
        self.problems = [
            (np.random.randint(n), np.random.randint(n))
            for _ in range(self.number_of_problems)
        ]

    def benchmarking(self, algorithms: List[Callable]) -> None:

        if not self.problems:
            self.generate_problems()
        assert self.problems

        for algorithm in algorithms:

            if algorithm.__name__ == 'boba_star':
                continue

            alg_times = []
            alg_iters = []

            for ind, problem in tqdm(enumerate(self.problems)):

                res = algorithm(self.task_map, problem[0], problem[1])
                alg_times.append([ind + 1, res.time_elapsed])
                alg_iters.append([ind + 1, res.iterations_count])

            self.res_times[algorithm.__name__] = alg_times
            self.res_iters[algorithm.__name__] = alg_iters

        if 'boba_star' in [alg.__name__ for alg in algorithms]:

            alg = [alg for alg in algorithms if alg.__name__ == 'boba_star'][0]

            alg_times_real = []
            alg_iters_real = []

            alg_times_emulated_parallel = []
            alg_iters_emulated_parallel = []

            for ind, problem in tqdm(enumerate(self.problems)):

                res = alg(self.task_map, problem[0], problem[1])

                alg_times_real.append([ind + 1, res.total_time_elapsed])
                alg_iters_real.append([ind + 1, res.total_iterations_count])

                alg_times_emulated_parallel.append(
                    [ind + 1, res.estimated_time_elapsed_in_parallel])
                alg_iters_emulated_parallel.append(
                    [ind + 1, res.estimated_iterations_count_in_parallel])

            self.res_times['boba_star_real'] = alg_times_real
            self.res_iters['boba_star_real'] = alg_iters_real

            self.res_times['boba_star_emulated_parallel'] = alg_times_emulated_parallel
            self.res_iters['boba_star_emulated_parallel'] = alg_iters_emulated_parallel

    def draw_results_for_times(self) -> None:

        map_name = self.task_map._name

        times_boa = np.array(self.res_times['boa_star'])[:, 1]
        indices = np.argsort(times_boa)

        for alg_name, res in self.res_times.items():
            self.plot_curve(res, alg_name, indices)

        plt.xlabel('problem')
        plt.ylabel('time elapsed (seconds)')
        plt.legend(loc='best')
        plt.title(f'{map_name} map, execution time')
        plt.savefig(f'results_{map_name}_times')
        plt.show()

    def draw_results_for_iters(self) -> None:

        map_name = self.task_map._name

        iters_boa = np.array(self.res_iters['boa_star'])[:, 1]
        sorting_indices = np.argsort(iters_boa)

        for alg_name, res in self.res_iters.items():
            self.plot_curve(res, alg_name, sorting_indices)

        plt.xlabel('problem')
        plt.ylabel('iterations count')
        plt.legend(loc='best')
        plt.title(f'{map_name} map, iterations count')
        plt.savefig(f'results_{map_name}_iterations')
        plt.show()

    def plot_curve(
        self,
        data: List[Tuple[int, int]],
        alg_name: str,
        sorting_indices: NDArray,
    ) -> None:
        actual = np.array(sorted(data))
        x_coords, y_coords = actual[:, 0], np.array(actual[:, 1])
        color = next(self.my_colors)
        plt.plot(x_coords, y_coords[sorting_indices],
                 label=alg_name, color=color)
        if self.plot_dots:
            plt.scatter(
                x_coords, y_coords[sorting_indices], marker='o', color=color)
        plt.fill_between(
            x_coords, y_coords[sorting_indices], 0, alpha=0.09, color=color)

    def test_algorithms(self, algorithms: List[Callable], plot_dots: bool = False) -> None:
        self.plot_dots = plot_dots
        self.generate_problems()
        self.benchmarking(algorithms)
        self.draw_results_for_times()
        self.draw_results_for_iters()
