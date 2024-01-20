import time

from typing import Callable, List

from map_and_scenarios import Map, TestScen


def test_algorithm(
    algorithm: Callable,
    n_tests: int,
    task_maps: List[Map],
    test_scens: List[TestScen]
):
    assert len(task_maps) == len(test_scens) == n_tests
    correct = 0

    for ind, (task_map, scen) in enumerate(zip(task_maps, test_scens)):
        print(f'test instance #{ind + 1}')
        print(f'n = {task_map._n}, m = {task_map._m}')

        res = None
        error = None
        start_time = time.time()
        try:
            res = algorithm(task_map, scen.s_start, scen.s_goal).pareto_set
        except Exception as e:
            error = e
        elapsed_time = time.time() - start_time

        if res:
            actual = sorted(res)
            if actual == scen.pareto_set:
                correct += 1
                print(f'correct: true')
                print(f'result: {actual}')
            else:
                print(f'correct: false')
                print(f'expected: {scen.pareto_set},')
                print(f'but got: {actual}')
        else:
            print(f'test execution finished with an error: {error}')

        print(f'time elapsed: {elapsed_time:.06f}')
        print()

    print(f'total count of correct tests: {correct} out of {n_tests}')
    if correct == n_tests:
        print('✨🌟💖💎🦄💎💖🌟✨🌟💖💎🦄💎💖🌟✨')
