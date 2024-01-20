from map_and_scenarios import read_map_from_file, read_scen_from_file
from boastar import boa_star
from bobastar import boba_star_real, boba_star_emulated_parallel, boa_star_enh
from test import test_algorithm

INF = int(1e12)


if __name__ == "__main__":

    # sample maps and scenrarios

    sample_maps = [
        read_map_from_file(
            f'data/samples/sample_{i}/dist.txt',
            f'data/samples/sample_{i}/time.txt',
            f'data/samples/sample_{i}/coord.txt'
        )
        for i in (1, 2, 3, 4, 5)
    ]

    sample_scens = [
        read_scen_from_file(
            f'data/samples/sample_{i}/scen.txt',
        )
        for i in (1, 2, 3, 4, 5)
    ]

    algorithms = [
        boa_star, 
        boa_star_enh, 
        boba_star_real, 
        boba_star_emulated_parallel
    ]

    for algorithm in algorithms:
        test_algorithm(
            algorithm,
            5,
            sample_maps,
            sample_scens
        )