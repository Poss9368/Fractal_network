from pathlib import Path
import os
from multiprocessing import Pool

import numpy as np
from dynamic import conjudate_gradient, structure_size

PATH = Path(__file__).resolve().parent
RESULTS_PATH = PATH / 'results'
N_SEEDS = 32
BASE_SEED = 45


def simulate_seed(args):
    seed, n, k_i, lambda_values = args

    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 2, size=n)
    ## last noise is always 0, to avoid the last angle to be fixed
    noise[:] = 1
    thetas = np.zeros(n)
    results = []

    structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(thetas)
    area = structure_size_x_max * structure_size_y_max
    results.append((0, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, area))

    for lambda_restriction in lambda_values:
        thetas = conjudate_gradient(thetas, k_i, noise, lambda_restriction)
        structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(thetas)
        area = structure_size_x_max * structure_size_y_max

        results.append((lambda_restriction, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, area))

    print(f"Seed {seed} completed.")
    return results


if __name__ == "__main__":
    for i in range(4, 8):
        # Ángulos iniciales del sistema
        n = 2 ** i
        thetas = np.zeros(n)  # Inicializar con ceros

        # Rango de valores para lambda_restriction
        x_min = -3
        x_max = 4
        bin = int((x_max - x_min)*3 + 1) 
        lambda_values = np.logspace(x_min, x_max, bin)  # Valores de lambda en escala logarítmica

        k_i = []
        for i in range(n):
            k_i.append(((2*4**(n-(i+1)))**-1))

        tasks = [
            (seed, n, k_i, lambda_values)
            for seed in range(BASE_SEED, BASE_SEED + N_SEEDS)
        ]

        process_count = 8
        with Pool(processes=process_count) as pool:
            results_by_seed = pool.map(simulate_seed, tasks)

        # Promediar cada columna para cada valor de lambda entre todas las semillas.
        results = np.mean(np.asarray(results_by_seed, dtype=float), axis=0)

        # Imprimir los resultadospython main.py en .csv y con cabecera
        file_name = f"{n}_angles_output.csv"
        with open(RESULTS_PATH / file_name, 'w') as f:
            f.write("lambda,structure_size_x_max,structure_size_x_min,structure_size_y_max,structure_size_y_min,area\n")
            for result in results:
                f.write(f"{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]}\n")


