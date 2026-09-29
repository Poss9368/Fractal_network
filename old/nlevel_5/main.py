from pathlib import Path
import csv
from multiprocessing import Pool

import numpy as np
from old.nlevel.dynamic import (
    balanced_random_signs,
    conjudate_gradient,
    maximum_length_angle,
    rest_deformations_from_thetas,
    rest_thetas_from_noise,
    structure_size,
)

PATH = Path(__file__).resolve().parent
RESULTS_PATH = PATH / 'results'
N_SEEDS = 8
BASE_SEED = 45
NOISE_FRACTION = 0.5


def simulate_seed(args):
    seed, n, k_i, lambda_values = args

    rng = np.random.RandomState(seed)
    signs = np.ones(n) #balanced_random_signs(n, rng)
    noise_amplitude = NOISE_FRACTION * maximum_length_angle(n)
    noise = noise_amplitude * signs
    thetas = rest_thetas_from_noise(noise)

    # Configuración elástica por nivel:
    # family_weights[i] = [peso_phi_minus, peso_phi_plus].
    # Los dos pesos son independientes y pueden coexistir.
    # family_weights = rng.random_sample((n, 2))

    # Ambos valores naturales se derivan aquí de la misma geometría theta^(0),
    # por lo que ``thetas`` tiene energía exactamente cero a fuerza nula.
    # También se puede construir esta matriz directamente para introducir
    # equilibrios incompatibles y estudiar frustración elástica.
    rest_deformations = rest_deformations_from_thetas(thetas)

    # Configuraciones deterministas útiles:
    # family_weights = np.ones((n, 2))  # ambas familias completas
    #family_weights = np.column_stack((np.ones(n), np.zeros(n)))  # sólo phi-
    family_weights = np.column_stack((np.zeros(n), np.ones(n)))  # sólo phi+
    results = []

    theta_star = np.full(n, maximum_length_angle(n))
    geometric_max_length = structure_size(theta_star)[0]

    length_x, length_x_min, length_y_max, length_y_min = structure_size(thetas)
    rest_length = length_x
    available_extension = geometric_max_length - rest_length
    length_tolerance = 100.0 * np.finfo(float).eps * max(1.0, geometric_max_length)
    if available_extension < -length_tolerance:
        raise RuntimeError("La configuración de reposo resultó más larga que theta_star")

    area = length_x * length_y_max
    results.append((
        0.0, 0.0, 0.0, 0.0,
        length_x, length_x_min, length_y_max, length_y_min, area,
        rest_length, geometric_max_length,
    ))

    for lambda_restriction in lambda_values:
        thetas = conjudate_gradient(
            thetas, k_i, family_weights, rest_deformations,
            lambda_restriction,
        )
        length_x, length_x_min, length_y_max, length_y_min = structure_size(thetas)
        extension_x = length_x - rest_length
        strain_x = extension_x / rest_length
        if available_extension > length_tolerance:
            straightening_fraction = extension_x / available_extension
        else:
            straightening_fraction = 0.0
        area = length_x * length_y_max

        results.append((
            lambda_restriction, strain_x, extension_x,
            straightening_fraction, length_x, length_x_min,
            length_y_max, length_y_min, area, rest_length,
            geometric_max_length,
        ))

    print(f"Seed {seed} completed.")
    return results


if __name__ == "__main__":
    for level_power in range(4, 7):
        n = 2 ** level_power

        # Rango preliminar de fuerza. Para medir el crossover hookeano a N
        # grande habrá que extender el extremo inferior y usar un solucionador
        # precondicionado: los pesos abarcan desde 4^-1 hasta 4^-N.
        x_min = -5
        x_max = 4
        number_of_points = int((x_max - x_min)*4 + 1)
        lambda_values = np.logspace(x_min, x_max, number_of_points)

        # Con i matemático = 1,...,N, el peso elástico es
        #     w_i = 4^(N-i) k_i.
        # Elegir la misma rigidez microscópica k_i=4^(-N) en todos los
        # niveles produce exactamente w_i=4^(-i), de modo que
        # El prefactor geométrico por familia es 4^(N-i) k_i. Los pesos
        # específicos de phi- y phi+ se guardan en family_weights.
        
        #microscopic_stiffness = np.exp2(-2.0 * n)  # 4**(-N)
        #k_i = np.full(n, microscopic_stiffness, dtype=float)

        k_i = []
        for i in range(n):
            k_i.append(((4**(n-(i+1)))**-1))

        tasks = [
            (seed, n, k_i, lambda_values)
            for seed in range(BASE_SEED, BASE_SEED + N_SEEDS)
        ]

        process_count = 8
        with Pool(processes=process_count) as pool:
            results_by_seed = pool.map(simulate_seed, tasks)

        samples = np.asarray(results_by_seed, dtype=float)
        seeds = np.arange(BASE_SEED, BASE_SEED + N_SEEDS)
        sample_header = (
            "seed", "lambda", "strain_engineering", "extension_x",
            "straightening_fraction", "length_x", "length_x_min",
            "length_y_max", "length_y_min", "area", "rest_length",
            "geometric_max_length",
        )

        # Guardamos las realizaciones individuales para no perder la
        # distribución al escoger posteriormente otra forma de promediar.
        samples_path = RESULTS_PATH / f"{n}_angles_samples.csv"
        with open(samples_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(sample_header)
            for seed, seed_results in zip(seeds, samples):
                for row in seed_results:
                    writer.writerow((seed, *row))

        lambdas = samples[0, :, 0]
        if not np.allclose(samples[:, :, 0], lambdas[None, :], rtol=0.0, atol=0.0):
            raise RuntimeError("Las realizaciones no comparten la misma grilla de fuerza")

        strain = samples[:, :, 1]
        straightening_fraction = samples[:, :, 3]
        length_x = samples[:, :, 4]
        area = samples[:, :, 8]
        rest_lengths = samples[:, 0, 9]
        geometric_max_length = samples[0, 0, 10]
        sample_count = samples.shape[0]
        ddof = 1 if sample_count > 1 else 0

        summary_header = (
            "lambda", "strain_mean", "strain_median", "strain_std",
            "strain_sem", "strain_q25", "strain_q75",
            "straightening_fraction_mean", "straightening_fraction_median",
            "length_x_mean", "length_x_std", "area_mean",
            "rest_length_mean", "rest_length_std", "geometric_max_length",
        )
        summary_path = RESULTS_PATH / f"{n}_angles_output.csv"
        with open(summary_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(summary_header)
            for point in range(samples.shape[1]):
                writer.writerow((
                    lambdas[point],
                    np.mean(strain[:, point]),
                    np.median(strain[:, point]),
                    np.std(strain[:, point], ddof=ddof),
                    np.std(strain[:, point], ddof=ddof) / np.sqrt(sample_count),
                    np.quantile(strain[:, point], 0.25),
                    np.quantile(strain[:, point], 0.75),
                    np.mean(straightening_fraction[:, point]),
                    np.median(straightening_fraction[:, point]),
                    np.mean(length_x[:, point]),
                    np.std(length_x[:, point], ddof=ddof),
                    np.mean(area[:, point]),
                    np.mean(rest_lengths),
                    np.std(rest_lengths, ddof=ddof),
                    geometric_max_length,
                ))
