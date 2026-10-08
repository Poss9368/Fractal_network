from pathlib import Path
import csv
import argparse
from multiprocessing import Pool

import numpy as np
try:
    from .dynamic import (conjudate_gradient, fire_minimize,
                          maximum_length_angle, rest_deformations_from_thetas,
                          structure_size)
except ImportError:
    from dynamic import (conjudate_gradient, fire_minimize,
                         maximum_length_angle, rest_deformations_from_thetas,
                         structure_size)


PATH = Path(__file__).resolve().parent
RESULTS_PATH = PATH / 'results'
N_SEEDS = 1
BASE_SEED = 45
REST_FRACTION = 0.5


def simulate_seed(args):
    seed, n, w_i, lambda_values, solver = args[:5]
    fire_options = args[5] if len(args) > 5 else {}
    state_directory = Path(args[6]) if len(args) > 6 else None
    minimize = fire_minimize if solver == "fire" else conjudate_gradient

    theta_star = maximum_length_angle(n)
    thetas = np.full(n, REST_FRACTION * theta_star)
    rest_deformations = rest_deformations_from_thetas(thetas)

    # Configuraciones deterministas útiles:
    family_weights = np.ones((n, 2))  # ambas familias completas
    # family_weights = np.column_stack((np.ones(n), np.zeros(n)))  # sólo phi-
    # family_weights = np.column_stack((np.zeros(n), np.ones(n)))  # sólo phi+
    results = []
    correction = np.zeros(n)
    precise_states, precise_corrections, diagnostics = [], [], []
    if solver == "fire":
        initial = fire_minimize(thetas,w_i,family_weights,rest_deformations,0.,
                                return_info=True,**fire_options)
        thetas, correction = initial.thetas, initial.theta_correction
        precise_states.append(thetas.copy())
        precise_corrections.append(correction.copy())
        diagnostics.append((0.,initial.gradient_max,initial.rounded_gradient_max,
                            initial.tolerance,initial.iterations))

    geometric_max_length = structure_size(np.full(n, theta_star))[0]

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
        if solver == "fire":
            result = fire_minimize(
                thetas,w_i,family_weights,rest_deformations,lambda_restriction,
                theta_correction=correction,return_info=True,**fire_options,
            )
            thetas, correction = result.thetas, result.theta_correction
            precise_states.append(thetas.copy())
            precise_corrections.append(correction.copy())
            diagnostics.append((lambda_restriction,result.gradient_max,
                                result.rounded_gradient_max,result.tolerance,
                                result.iterations))
        else:
            thetas = minimize(thetas,w_i,family_weights,rest_deformations,
                              lambda_restriction)
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

    if solver == "fire" and state_directory is not None:
        state_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            state_directory / f"{n}_seed{seed}_fire_states.npz",
            theta_hi=np.asarray(precise_states), theta_lo=np.asarray(precise_corrections),
            diagnostics=np.asarray(diagnostics),
            diagnostic_columns=np.array(["force", "gradient_max", "rounded_gradient_max",
                                         "tolerance", "iterations"]),
            rest_deformations=rest_deformations, weights=w_i, family_weights=family_weights,
        )
    print(f"Seed {seed} completed.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Respuesta a fuerza impuesta con continuación")
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 16, 64, 256, 1024])
    parser.add_argument("--solver", choices=["fire", "cg"], default="fire")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_PATH)
    parser.add_argument("--ftol", type=float, default=1e-12,
                        help="Tolerancia absoluta del gradiente máximo para FIRE")
    parser.add_argument("--fire-dt", type=float, default=0.1)
    parser.add_argument("--fire-dt-max", type=float, default=0.1)
    parser.add_argument("--fire-max-steps", type=int, default=50000)
    options = parser.parse_args()
    if (not np.all(np.isfinite([options.ftol, options.fire_dt, options.fire_dt_max]))
            or options.ftol <= 0 or not 0 < options.fire_dt <= options.fire_dt_max
            or options.fire_max_steps < 1):
        parser.error("FIRE requiere ftol > 0, 0 < dt <= dt_max y max_steps > 0")
    fire_options = dict(ftol=options.ftol, rtol=0., dt=options.fire_dt,
                        dt_max=options.fire_dt_max, max_steps=options.fire_max_steps)
    if any(n < 1 for n in options.sizes) or options.processes < 1:
        parser.error("Los tamaños y el número de procesos deben ser positivos")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for n in options.sizes:

        x_min = -5
        x_max = 6
        number_of_points = int((x_max - x_min)*3 + 1)
        lambda_values = np.logspace(x_min, x_max, number_of_points)

        # w_i es el peso elástico efectivo completo del nivel:
        #     w_i = 4^(N-i) k_i.
        # Para el caso uniforme usamos directamente w_i = 1 y evitamos
        # calcular por separado potencias enormes y números diminutos.
        w_i = np.ones(n, dtype=float)

        tasks = [
            (seed, n, w_i, lambda_values, options.solver, fire_options, options.output_dir)
            for seed in range(BASE_SEED, BASE_SEED + N_SEEDS)
        ]

        process_count = min(options.processes, len(tasks))
        if process_count == 1:
            results_by_seed = [simulate_seed(task) for task in tasks]
        else:
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
        samples_path = options.output_dir / f"{n}_angles_samples.csv"
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
        summary_path = options.output_dir / f"{n}_angles_output.csv"
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
