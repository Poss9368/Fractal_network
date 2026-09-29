"""Force sweeps for the reduced hierarchy with elastic phi-minus hinges only.

Run from the repository root with either::

    python3 -m src.nlevel_phi_minus.main --levels 4 8 12
    python3 src/nlevel_phi_minus/main.py --levels 4 --points 21

Each invocation creates a new results directory; existing results are never
overwritten.  Forces are conjugate to the dimensionless horizontal length X.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .model import PhiMinusModel
    from .plot_results import generate_plots, local_exponent
    from .solvers import solve_nonlinear, solve_quadratic
else:
    from old.nlevel_phi_minus.model import PhiMinusModel
    from old.nlevel_phi_minus.plot_results import generate_plots, local_exponent
    from old.nlevel_phi_minus.solvers import solve_nonlinear, solve_quadratic


CURVE_FIELDS = [
    "N", "s", "k_scale", "method", "force", "epsilon", "epsilon_geometry",
    "epsilon_quadratic", "x_max", "x_min", "y_max", "y_min", "energy",
    "max_abs_theta", "small_angle", "geometry_relative_error", "converged",
    "stable", "iterations", "residual_relative", "hessian_min_scaled",
    "beta_local", "message",
]
ANGLE_FIELDS = [
    "N", "s", "k_scale", "method", "force", "level", "theta", "alpha",
    "phi_minus", "S", "energy", "converged", "stable",
]
SPECTRUM_FIELDS = ["N", "s", "k_scale", "mode", "b", "modal_weight"]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[4, 8, 12],
                        help="Numbers of hierarchical levels (default: 4 8 12).")
    parser.add_argument("--exponents", nargs="+", type=float, default=[1.0],
                        help="s in k_i=k_scale*4**(s*(i-N)); default: 1.")
    parser.add_argument("--k-scale", type=float, default=1.0)
    parser.add_argument("--method", choices=["both", "quadratic", "nonlinear"],
                        default="both")
    parser.add_argument("--points", type=int, default=81,
                        help="Positive logarithmic force samples, plus f=0.")
    parser.add_argument("--f-min", type=float, default=None,
                        help="Lowest positive force; default: 1e-3*b_min per model.")
    parser.add_argument("--f-max", type=float, default=None,
                        help="Highest force; default: b_max per model.")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="Relative nonlinear equilibrium tolerance.")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None,
                        help="New output directory. An existing path is rejected.")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def force_grid(model: PhiMinusModel, args: argparse.Namespace) -> np.ndarray:
    f_min = 1e-3 * float(model.b[0]) if args.f_min is None else args.f_min
    f_max = float(model.b[-1]) if args.f_max is None else args.f_max
    if not (np.isfinite(f_min) and np.isfinite(f_max) and 0 < f_min < f_max):
        raise ValueError("The force bounds must be finite and satisfy 0 < f_min < f_max.")
    return np.concatenate(([0.0], np.geomspace(f_min, f_max, args.points)))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def measure_solution(model: PhiMinusModel, method: str, force: float, solution):
    theta = np.asarray(solution.theta, dtype=float)
    x_max, x_min, y_max, y_min = model.structure_size(theta)
    geometry_epsilon = float(model.strain(theta))
    quadratic_epsilon = float(model.quadratic_strain(theta))
    diagnostics = model.diagnostics(theta)
    energies = np.asarray(model.energy_by_level(theta), dtype=float)
    hinge_angles = np.asarray(model.hinge_angles(theta), dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(theta[:-1])))
    row = {
        "N": model.levels, "s": model.exponent, "k_scale": model.k_scale,
        "method": method, "force": float(force),
        "epsilon": quadratic_epsilon if method == "quadratic" else geometry_epsilon,
        "epsilon_geometry": geometry_epsilon, "epsilon_quadratic": quadratic_epsilon,
        "x_max": float(x_max), "x_min": float(x_min), "y_max": float(y_max),
        "y_min": float(y_min), "energy": float(np.sum(energies)),
        "max_abs_theta": float(diagnostics["max_abs_theta"]),
        "small_angle": bool(diagnostics["small_angle"]),
        "geometry_relative_error": float(diagnostics["geometry_relative_error"]),
        "converged": bool(solution.converged), "stable": bool(solution.stable),
        "iterations": int(solution.iterations),
        "residual_relative": float(solution.residual_relative),
        "hessian_min_scaled": float(solution.hessian_min_scaled),
        "beta_local": float("nan"), "message": str(solution.message),
    }
    angles = [
        {"N": model.levels, "s": model.exponent, "k_scale": model.k_scale,
         "method": method, "force": float(force), "level": i + 1,
         "theta": float(angle), "alpha": float(1 - angle / 2),
         "phi_minus": float(hinge_angles[i]), "S": float(cumulative[i]),
         "energy": float(energies[i]), "converged": bool(solution.converged),
         "stable": bool(solution.stable)}
        for i, angle in enumerate(theta)
    ]
    return row, angles


def failed_row(model: PhiMinusModel, method: str, force: float, error: Exception):
    row = {field: float("nan") for field in CURVE_FIELDS}
    row.update(N=model.levels, s=model.exponent, k_scale=model.k_scale,
               method=method, force=float(force), converged=False, stable=False,
               small_angle=False, iterations=0, message=f"{type(error).__name__}: {error}")
    return row


def run(args: argparse.Namespace) -> int:
    if args.points < 3:
        raise ValueError("Use at least three positive force samples.")
    if args.maxiter < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        raise ValueError("maxiter and tol must be positive.")
    # Preserve order while avoiding duplicate curves in an invocation.
    levels = list(dict.fromkeys(args.levels))
    exponents = list(dict.fromkeys(args.exponents))
    models = [PhiMinusModel(n, exponent=s, k_scale=args.k_scale)
              for n in levels for s in exponents]
    grids = [force_grid(model, args) for model in models]
    methods = ["quadratic", "nonlinear"] if args.method == "both" else [args.method]
    created = datetime.now(timezone.utc)
    run_dir = args.output
    if run_dir is None:
        run_dir = Path(__file__).resolve().parent / "results" / created.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    curves, angle_rows, spectrum_rows, failures, model_metadata = [], [], [], [], []
    for model, forces in zip(models, grids):
        model_metadata.append({
            "N": model.levels, "s": model.exponent, "k_scale": model.k_scale,
            "k_i": np.asarray(model.k_i).tolist(), "q_i": np.asarray(model.q).tolist(),
            "b_min": float(model.b[0]), "b_max": float(model.b[-1]),
            "condition_number": float(model.condition_number),
            "force_min_positive": float(forces[1]), "force_max": float(forces[-1]),
        })
        spectrum_rows.extend(
            {"N": model.levels, "s": model.exponent, "k_scale": model.k_scale,
             "mode": p + 1, "b": float(b), "modal_weight": float(weight)}
            for p, (b, weight) in enumerate(zip(model.b, model.modal_weights))
        )
        for method in methods:
            rows = []
            initial = np.zeros(model.levels)
            for force in forces:
                try:
                    if method == "quadratic":
                        solution = solve_quadratic(model, float(force))
                    else:
                        solution = solve_nonlinear(model, float(force), initial=initial,
                                                   tol=args.tol, maxiter=args.maxiter)
                    row, new_angles = measure_solution(model, method, float(force), solution)
                    angle_rows.extend(new_angles)
                    usable = (row["converged"] and row["stable"]
                              and np.isfinite(row["epsilon"])
                              and np.all(np.isfinite(solution.theta)))
                    if usable and method == "nonlinear":
                        initial = np.asarray(solution.theta).copy()
                except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as error:
                    row = failed_row(model, method, float(force), error)
                    usable = False
                rows.append(row)
                if not usable:
                    failure = {"N": model.levels, "s": model.exponent, "method": method,
                               "force": float(force), "message": row["message"]}
                    failures.append(failure)
                    print(f"Curve stopped: N={model.levels}, s={model.exponent:g}, "
                          f"{method}, f={force:.6g}: {row['message']}", file=sys.stderr)
                    break
            beta = local_exponent(rows)
            for row, value in zip(rows, beta):
                row["beta_local"] = float(value)
            curves.extend(rows)
            accepted = sum(bool(r["converged"] and r["stable"]) for r in rows)
            print(f"N={model.levels}, s={model.exponent:g}, {method}: "
                  f"{accepted}/{len(forces)} stable equilibria")

    write_csv(run_dir / "curves.csv", CURVE_FIELDS, curves)
    write_csv(run_dir / "angles.csv", ANGLE_FIELDS, angle_rows)
    write_csv(run_dir / "spectrum.csv", SPECTRUM_FIELDS, spectrum_rows)
    metadata = {
        "created_utc": created.isoformat(), "status": "incomplete" if failures else "complete",
        "model": "phi-minus only, delta=0, one angular degree of freedom per level",
        "loading": "constant horizontal force conjugate to normalized X; f=F*D",
        "geometry": "four-length recursive coordinate; no contact or grip-node verification",
        "nonlinear_protocol": "local stable-equilibrium continuation from theta=0; not a global-minimum proof",
        "strain_reference": "X(0)=1; epsilon=X-1",
        "small_angle_threshold_rad": 0.2,
        "beta_definition": "d log(f) / d log(epsilon); finite differences only on positive, stable, converged contiguous samples; endpoints omitted",
        "methods": methods, "positive_force_points": args.points,
        "tol": args.tol, "maxiter": args.maxiter,
        "models": model_metadata, "failures": failures,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    with (run_dir / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Results: {run_dir}")
    if not args.no_plots:
        generate_plots(run_dir)
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
