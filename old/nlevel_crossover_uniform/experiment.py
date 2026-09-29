"""All phi+, uniform effective stiffness, uniform nonzero stress-free angles.

Exact trigonometric length; no small-angle approximation in the solver.
Run: python3 src/nlevel_crossover_uniform/experiment.py
"""

from pathlib import Path
import argparse
import csv
import json
import os
import tempfile

import numpy as np
from scipy.linalg import solve_banded


def difference(x):
    return np.r_[x[0], np.diff(x)]


def transpose_difference(x):
    return np.r_[x[:-1] - x[1:], x[-1]]


def theta_star(n):
    return 2 * np.arctan(2 / (1 + np.sqrt(4 * n - 3)))


def geometry(theta):
    t = np.tan(theta / 2)
    cosines = np.cos(theta / 2)
    r = 1 + t.sum()
    if np.any(cosines <= 0) or r <= 0:
        raise ValueError("Outside positive-length branch")
    # log1p avoids losing the O(theta**2) contribution at very large N.
    log_c = np.log1p(-2 * np.sin(theta / 4) ** 2).sum()
    c = np.exp(log_c)
    log_x = log_c + np.log(r)
    x = np.exp(log_x)
    v = 1 + t * t
    grad = 0.5 * c * (v - r * t)
    # Hessian_theta X = diag(d) + U @ C @ U.T.
    d = 0.25 * c * v * (2 * t - r)
    u = np.column_stack((t, v))
    small = 0.25 * c * np.array([[r, -1.0], [-1.0, 0.0]])
    return log_x, x, grad, d, u, small


def hessian_solve(rhs, weights, force, d, u, small):
    diagonal = 2 * weights - force * (d + np.r_[d[1:], 0.0])
    off_diagonal = force * d[1:]
    bands = np.zeros((3, len(weights)))
    bands[0, 1:] = off_diagonal
    bands[1] = diagonal
    bands[2, :-1] = off_diagonal
    du = np.column_stack([transpose_difference(u[:, j]) for j in range(2)])
    solved = solve_banded((1, 1), bands, np.column_stack((rhs, du)))
    z, v = solved[:, 0], solved[:, 1:]
    correction = -force * small
    return z - v @ np.linalg.solve(
        np.eye(2) + correction @ (du.T @ v), correction @ (du.T @ z)
    )


def equilibrium(rest, weights, force, initial, log_x0):
    s = initial.copy()  # Cumulative angular DEVIATIONS from the natural state.
    for iteration in range(100):
        theta = rest + difference(s)
        log_x, x, gx, d, u, small = geometry(theta)
        gradient = 2 * weights * s - force * transpose_difference(gx)
        scale = max(force * np.max(np.abs(gx)), np.max(np.abs(2 * weights * s)), 1e-20)
        residual = np.max(np.abs(gradient))
        if residual <= 2e-13 + 2e-10 * scale:
            return s, log_x, iteration, residual / max(scale, 1e-20)
        step = -hessian_solve(gradient, weights, force, d, u, small)
        slope = gradient @ step
        if slope >= 0:
            raise RuntimeError("Newton Hessian is not positive along the proposed step")
        value = np.dot(weights, s * s) - force * np.exp(log_x0) * np.expm1(log_x - log_x0)
        fraction = 1.0
        for _ in range(45):
            candidate = s + fraction * step
            try:
                new_log_x = geometry(rest + difference(candidate))[0]
                trial = np.dot(weights, candidate * candidate) - force * np.exp(log_x0) * np.expm1(new_log_x - log_x0)
            except ValueError:
                trial = np.inf
            roundoff = 8 * np.finfo(float).eps * max(abs(value), force * x, 1e-30)
            if trial <= value + 1e-4 * fraction * slope + roundoff:
                break
            fraction *= 0.5
        else:
            raise RuntimeError("Line search did not converge")
        s = candidate
    raise RuntimeError(f"Newton did not converge: N={len(rest)}, f={force}, residual={residual}")


def verify_derivatives():
    rng = np.random.default_rng(27)
    n = 7
    theta = rng.uniform(0.02, 0.2, n)
    weights = rng.uniform(0.5, 2, n)
    force = 0.7
    _, _, grad, d, u, small = geometry(theta)
    h = 1e-5
    numerical = np.column_stack([
        (geometry(theta + h * np.eye(n)[j])[2] - geometry(theta - h * np.eye(n)[j])[2]) / (2 * h)
        for j in range(n)
    ])
    exact = np.diag(d) + u @ small @ u.T
    np.testing.assert_allclose(exact, numerical, atol=1e-9, rtol=1e-8)
    D = np.eye(n) - np.eye(n, k=-1)
    dense = 2 * np.diag(weights) - force * D.T @ exact @ D
    rhs = rng.normal(size=n)
    np.testing.assert_allclose(hessian_solve(rhs, weights, force, d, u, small), np.linalg.solve(dense, rhs), atol=2e-12)
    numerical_grad = np.array([
        (geometry(theta + h * np.eye(n)[j])[1] - geometry(theta - h * np.eye(n)[j])[1]) / (2 * h)
        for j in range(n)
    ])
    np.testing.assert_allclose(grad, numerical_grad, atol=1e-9)
    return "Analytic gradient, Hessian and structured Newton solve checked against finite differences/dense solve."


def run_case(n, c, points):
    rest = np.full(n, c * theta_star(n))
    weights = np.ones(n)
    log_x0, x0, g0, *_ = geometry(rest)
    x_max = geometry(np.full(n, theta_star(n)))[1]
    compliance = g0[-1] ** 2 / (2 * x0)
    d_n = 2 * np.sqrt(n) * g0[-1] / x0
    q_values = np.logspace(-5, np.log10(0.2 * n * n), points)
    forces = q_values / x0
    s = np.zeros(n)
    rows = []
    for force in forces:
        s, log_x, it, residual = equilibrium(rest, weights, force, s, log_x0)
        strain = np.expm1(log_x - log_x0)
        rows.append([n, c, force, force / n**1.5, strain, compliance * force, it, residual])
    data = np.asarray(rows)
    strains = data[:, 4]
    exponent = np.gradient(np.log(forces), np.log(strains))
    ratio = strains / (compliance * forces)
    crossing = np.flatnonzero(ratio <= 0.9)
    j = int(crossing[0])
    if j == 0:
        raise RuntimeError("Force grid does not resolve the Hookean regime")
    t = (0.9 - ratio[j-1]) / (ratio[j] - ratio[j-1])
    cross_e = np.exp((1-t) * np.log(strains[j-1]) + t * np.log(strains[j]))
    cross_f = np.exp((1-t) * np.log(forces[j-1]) + t * np.log(forces[j]))
    # Mid-band probe: q=N means penetration depth sqrt(N), away from both cutoffs.
    mid_slope = np.interp(np.log(n), np.log(q_values), exponent)
    summary = dict(n=n, c=c, x0=x0, epsilon_max=x_max/x0-1,
                   compliance=compliance, crossover_strain=cross_e,
                   crossover_force=cross_f, n_epsilon_cross=n*cross_e,
                   midband_exponent=mid_slope,
                   max_relative_gradient=float(data[:, 7].max()),
                   first_linear_ratio=float(ratio[0]))
    s_values = q_values / 8
    r_values = 2 * s_values / (1 + 2 * s_values + np.sqrt(1 + 4 * s_values))
    boundary_prediction = d_n**2 / (2*n) * r_values * (2+r_values) / (1-r_values**2)
    probe = np.argmin(abs(np.log(s_values)))
    summary["exact_over_boundary_at_s1"] = float(strains[probe] / boundary_prediction[probe])
    return np.column_stack((data, exponent, boundary_prediction)), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024, 4096, 16384])
    parser.add_argument("--c", type=float, default=0.8)
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if not 0 < args.c < 1:
        parser.error("c must be strictly between 0 and 1")
    if len(args.sizes) < 2 or any(n < 2 for n in args.sizes):
        parser.error("provide at least two sizes, each >= 2")
    print(verify_derivatives(), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    curves, summaries = [], []
    for n in args.sizes:
        curve, summary = run_case(n, args.c, args.points)
        curves.append(curve)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    np.savetxt(args.output / "curves.csv", np.vstack(curves), delimiter=",", comments="",
               header="N,c,force,force_over_N_1p5,strain,hookean_strain,newton_iterations,relative_gradient,local_exponent,boundary_prediction")
    with (args.output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    fit = np.polyfit(np.log([s["n"] for s in summaries[-4:]]), np.log([s["crossover_strain"] for s in summaries[-4:]]), 1)
    print(f"Last {min(4, len(summaries))} sizes: epsilon_cross proportional to N^({fit[0]:.6f})", flush=True)
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="hierarchical-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    for data in curves:
        n = int(data[0, 0])
        ax[0].loglog(data[:, 4], data[:, 3], label=f"N={n}")
        ax[1].semilogx(data[:, 4], data[:, 8])
    ax[0].set(xlabel=r"Strain $\epsilon=(X-X_0)/X_0$", ylabel=r"Fuerza $f/N^{3/2}$", title=rf"Todos $\phi^+$; $\theta_0={args.c}\theta_*$")
    ax[0].legend(fontsize=8, ncol=2)
    ax[1].axhline(1, color="0.5", ls=":")
    ax[1].axhline(2, color="0.2", ls="--")
    ax[1].set(xlabel=r"Strain $\epsilon$", ylabel=r"$d\log f/d\log\epsilon$", ylim=(0.85, 3.1), title="Pendiente local")
    ns = np.array([s["n"] for s in summaries])
    es = np.array([s["crossover_strain"] for s in summaries])
    ax[2].loglog(ns, es, "o-", label="Geometría exacta")
    ax[2].loglog(ns, es[-1]*ns[-1]/ns, "k--", label=r"$1/N$")
    ax[2].set(xlabel="Número de niveles N", ylabel=r"$\epsilon_\times$ (desviación 10 %)", title=rf"Ajuste: $N^{{{fit[0]:.3f}}}$")
    ax[2].legend(fontsize=9)
    for axis in ax:
        axis.grid(alpha=0.15, which="both")
    xmin = min(es) / 30
    ax[0].set_xlim(left=xmin)
    ax[1].set_xlim(left=xmin)
    fig.savefig(args.output / "crossover_uniform.png", dpi=170)
    fig.savefig(args.output / "crossover_uniform.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
