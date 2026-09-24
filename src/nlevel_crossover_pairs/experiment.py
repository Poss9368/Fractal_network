"""Deterministic phi+ hierarchy with pairwise rest corrugations.

Run: python3 src/nlevel_crossover_pairs/experiment.py
Uses exact geometric length and unconstrained even cumulative angles.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="hierarchical-pairs-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded


def difference(s):
    return np.diff(s, prepend=0.0)


def transpose_difference(x):
    out = x.copy()
    out[:-1] -= x[1:]
    return out


def geometry(s, rest_theta=None):
    theta = difference(s)
    if rest_theta is not None:
        theta = rest_theta + theta
    z = np.tan(theta / 2)
    q = 1 + z*z
    logp = np.log1p(-2*np.sin(theta / 4)**2).sum()
    product = np.exp(logp)
    total = 1 + z.sum()
    length = product * total
    gradient = transpose_difference(product / 2 * (q - z * total))
    diagonal = q * (2*z - total)
    lowrank = np.column_stack((transpose_difference(z), transpose_difference(q)))
    core = np.array([[total, -1.0], [-1.0, 0.0]])
    return length, gradient, diagonal, lowrank, core, product, logp, total


def solve_hessian(a, force, geo, rhs):
    _, _, hdiag, u, core, product, _, _ = geo
    factor = -force * product / 4
    diagonal = hdiag.copy()
    diagonal[:-1] += hdiag[1:]
    bands = np.zeros((3, len(a)))
    bands[1] = 2*a + factor*diagonal
    bands[0, 1:] = -factor*hdiag[1:]
    bands[2, :-1] = -factor*hdiag[1:]
    solutions = solve_banded((1, 1), bands, np.column_stack((rhs, u)))
    base, response = solutions[:, 0], solutions[:, 1:]
    c = factor * core
    correction = np.linalg.solve(np.eye(2) + c @ (u.T @ response), c @ (u.T @ base))
    return base - response @ correction


def solve_equilibrium(a, rest_theta, force, initial, rest_geo):
    # Cumulative DEVIATIONS avoid cancellation against the O(sqrt(N)) rest sum.
    s = initial.copy()
    max_residual = max(1e-22, 2e-11 * force * np.max(np.abs(rest_geo[1])))
    def objective(x, geo):
        # Avoid subtracting two almost equal absolute lengths at very low force.
        delta_x = rest_geo[0]*np.expm1(geo[6]-rest_geo[6]) + geo[5]*(geo[7]-rest_geo[7])
        return np.dot(a*x, x) - force*delta_x
    for iteration in range(60):
        geo = geometry(s, rest_theta)
        grad = 2*a*s - force*geo[1]
        if np.max(np.abs(grad)) < max_residual:
            return s, geo, iteration, np.max(np.abs(grad))
        step = solve_hessian(a, force, geo, grad)
        if np.max(np.abs(step)) < 5e-14:
            trial = s-step
            trial_geo = geometry(trial, rest_theta)
            return trial, trial_geo, iteration+1, np.max(np.abs(2*a*trial-force*trial_geo[1]))
        directional = np.dot(grad, step)
        if directional <= 0:
            raise RuntimeError("Nonpositive Newton curvature")
        current = objective(s, geo)
        scale = 1.0
        for _ in range(35):
            trial = s-scale*step
            trial_geo = geometry(trial, rest_theta)
            trial_grad = 2*a*trial-force*trial_geo[1]
            if np.max(np.abs(trial_grad)) < .9*np.max(np.abs(grad)):
                break
            # At machine precision accept a sufficiently small Newton update.
            if objective(trial, trial_geo) <= current - 1e-4*scale*directional + 1e-30:
                break
            if np.max(np.abs(scale*step)) < 2e-15:
                return trial, trial_geo, iteration+1, np.max(np.abs(2*a*trial-force*trial_geo[1]))
            scale /= 2
        s = trial
    raise RuntimeError(f"Newton iteration limit at force={force}")


def verify():
    rng = np.random.default_rng(71)
    n = 8
    s = rng.normal(0, .04, n)
    a = rng.uniform(.1, 1, n)
    rest = rng.normal(0, .04, n)
    theta = difference(s)
    rest_theta = difference(rest)
    original = sum(a[i]*(theta[i]+theta[:i].sum()-rest_theta[:i+1].sum())**2 for i in range(n))
    assert np.allclose(original, np.dot(a*(s-rest), s-rest), atol=1e-14)
    # Independent recurrence matching nlevel_5.structure_size.
    xmax, ymin = 1., 1.
    for angle in theta:
        xmax, ymin = np.cos(angle/2)*xmax + np.sin(angle/2)*ymin, np.cos(angle/2)*ymin
    assert np.allclose(xmax, geometry(s)[0], atol=1e-14)
    geo = geometry(s)
    delta = 1e-6
    numeric_gradient = np.array([(geometry(s+delta*np.eye(n)[i])[0]-geometry(s-delta*np.eye(n)[i])[0])/(2*delta) for i in range(n)])
    assert np.allclose(geo[1], numeric_gradient, atol=2e-10)
    numeric_hess = np.column_stack([(geometry(s+delta*np.eye(n)[i])[1]-geometry(s-delta*np.eye(n)[i])[1])/(2*delta) for i in range(n)])
    force = .12
    rhs = rng.normal(size=n)
    solution = solve_hessian(a, force, geo, rhs)
    assert np.allclose((np.diag(2*a)-force*numeric_hess) @ solution, rhs, atol=2e-10)
    print("Verified: original phi+ energy, original length recurrence, gradient, Hessian solve")


def sweep(n, p, eta, points, reference="maximum"):
    m = n//2
    a = np.ones(n)
    a[::2] = (np.arange(1, m+1)/m)**p
    rest_theta = np.full(n, -eta/np.sqrt(m))
    rest_theta[::2] = eta/np.sqrt(m)
    if reference == "maximum":
        theta_star = 2*np.arctan(2/(1+np.sqrt(4*n-3)))
        rest_theta += theta_star
        if np.min(rest_theta) <= 0:
            raise ValueError("eta is too large to keep all reference angles positive")
    rest_geo = geometry(np.zeros(n), rest_theta)
    x0 = rest_geo[0]
    compliance = np.dot(rest_geo[1]/(2*a), rest_geo[1]) / x0
    forces = np.geomspace(1e-3*m**(-p), 1., points) / (x0 if reference == "maximum" else 1.)
    s = np.zeros(n)
    strains, residuals, relative_residuals, max_angles, min_angles = [], [], [], [], []
    for force in forces:
        s, geo, _, residual = solve_equilibrium(a, rest_theta, force, s, rest_geo)
        strain = np.expm1(geo[6]-rest_geo[6]) + geo[5]/x0*(geo[7]-rest_geo[7])
        strains.append(strain)
        residuals.append(residual)
        relative_residuals.append(residual / max(force*np.max(abs(geo[1])), 1e-30))
        angles = rest_theta + difference(s)
        max_angles.append(np.abs(angles).max())
        min_angles.append(angles.min())
    strains = np.array(strains)
    ratios = strains/(compliance*forces)
    hits = np.flatnonzero(ratios <= .9)
    if len(hits) == 0 or hits[0] == 0:
        raise RuntimeError("Force grid does not bracket the crossover")
    idx = hits[0]
    fraction = (.9-ratios[idx-1])/(ratios[idx]-ratios[idx-1])
    f_cross = np.exp(np.log(forces[idx-1])+fraction*np.log(forces[idx]/forces[idx-1]))
    eps_cross = .9*compliance*f_cross
    slopes = np.gradient(np.log(forces), np.log(strains))
    # Broad intermediate interval, deliberately independent of fitted slopes.
    normalized_forces = forces * (x0 if reference == "maximum" else 1.)
    asymptotic = (normalized_forces > 100*m**(-p)) & (normalized_forces < .01)
    fit = np.polyfit(np.log(strains[asymptotic]), np.log(forces[asymptotic]), 1)[0] if asymptotic.sum()>3 else None
    record = dict(n=n, m=m, p=p, eta=eta, reference=reference, x0=x0, compliance=compliance,
                  f_cross_10pct=f_cross, strain_cross_10pct=eps_cross,
                  fitted_exponent=fit, max_absolute_residual=max(residuals),
                  max_relative_residual=max(relative_residuals),
                  max_angle=max(max_angles), min_angle=min(min_angles))
    return record, np.column_stack((forces, strains, slopes, ratios, residuals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--powers", type=float, nargs="+", default=[2., 3.])
    parser.add_argument("--eta", type=float, default=.5)
    parser.add_argument("--points", type=int, default=360)
    parser.add_argument("--reference", choices=["square", "maximum"], default="maximum")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.sizes) < 2 or any(n < 4 or n % 2 for n in args.sizes):
        parser.error("provide at least two even sizes, each >=4")
    if any(p <= 1 for p in args.powers) or args.eta <= 0:
        parser.error("powers must exceed one and eta must be positive")
    verify()
    out = Path(__file__).resolve().parent / "results"
    if args.reference == "maximum":
        out /= "maximum"
    if args.output is not None:
        out = args.output
    out.mkdir(exist_ok=True, parents=True)
    records = []
    fig, axes = plt.subplots(2, len(args.powers), figsize=(6*len(args.powers), 8), squeeze=False)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    for column, p in enumerate(args.powers):
        subset = []
        for n in args.sizes:
            record, curve = sweep(n, p, args.eta, args.points, args.reference)
            records.append(record)
            subset.append(record)
            np.savetxt(out/f"curve_p{p:g}_N{n}.csv", curve, delimiter=",", header="force,strain_relative_to_X0,local_exponent,linear_response_ratio,absolute_gradient_residual", comments="")
            print(json.dumps(record), flush=True)
            vertical_scale = record["x0"] if args.reference == "maximum" else 1.0
            axes[0,column].loglog(curve[:,1], curve[:,0]*vertical_scale, label=f"N={n}")
            axes[1,column].semilogx(curve[:,1], curve[:,2])
        ylabel = "Force f X0" if args.reference == "maximum" else "Force f"
        axes[0,column].set(title=f"All phi+, graded pairs, p={p:g}", ylabel=ylabel, xlabel="Strain (X-X0)/X0")
        axes[0,column].legend(fontsize=8)
        axes[1,column].axhline(p, color="black", linestyle="--", linewidth=1)
        axes[1,column].axhline(1, color="gray", linestyle=":")
        axes[1,column].set(xlabel="Strain (X-X0)/X0", ylabel="d log f / d log strain", ylim=(.5, 5), xlim=(1e-5, .2))
        ns = np.array([r['n'] for r in subset])
        eh = np.array([r['strain_cross_10pct'] for r in subset])
        exponent = np.polyfit(np.log(ns), np.log(eh), 1)[0]
        ax2.loglog(ns, eh, 'o-', label=f"p={p:g}, fitted size exponent {exponent:.3f}")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/"force_strain_and_exponents.png", dpi=180)
    ax2.set(xlabel="Number of levels N", ylabel="10% crossover strain")
    ax2.grid(alpha=.2)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out/"crossover_scaling.png", dpi=180)
    (out/"summary.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
