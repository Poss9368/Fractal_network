"""Plot saved phi-minus simulation CSVs without rerunning the solver."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path
import tempfile

import numpy as np


def _truth(value) -> bool:
    return value is True or str(value).lower() in {"true", "1"}


def _float(row, key) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _segments(mask: np.ndarray):
    """Yield contiguous valid index ranges, preserving gaps from failed samples."""
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        yield np.arange(start, stop)


def local_exponent(rows: list[dict]) -> np.ndarray:
    """Return d log(f)/d log(epsilon), never bridging invalid/nonmonotone points.

    The origin, endpoints of each usable segment and segments shorter than three
    samples are left NaN. Input rows must belong to a single ordered curve.
    """
    force = np.array([_float(row, "force") for row in rows])
    epsilon = np.array([_float(row, "epsilon") for row in rows])
    valid = np.array([_truth(r.get("converged")) and _truth(r.get("stable")) for r in rows])
    valid &= np.isfinite(force) & np.isfinite(epsilon) & (force > 0) & (epsilon > 0)
    result = np.full(len(rows), np.nan)
    for segment in _segments(valid):
        # Split at repeats or reversals before taking logarithmic derivatives.
        breaks = np.flatnonzero((np.diff(force[segment]) <= 0)
                               | (np.diff(epsilon[segment]) <= 0)) + 1
        for indices in np.split(segment, breaks):
            if indices.size >= 3:
                derivatives = np.gradient(np.log(force[indices]), np.log(epsilon[indices]))
                result[indices[1:-1]] = derivatives[1:-1]
    return result


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generate_plots(run_dir: str | Path) -> list[Path]:
    """Write four diagnostic PNGs; matplotlib is required only for this step."""
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="phi_minus_mpl_")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    curve_rows = _read_csv(run_dir / "curves.csv")
    spectrum_rows = _read_csv(run_dir / "spectrum.csv")
    groups = defaultdict(list)
    for row in curve_rows:
        groups[(int(row["N"]), float(row["s"]), float(row["k_scale"]), row["method"])].append(row)
    model_keys = sorted({key[:3] for key in groups})
    colors = {key: plt.get_cmap("tab10")(i % 10) for i, key in enumerate(model_keys)}

    fig_curve, ax_curve = plt.subplots(figsize=(8, 5.5))
    fig_beta, ax_beta = plt.subplots(figsize=(8, 5.5))
    fig_angle, ax_angle = plt.subplots(figsize=(8, 5.5))
    for key, rows in sorted(groups.items()):
        n, s, k_scale, method = key
        label = f"N={n}, s={s:g}, {method}"
        if len({key[2] for key in model_keys}) > 1:
            label += f", k*={k_scale:g}"
        color = colors[key[:3]]
        linestyle = "--" if method == "quadratic" else "-"
        force = np.array([_float(row, "force") for row in rows])
        epsilon = np.array([_float(row, "epsilon") for row in rows])
        angles = np.array([_float(row, "max_abs_theta") for row in rows])
        valid = np.array([_truth(row["converged"]) and _truth(row["stable"]) for row in rows])
        valid &= np.isfinite(force) & (force > 0)
        labeled_curve = labeled_angle = False
        for indices in _segments(valid & np.isfinite(epsilon) & (epsilon > 0)):
            ax_curve.loglog(epsilon[indices], force[indices], linestyle=linestyle,
                            color=color, label=label if not labeled_curve else None)
            labeled_curve = True
        for indices in _segments(valid & np.isfinite(angles)):
            ax_angle.semilogx(force[indices], angles[indices], linestyle=linestyle,
                             color=color, label=label if not labeled_angle else None)
            labeled_angle = True
        beta = local_exponent(rows)
        labeled_beta = False
        for indices in _segments(np.isfinite(beta)):
            ax_beta.semilogx(epsilon[indices], beta[indices], linestyle=linestyle,
                            color=color, label=label if not labeled_beta else None)
            labeled_beta = True
        # Mark where the small-angle diagnostic first fails on the sampled branch.
        outside = np.flatnonzero(valid & (angles > 0.2) & (epsilon > 0))
        if outside.size:
            first = outside[0]
            ax_curve.plot(epsilon[first], force[first], "x", color=color, markersize=7)
            if np.isfinite(beta[first]):
                ax_beta.plot(epsilon[first], beta[first], "x", color=color, markersize=7)

    ax_curve.set(xlabel=r"Strain $\varepsilon$", ylabel=r"Force $f=FD$",
                 title="Horizontal response; x marks first max |theta| > 0.2 rad")
    ax_beta.axhline(1, color="0.5", linewidth=1, label="Hooke (beta=1)")
    ax_beta.axhline(2, color="0.5", linewidth=1, linestyle=":", label="beta=2 guide")
    ax_beta.set(xlabel=r"Strain $\varepsilon$", ylabel=r"$\beta=d\log f/d\log\varepsilon$",
                title="Local slope; a crossing alone does not establish a power law")
    ax_angle.axhline(0.2, color="0.5", linewidth=1, linestyle=":", label="0.2 rad diagnostic")
    ax_angle.set(xlabel=r"Force $f=FD$", ylabel=r"$\max_i |\theta_i|$ [rad]",
                 title="Angular range of the equilibrium branch")

    fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 5.5))
    spectral_groups = defaultdict(list)
    for row in spectrum_rows:
        spectral_groups[(int(row["N"]), float(row["s"]), float(row["k_scale"]))].append(row)
    for key, rows in sorted(spectral_groups.items()):
        b = np.array([_float(row, "b") for row in rows])
        weight = np.array([_float(row, "modal_weight") for row in rows])
        valid = np.isfinite(b) & np.isfinite(weight) & (b > 0) & (weight > 0)
        ax_spectrum.loglog(b[valid], weight[valid], "o", color=colors.get(key),
                          label=f"N={key[0]}, s={key[1]:g}")
    ax_spectrum.set(xlabel=r"Modal stiffness $b_p$", ylabel=r"Weight $P_p=(v_p^T\mathbf{1})^2$",
                    title="Stiffnesses and coupling to the reference geometry")

    files = []
    for figure, axes, filename in [
        (fig_curve, ax_curve, "force_strain.png"),
        (fig_beta, ax_beta, "local_exponent.png"),
        (fig_angle, ax_angle, "max_angle.png"),
        (fig_spectrum, ax_spectrum, "weighted_spectrum.png"),
    ]:
        axes.grid(True, which="major", alpha=0.25)
        handles, labels = axes.get_legend_handles_labels()
        if handles:
            axes.legend(fontsize=8, loc="best")
        figure.tight_layout()
        destination = run_dir / filename
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        files.append(destination)
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing curves.csv and spectrum.csv.")
    args = parser.parse_args(argv)
    for path in generate_plots(args.run_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
