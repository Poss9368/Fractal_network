"""Equilibrio cuadrático por modos y minimización trigonométrica a fuerza fija."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class Solution:
    theta: np.ndarray
    converged: bool
    stable: bool
    iterations: int
    residual_relative: float
    hessian_min_scaled: float
    message: str


def _check_force(force):
    if not np.isfinite(force) or force < 0:
        raise ValueError("force debe ser finita y no negativa")


def solve_quadratic(model, force):
    """Resuelve (B_alpha+f I) theta=2 f 1 sin formar/invertir B_alpha."""
    _check_force(force)
    projections = model.modes.T @ np.ones(model.levels)
    coefficients = 2 * (force / (model.b + force)) * projections
    theta = model.modes @ coefficients
    residual = (model.b + force) * coefficients - 2 * force * projections
    relative = float(np.max(np.abs(residual)) / max(2 * force * np.max(np.abs(projections)), 1e-300))
    return Solution(theta, True, True, 0, relative, 1.0, "Solución espectral del Hamiltoniano cuadrático")


def solve_nonlinear(model, force, initial=None, tol=1e-8, maxiter=200):
    """Mínimo local de E-f(X-1), sin imponer cotas artificiales a theta.

    Para f>0: theta=P z, P=2 V diag(sqrt(f/(b+f))). Se minimiza H/f.
    En estas coordenadas la Hessiana cuadrática es I, incluso si hay un
    modo muy blando. El resultado se acepta por residuo Y estabilidad,
    no sólo por el indicador de éxito del optimizador.
    """
    _check_force(force)
    if not np.isfinite(tol) or tol <= 0 or maxiter < 1:
        raise ValueError("tol y maxiter deben ser positivos")
    if force == 0:
        return Solution(np.zeros(model.levels), True, True, 0, 0.0, 1.0, "Referencia descargada")

    r = np.sqrt(force / (model.b + force))
    elastic = model.b / (model.b + force)
    transform = 2 * model.modes * r[None, :]
    if initial is None:
        z0 = r * (model.modes.T @ np.ones(model.levels))
    else:
        z0 = (model.modes.T @ model._theta(initial)) / (2 * r)

    # scipy puede pedir valor, gradiente y Hessiana en el mismo punto.
    cache_z = None
    cache = None

    def evaluate(z):
        nonlocal cache_z, cache
        if cache_z is None or not np.array_equal(cache_z, z):
            eps, gx, hx, _ = model.geometry(transform @ z)
            value = 0.5 * np.dot(elastic * z, z) - eps
            gradient = elastic * z - transform.T @ gx
            hessian = np.diag(elastic) - transform.T @ hx @ transform
            cache_z = z.copy()
            cache = (value, gradient, (hessian + hessian.T) / 2, transform.T @ gx)
        return cache

    initial_drive = np.linalg.norm(evaluate(z0)[3], ord=np.inf)
    result = minimize(
        lambda z: evaluate(z)[0], z0,
        jac=lambda z: evaluate(z)[1], hess=lambda z: evaluate(z)[2],
        method="trust-exact",
        options={"gtol": tol * max(initial_drive, 1e-12), "maxiter": int(maxiter)},
    )
    z = result.x.copy()
    polish_steps = 0
    # Cerca del mínimo, la mejora de energía puede caer por debajo del
    # redondeo antes que el residuo. Corregimos sólo si una dirección de
    # Newton estable reduce el residuo y no aumenta la energía salvo redondeo.
    for _ in range(8):
        value, gradient, hessian, drive = evaluate(z)
        target = tol * max(np.linalg.norm(drive, ord=np.inf), 1e-12)
        if np.linalg.norm(gradient, ord=np.inf) <= target:
            break
        if np.linalg.eigvalsh(hessian)[0] <= 1e-10:
            break
        step = np.linalg.solve(hessian, -gradient)
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = z + fraction * step
            trial = evaluate(candidate)
            energy_slack = 64 * np.finfo(float).eps * max(1.0, abs(value))
            if (trial[0] <= value + energy_slack
                    and np.linalg.norm(trial[1], ord=np.inf) < np.linalg.norm(gradient, ord=np.inf)):
                z = candidate
                polish_steps += 1
                accepted = True
                break
        if not accepted:
            break
    value, gradient, hessian, drive = evaluate(z)
    relative = float(np.linalg.norm(gradient, ord=np.inf) / max(np.linalg.norm(drive, ord=np.inf), 1e-12))
    hmin = float(np.linalg.eigvalsh(hessian)[0])
    finite = np.isfinite(value) and np.all(np.isfinite(z)) and np.isfinite(relative)
    stable = bool(finite and hmin > 1e-10)
    converged = bool(finite and relative <= 10 * tol and stable)
    return Solution(
        transform @ z, converged, stable, int(result.nit) + polish_steps, relative, hmin,
        str(result.message) + (f"; {polish_steps} correcciones de Newton verificadas" if polish_steps else ""),
    )
