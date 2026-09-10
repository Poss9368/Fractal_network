"""Energía y coordenada geométrica del modelo phi-, con delta=0.

Índices matemáticos i=1..N; índices Python 0..N-1. X(0)=1.
E = 2 sum q_i (theta_i - sum_{m<i} theta_m)^2,
q_i = 4**(N-i) k_i. La fuerza es conjugada a X, no a log(X).
Las longitudes son las coordenadas recursivas del modelo reducido;
no se identifican automáticamente con la caja envolvente de los cuadrados.
"""

import numpy as np


class PhiMinusModel:
    def __init__(self, levels, exponent=1.0, k_scale=1.0, k_i=None):
        if isinstance(levels, bool) or int(levels) != levels or levels < 1:
            raise ValueError("levels debe ser un entero positivo")
        self.levels = int(levels)
        if not np.isfinite(exponent) or not np.isfinite(k_scale) or k_scale <= 0:
            raise ValueError("exponent debe ser finito y k_scale positivo")
        self.exponent = float(exponent)
        self.k_scale = float(k_scale)
        offsets = np.arange(self.levels, dtype=float) + 1 - self.levels
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if k_i is None:
                self.k_i = self.k_scale * np.exp2(2 * self.exponent * offsets)
                self.q = self.k_scale * np.exp2(2 * (self.exponent - 1) * offsets)
            else:
                self.k_i = np.array(k_i, dtype=float, copy=True)
                self.q = self.k_i * np.exp2(-2 * offsets)
        for name, values in (("k_i", self.k_i), ("q", self.q)):
            if values.shape != (self.levels,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"{name} debe contener N valores positivos finitos representables")

        self.M = np.eye(self.levels) - np.tril(np.ones((self.levels, self.levels)), -1)
        # B_alpha = A.T @ A. SVD evita perder el modo blando al formar B.
        factor = 4 * np.sqrt(self.q)[:, None] * self.M
        _, singular, vt = np.linalg.svd(factor, full_matrices=False)
        if singular[-1] <= 100 * np.finfo(float).eps * singular[0]:
            raise ValueError("El modo blando no está resuelto en doble precisión; reduce N o el contraste de k_i")
        self.condition_number = float(singular[0] / singular[-1])
        self.b = singular[::-1] ** 2
        if not np.all(np.isfinite(self.b)) or np.any(self.b <= 0):
            raise ValueError("Las rigideces modales no son representables")
        self.modes = vt[::-1].T.copy()
        self.modal_weights = (self.modes.T @ np.ones(self.levels)) ** 2
        self.B_alpha = factor.T @ factor  # Para inspección; no se usa para resolver.

    def _theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.levels,) or not np.all(np.isfinite(theta)):
            raise ValueError("theta debe contener N ángulos finitos en radianes")
        return theta

    def hinge_angles(self, theta):
        theta = self._theta(theta)
        return theta - np.r_[0.0, np.cumsum(theta[:-1])]

    def energy_by_level(self, theta):
        return 2 * self.q * self.hinge_angles(theta) ** 2

    def energy(self, theta):
        return float(np.sum(self.energy_by_level(theta)))

    def energy_gradient(self, theta):
        return 4 * self.M.T @ (self.q * self.hinge_angles(theta))

    def geometry(self, theta):
        """Devuelve (X-1, grad X, Hess X, cuatro tamaños).

        Recurre X_new=c X_old+s B_old, B_new=c B_old, con X=B=1
        inicialmente. B aquí es Ly_min, no la matriz elástica.
        No usa tangentes: sigue siendo regular cuando cos(theta_i/2)=0.
        X-1 se acumula directamente para conservar precisión a fuerza baja.
        """
        theta = self._theta(theta)
        n = self.levels
        strain, inner_y = 0.0, 1.0
        gx, gy = np.zeros(n), np.zeros(n)
        hx, hy = np.zeros((n, n)), np.zeros((n, n))
        for i, angle in enumerate(theta):
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            old_x = 1 + strain
            xmax = c * old_x + s * inner_y
            ymax = s * old_x + c * inner_y
            xmin = s * inner_y
            next_gx = c * gx + s * gy
            next_gx[i] += (-s * old_x + c * inner_y) / 2
            next_hx = c * hx + s * hy
            cross_x = (-s * gx + c * gy) / 2
            next_hx[i, :] += cross_x
            next_hx[:, i] += cross_x
            next_hx[i, i] -= xmax / 4
            next_gy = c * gy
            next_gy[i] -= s * inner_y / 2
            next_hy = c * hy
            cross_y = -s * gy / 2
            next_hy[i, :] += cross_y
            next_hy[:, i] += cross_y
            next_hy[i, i] -= c * inner_y / 4
            strain = c * strain - 2 * np.sin(angle / 4) ** 2 + s * inner_y
            inner_y = c * inner_y
            gx, gy, hx, hy = next_gx, next_gy, next_hx, next_hy
        return float(strain), gx, hx, (float(1 + strain), float(xmin), float(ymax), float(inner_y))

    def structure_size(self, theta):
        return self.geometry(theta)[3]

    def strain(self, theta):
        return self.geometry(theta)[0]

    def quadratic_strain(self, theta):
        theta = self._theta(theta)
        return float(theta.sum() / 2 - theta @ theta / 8)

    def diagnostics(self, theta):
        theta = self._theta(theta)
        exact = self.strain(theta)
        return {
            "max_abs_theta": float(np.max(np.abs(theta))),
            "small_angle": bool(np.max(np.abs(theta)) <= 0.2),
            "geometry_relative_error": float(abs(exact - self.quadratic_strain(theta)) / max(abs(exact), 1e-12)),
        }

    def hamiltonian(self, theta, force, quadratic=False):
        if not np.isfinite(force) or force < 0:
            raise ValueError("force debe ser finita y no negativa")
        strain = self.quadratic_strain(theta) if quadratic else self.strain(theta)
        return self.energy(theta) - force * strain
