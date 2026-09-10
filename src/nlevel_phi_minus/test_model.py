"""Verificación independiente de energía, geometría y equilibrio de phi-.

Ejecutar desde la raíz: python3 -m unittest src.nlevel_phi_minus.test_model
También admite unittest discover con esta carpeta como directorio inicial.
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import brentq

try:
    from .model import PhiMinusModel
    from .solvers import solve_nonlinear, solve_quadratic
except ImportError:
    from model import PhiMinusModel
    from solvers import solve_nonlinear, solve_quadratic


def direct_length(theta):
    """Polinomio trigonométrico sin recurrencia ni divisiones por coseno."""
    c = np.cos(np.asarray(theta) / 2)
    s = np.sin(np.asarray(theta) / 2)
    return np.prod(c) + sum(s[i] * np.prod(np.delete(c, i)) for i in range(len(c)))


class EnergyTests(unittest.TestCase):
    def test_physical_hinge_energy_and_alpha_normalization(self):
        theta = np.array([0.07, -0.11, 0.19, 0.03])
        k_i = np.array([0.03, 0.2, 0.7, 1.3])
        model = PhiMinusModel(4, k_i=k_i)
        phi_minus = np.array([theta[i] - sum(theta[:i]) for i in range(4)])
        by_level = np.array([
            2 * 4 ** (3 - i) * k_i[i] * phi_minus[i] ** 2
            for i in range(4)
        ])
        assert_allclose(model.hinge_angles(theta), phi_minus)
        assert_allclose(model.energy_by_level(theta), by_level)
        self.assertAlmostEqual(model.energy(theta), sum(by_level))
        alpha_delta = -theta / 2
        self.assertAlmostEqual(
            model.energy(theta),
            0.5 * alpha_delta @ model.B_alpha @ alpha_delta,
        )

    def test_compensated_and_uniform_individual_rigidities(self):
        scale = 2.5
        compensated = PhiMinusModel(4, exponent=1, k_scale=scale)
        assert_allclose(compensated.k_i, scale * np.array([1 / 64, 1 / 16, 1 / 4, 1]))
        assert_allclose(compensated.q, np.full(4, scale))
        individual = PhiMinusModel(4, exponent=0, k_scale=scale)
        assert_allclose(individual.k_i, np.full(4, scale))
        assert_allclose(individual.q, scale * np.array([64, 16, 4, 1]))

    def test_custom_rigidities_override_schedule_and_are_copied(self):
        k_i = np.array([0.4, 0.7, 1.3])
        model = PhiMinusModel(3, exponent=0.5, k_scale=4, k_i=k_i)
        assert_allclose(model.q, [6.4, 2.8, 1.3])
        k_i[0] = 7
        self.assertAlmostEqual(model.k_i[0], 0.4)

    def test_invalid_rigidities_and_parameters(self):
        for k_i in ([1, 2], [1, 2, 3, 4], [1, 0, 1], [1, -1, 1],
                    [1, np.nan, 1], [1, np.inf, 1]):
            with self.subTest(k_i=k_i), self.assertRaises(ValueError):
                PhiMinusModel(3, k_i=k_i)
        for levels in (0, -2, 1.5, True):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                PhiMinusModel(levels)
        for kwargs in ({"exponent": np.nan}, {"k_scale": 0}, {"k_scale": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PhiMinusModel(3, **kwargs)

    def test_energy_gradient_against_finite_difference(self):
        model = PhiMinusModel(5, exponent=0.5)
        theta = np.array([0.1, -0.03, 0.07, 0.09, -0.15])
        step = 1e-6
        eye = np.eye(5)
        numerical = np.array([
            (model.energy(theta + step * e) - model.energy(theta - step * e)) / (2 * step)
            for e in eye
        ])
        assert_allclose(model.energy_gradient(theta), numerical, rtol=2e-8, atol=1e-8)


class GeometryTests(unittest.TestCase):
    def test_reference_and_quadratic_derivatives(self):
        model = PhiMinusModel(5)
        eps, gradient, hessian, sizes = model.geometry(np.zeros(5))
        self.assertEqual(eps, 0)
        self.assertEqual(model.energy(np.zeros(5)), 0)
        assert_allclose(sizes, [1, 0, 1, 1])
        assert_allclose(gradient, np.full(5, 0.5))
        assert_allclose(hessian, -np.eye(5) / 4)

    def test_recursion_matches_closed_coordinate_and_four_sizes(self):
        theta = np.array([0.12, -0.23, 0.31, 0.08])
        model = PhiMinusModel(4)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        closed_x = np.prod(c) * (1 + np.tan(theta / 2).sum())
        prior_x = direct_length(theta[:-1])
        prior_inner_y = np.prod(c[:-1])
        expected = [
            closed_x,
            s[-1] * prior_inner_y,
            s[-1] * prior_x + c[-1] * prior_inner_y,
            np.prod(c),
        ]
        assert_allclose(model.structure_size(theta), expected, rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(model.strain(theta), direct_length(theta) - 1)

    def test_coordinate_remains_regular_at_cosine_zeros(self):
        for theta in ([np.pi], [np.pi, 0.2, -0.4], [np.pi, -np.pi, 0.1]):
            with self.subTest(theta=theta):
                model = PhiMinusModel(len(theta))
                eps, gradient, hessian, sizes = model.geometry(theta)
                self.assertTrue(np.isfinite(eps))
                self.assertTrue(np.all(np.isfinite(gradient)))
                self.assertTrue(np.all(np.isfinite(hessian)))
                self.assertAlmostEqual(sizes[0], direct_length(theta), places=14)

    def test_geometry_gradient_and_hessian_against_finite_difference(self):
        model = PhiMinusModel(4)
        theta = np.array([0.25, -0.12, 0.4, -0.21])
        _, gradient, hessian, sizes = model.geometry(theta)
        step = 1e-5
        eye = np.eye(4)
        numerical_gradient = np.array([
            (model.strain(theta + step * e) - model.strain(theta - step * e)) / (2 * step)
            for e in eye
        ])
        numerical_hessian = np.column_stack([
            (model.geometry(theta + step * e)[1] - model.geometry(theta - step * e)[1])
            / (2 * step)
            for e in eye
        ])
        assert_allclose(gradient, numerical_gradient, rtol=1e-8, atol=1e-10)
        assert_allclose(hessian, numerical_hessian, rtol=1e-8, atol=1e-10)
        assert_allclose(hessian, hessian.T, atol=1e-15)
        assert_allclose(np.diag(hessian), np.full(4, -sizes[0] / 4))

    def test_tiny_strain_does_not_disappear_in_subtraction(self):
        model = PhiMinusModel(3)
        theta = np.full(3, 1e-20)
        self.assertGreater(model.strain(theta), 0)
        self.assertAlmostEqual(model.strain(theta) / (1.5e-20), 1, places=14)


class EquilibriumTests(unittest.TestCase):
    def test_spectral_quadratic_solution_matches_independent_linear_system(self):
        for exponent in (0, 0.5, 1):
            model = PhiMinusModel(5, exponent=exponent)
            for force in (0, model.b[0] * 0.01, model.b[0], 3.0):
                with self.subTest(exponent=exponent, force=force):
                    result = solve_quadratic(model, force)
                    expected = np.linalg.solve(
                        model.B_alpha + force * np.eye(5), np.full(5, 2 * force)
                    )
                    assert_allclose(result.theta, expected, rtol=1e-10, atol=1e-12)
                    self.assertTrue(result.converged)

    def test_modal_strain_and_alpha_filter(self):
        model = PhiMinusModel(6)
        projection = model.modes.T @ np.ones(6)
        for force in (model.b[0] * 0.1, model.b[0], 1):
            with self.subTest(force=force):
                theta = solve_quadratic(model, force).theta
                alpha = 1 - theta / 2
                assert_allclose(
                    model.modes.T @ alpha,
                    model.b / (model.b + force) * projection,
                    atol=1e-14,
                )
                modal_strain = 0.5 * np.sum(
                    model.modal_weights * (1 - (model.b / (model.b + force)) ** 2)
                )
                self.assertAlmostEqual(model.quadratic_strain(theta), modal_strain, places=12)

    def test_single_level_nonlinear_solution_matches_scalar_root(self):
        stiffness = 1.7
        model = PhiMinusModel(1, k_scale=stiffness)
        for force in (1e-5, 0.1, 1, 16):
            with self.subTest(force=force):
                result = solve_nonlinear(model, force, tol=1e-10)
                expected = brentq(
                    lambda theta: 4 * stiffness * theta
                    - force * (np.cos(theta / 2) - np.sin(theta / 2)) / 2,
                    0,
                    np.pi / 2,
                    xtol=1e-14,
                )
                self.assertTrue(result.converged, result.message)
                self.assertTrue(result.stable)
                assert_allclose(result.theta, [expected], rtol=1e-8, atol=1e-10)

    def test_nonlinear_continuation_and_small_force_limit(self):
        model = PhiMinusModel(6)
        initial = None
        previous_strain = 0
        forces = model.b[0] * np.geomspace(1e-6, 0.03, 8)
        for force in forces:
            with self.subTest(force=force):
                result = solve_nonlinear(model, force, initial=initial)
                self.assertTrue(result.converged, result.message)
                self.assertTrue(result.stable)
                self.assertGreater(result.hessian_min_scaled, 0)
                self.assertLess(result.residual_relative, 1e-7)
                eps, drive, _, _ = model.geometry(result.theta)
                gradient = model.energy_gradient(result.theta) - force * drive
                self.assertLess(np.linalg.norm(gradient, np.inf) / force, 2e-7)
                self.assertGreater(eps, previous_strain)
                self.assertLessEqual(model.hamiltonian(result.theta, force), 0)
                if initial is None:
                    assert_allclose(
                        result.theta, solve_quadratic(model, force).theta,
                        rtol=1e-4, atol=1e-12,
                    )
                initial = result.theta
                previous_strain = eps

    def test_zero_force_and_invalid_force(self):
        model = PhiMinusModel(3)
        for solve in (solve_quadratic, solve_nonlinear):
            zero = solve(model, 0)
            assert_allclose(zero.theta, np.zeros(3), atol=0)
            self.assertTrue(zero.converged)
            for force in (-1, np.nan, np.inf):
                with self.subTest(solver=solve.__name__, force=force), self.assertRaises(ValueError):
                    solve(model, force)
        for force in (-1, np.nan, np.inf):
            with self.subTest(force=force), self.assertRaises(ValueError):
                model.hamiltonian(np.zeros(3), force)


if __name__ == "__main__":
    unittest.main()
