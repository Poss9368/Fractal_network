"""Pruebas de consistencia para la energía phi- de nlevel_3."""

import unittest

import numpy as np

try:
    from .dynamic import (
        balanced_random_signs,
        potential,
        potential_by_level,
        potential_gradient,
        modified_hamiltonian,
        modified_hamiltonian_gradient,
        rest_thetas_from_noise,
    )
except ImportError:  # Permite ejecutar este archivo directamente.
    from old.nlevel.dynamic import (
        balanced_random_signs,
        potential,
        potential_by_level,
        potential_gradient,
        modified_hamiltonian,
        modified_hamiltonian_gradient,
        rest_thetas_from_noise,
    )


class PhiMinusEnergyTests(unittest.TestCase):
    def test_balanced_random_signs(self):
        signs = balanced_random_signs(10, np.random.RandomState(123))
        self.assertEqual(signs.shape, (10,))
        self.assertEqual(np.sum(signs), 0.0)
        self.assertEqual(np.count_nonzero(signs == -1.0), 5)
        self.assertEqual(np.count_nonzero(signs == 1.0), 5)

    def test_balanced_random_signs_is_reproducible(self):
        first = balanced_random_signs(8, np.random.RandomState(42))
        second = balanced_random_signs(8, np.random.RandomState(42))
        np.testing.assert_array_equal(first, second)

    def test_balanced_random_signs_rejects_invalid_n(self):
        for invalid_n in (0, 1, 3, 5, -2, 4.0, True):
            with self.subTest(n=invalid_n), self.assertRaises(ValueError):
                balanced_random_signs(invalid_n)

    def test_energy_uses_physical_hinge_stiffness(self):
        theta = np.array([0.12, -0.07, 0.21, 0.04])
        k_i = np.array([0.3, 0.8, 1.4, 0.6])
        noise = np.array([0.02, -0.01, 0.03, -0.04])
        rest = rest_thetas_from_noise(noise)

        residual = np.empty(theta.size)
        for i in range(theta.size):
            residual[i] = (
                theta[i] - np.sum(theta[:i])
                - rest[i] + np.sum(rest[:i])
            )

        weights = 4.0 ** np.arange(theta.size - 1, -1, -1) * k_i
        expected_by_level = weights * residual**2

        np.testing.assert_allclose(
            potential_by_level(theta, k_i, noise), expected_by_level
        )
        self.assertAlmostEqual(
            potential(theta, k_i, noise), float(np.sum(expected_by_level))
        )

    def test_energy_gradient_matches_finite_differences(self):
        theta = np.array([0.16, -0.03, 0.09, 0.24])
        k_i = np.array([0.4, 1.1, 0.7, 1.6])
        noise = np.array([-0.02, 0.04, -0.01, 0.03])
        step = 1e-6
        numerical = np.empty(theta.size)

        for j in range(theta.size):
            direction = np.zeros(theta.size)
            direction[j] = step
            numerical[j] = (
                potential(theta + direction, k_i, noise)
                - potential(theta - direction, k_i, noise)
            ) / (2.0 * step)

        np.testing.assert_allclose(
            potential_gradient(theta, k_i, noise),
            numerical,
            rtol=2e-8,
            atol=2e-8,
        )

    def test_rest_configuration_has_zero_energy(self):
        noise = np.array([0.02, -0.03, 0.01])
        rest = rest_thetas_from_noise(noise)
        self.assertAlmostEqual(potential(rest, np.ones(3), noise), 0.0)

    def test_hamiltonian_gradient_matches_finite_differences(self):
        theta = np.array([0.14, 0.05, -0.08])
        k_i = np.array([0.6, 1.3, 0.9])
        noise = np.array([0.01, -0.02, 0.04])
        force = 0.37
        step = 1e-6
        numerical = np.empty(theta.size)

        for j in range(theta.size):
            direction = np.zeros(theta.size)
            direction[j] = step
            numerical[j] = (
                modified_hamiltonian(theta + direction, k_i, noise, force)
                - modified_hamiltonian(theta - direction, k_i, noise, force)
            ) / (2.0 * step)

        np.testing.assert_allclose(
            modified_hamiltonian_gradient(theta, k_i, noise, force),
            numerical,
            rtol=2e-8,
            atol=2e-8,
        )


if __name__ == "__main__":
    unittest.main()
