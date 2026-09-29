"""Pruebas de la energía general phi-/phi+ de nlevel."""

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
        rest_deformations_from_thetas,
        rest_thetas_from_noise,
    )
except ImportError:  # Permite ejecutar este archivo directamente.
    from dynamic import (
        balanced_random_signs,
        potential,
        potential_by_level,
        potential_gradient,
        modified_hamiltonian,
        modified_hamiltonian_gradient,
        rest_deformations_from_thetas,
        rest_thetas_from_noise,
    )


class GeneralHingeEnergyTests(unittest.TestCase):
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

    def test_energy_uses_weights_and_independent_rest_deformations(self):
        theta = np.array([0.12, -0.07, 0.21, 0.04])
        k_i = np.array([0.3, 0.8, 1.4, 0.6])
        family_weights = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 0.75],
            [1.0, 1.0],
        ])
        rest_deformations = np.array([
            [0.02, -0.03],
            [-0.01, 0.04],
            [0.03, 0.02],
            [-0.04, 0.01],
        ])

        expected_by_level = np.empty(theta.size)
        cumulative = 0.0
        for i in range(theta.size):
            residual_minus = theta[i] - cumulative - rest_deformations[i, 0]
            residual_plus = theta[i] + cumulative - rest_deformations[i, 1]
            # El argumento ya contiene la multiplicidad: w_i, no k_i microscópico.
            prefactor = k_i[i]
            expected_by_level[i] = prefactor * (
                family_weights[i, 0] * residual_minus**2
                + family_weights[i, 1] * residual_plus**2
            )
            cumulative += theta[i]

        np.testing.assert_allclose(
            potential_by_level(
                theta, k_i, family_weights, rest_deformations
            ),
            expected_by_level,
        )
        self.assertAlmostEqual(
            potential(theta, k_i, family_weights, rest_deformations),
            float(np.sum(expected_by_level)),
        )

    def test_energy_gradient_matches_finite_differences(self):
        theta = np.array([0.16, -0.03, 0.09, 0.24])
        k_i = np.array([0.4, 1.1, 0.7, 1.6])
        family_weights = np.array([
            [0.2, 0.9], [1.0, 0.3], [0.4, 0.7], [0.8, 0.1]
        ])
        rest_deformations = np.array([
            [-0.02, 0.03], [0.04, -0.01], [-0.01, 0.02], [0.03, 0.05]
        ])
        step = 1e-6
        numerical = np.empty(theta.size)

        for j in range(theta.size):
            direction = np.zeros(theta.size)
            direction[j] = step
            numerical[j] = (
                potential(
                    theta + direction, k_i, family_weights,
                    rest_deformations,
                )
                - potential(
                    theta - direction, k_i, family_weights,
                    rest_deformations,
                )
            ) / (2.0 * step)

        np.testing.assert_allclose(
            potential_gradient(
                theta, k_i, family_weights, rest_deformations
            ),
            numerical,
            rtol=2e-8,
            atol=2e-8,
        )

    def test_compatible_rest_configuration_has_zero_energy(self):
        noise = np.array([0.02, -0.03, 0.01])
        rest = rest_thetas_from_noise(noise)
        rest_deformations = rest_deformations_from_thetas(rest)
        family_weights = np.ones((3, 2))
        self.assertAlmostEqual(
            potential(rest, np.ones(3), family_weights, rest_deformations),
            0.0,
        )

    def test_family_limits(self):
        theta = np.array([0.1, 0.2, -0.05])
        k_i = np.array([0.4, 0.7, 1.2])
        rest = np.zeros((3, 2))
        only_minus = np.column_stack((np.ones(3), np.zeros(3)))
        only_plus = np.column_stack((np.zeros(3), np.ones(3)))
        both = np.ones((3, 2))

        self.assertAlmostEqual(
            potential(theta, k_i, both, rest),
            potential(theta, k_i, only_minus, rest)
            + potential(theta, k_i, only_plus, rest),
        )

    def test_hamiltonian_gradient_matches_finite_differences(self):
        theta = np.array([0.14, 0.05, -0.08])
        k_i = np.array([0.6, 1.3, 0.9])
        family_weights = np.array([[0.3, 0.8], [1.0, 0.2], [0.6, 0.7]])
        rest_deformations = np.array([[0.01, 0.03], [-0.02, 0.04], [0.04, -0.01]])
        force = 0.37
        step = 1e-6
        numerical = np.empty(theta.size)

        for j in range(theta.size):
            direction = np.zeros(theta.size)
            direction[j] = step
            numerical[j] = (
                modified_hamiltonian(
                    theta + direction, k_i, family_weights,
                    rest_deformations, force,
                )
                - modified_hamiltonian(
                    theta - direction, k_i, family_weights,
                    rest_deformations, force,
                )
            ) / (2.0 * step)

        np.testing.assert_allclose(
            modified_hamiltonian_gradient(
                theta, k_i, family_weights, rest_deformations, force
            ),
            numerical,
            rtol=2e-8,
            atol=2e-8,
        )

    def test_rejects_invalid_weight_shape_and_range(self):
        theta = np.zeros(2)
        k_i = np.ones(2)
        rest = np.zeros((2, 2))

        with self.assertRaises(ValueError):
            potential(theta, k_i, np.ones(2), rest)

        invalid = np.array([[1.0, 0.0], [0.0, 1.1]])
        with self.assertRaises(ValueError):
            potential(theta, k_i, invalid, rest)


if __name__ == "__main__":
    unittest.main()
