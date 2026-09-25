import unittest

import numpy as np

from vqe_tfim_mps import alternating_state, exact_ground_energy, tfim_hamiltonian


class TransverseFieldIsingTests(unittest.TestCase):
    def test_alternating_state(self) -> None:
        np.testing.assert_array_equal(
            alternating_state(10), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        )

    def test_initial_neel_energy_without_field_expectation(self) -> None:
        state = alternating_state(4)
        hamiltonian = tfim_hamiltonian(4, coupling=1.0, field=1.0)
        matrix = hamiltonian.matrix(wire_order=range(4))
        basis_index = int("".join(map(str, state)), 2)
        basis_vector = np.zeros(16, dtype=complex)
        basis_vector[basis_index] = 1.0
        energy = np.vdot(basis_vector, matrix @ basis_vector).real
        self.assertAlmostEqual(energy, -3.0)

    def test_sparse_exact_energy_matches_dense_reference(self) -> None:
        hamiltonian = tfim_hamiltonian(4, coupling=0.7, field=1.2)
        dense_ground_energy = np.linalg.eigvalsh(
            hamiltonian.matrix(wire_order=range(4))
        )[0]
        sparse_ground_energy = exact_ground_energy(4, coupling=0.7, field=1.2)
        self.assertAlmostEqual(sparse_ground_energy, dense_ground_energy, places=10)


if __name__ == "__main__":
    unittest.main()
