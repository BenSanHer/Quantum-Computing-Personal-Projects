"""Exact diagonalization utilities for the 20-spin transverse-field Ising model.

The Hamiltonian is applied as a matrix-free sparse operator:

    H = -J sum_i Z_i Z_(i+1) - h sum_i X_i

This keeps the 2**20-dimensional Hilbert space explicit without constructing
a dense 2**20 by 2**20 matrix. The returned eigenvectors are the recovered
ground-state amplitudes in the computational basis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


DEFAULT_FIELDS = tuple(round(value, 1) for value in np.arange(0.2, 2.01, 0.2))


def basis_data(num_spins: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Prepare computational-basis indices, diagonal ZZ energy, and X flips."""

    dimension = 2**num_spins
    basis = np.arange(dimension, dtype=np.int64)
    zz_diagonal = np.zeros(dimension, dtype=np.float64)
    for spin in range(num_spins - 1):
        z_first = 1.0 - 2.0 * ((basis >> spin) & 1)
        z_second = 1.0 - 2.0 * ((basis >> (spin + 1)) & 1)
        zz_diagonal -= z_first * z_second
    flips = [basis ^ (1 << spin) for spin in range(num_spins)]
    return basis, zz_diagonal, flips


def tfim_linear_operator(
    num_spins: int, coupling: float, field: float
) -> tuple[LinearOperator, np.ndarray, list[np.ndarray]]:
    """Build a matrix-free real symmetric operator for the TFIM Hamiltonian."""

    _, zz_diagonal, flips = basis_data(num_spins)
    dimension = 2**num_spins

    def matvec(vector: np.ndarray) -> np.ndarray:
        result = coupling * zz_diagonal * vector
        for flip_indices in flips:
            result = result - field * vector[flip_indices]
        return result

    operator = LinearOperator(
        shape=(dimension, dimension),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )
    return operator, zz_diagonal, flips


def diagonalize_ground_state(
    *,
    num_spins: int = 20,
    coupling: float = 1.0,
    field: float = 0.2,
    initial_vector: np.ndarray | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 1000,
    krylov_vectors: int = 24,
) -> dict[str, Any]:
    """Recover the lowest eigenvalue and eigenvector for one field value."""

    operator, _, _ = tfim_linear_operator(num_spins, coupling, field)
    dimension = 2**num_spins
    if initial_vector is not None:
        initial_vector = np.asarray(initial_vector, dtype=np.float64)
        initial_vector = initial_vector / np.linalg.norm(initial_vector)

    start_time = time.perf_counter()
    eigenvalues, eigenvectors = eigsh(
        operator,
        k=1,
        which="SA",
        v0=initial_vector,
        tol=tolerance,
        maxiter=max_iterations,
        ncv=krylov_vectors,
        return_eigenvectors=True,
    )
    elapsed = time.perf_counter() - start_time
    energy = float(eigenvalues[0])
    state = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    state /= np.linalg.norm(state)
    residual = np.linalg.norm(operator.matvec(state) - energy * state)

    probabilities = state**2
    basis = np.arange(dimension, dtype=np.int64)
    mean_x = 0.0
    for spin in range(num_spins):
        mean_x += float(np.dot(state, state[basis ^ (1 << spin)]))
    mean_x /= num_spins

    mean_zz = 0.0
    _, zz_diagonal, _ = tfim_linear_operator(num_spins, coupling, field)
    mean_zz = float(np.dot(probabilities, zz_diagonal) / (num_spins - 1))

    return {
        "num_spins": num_spins,
        "coupling_J": coupling,
        "transverse_field_h": field,
        "energy": energy,
        "state": state,
        "state_norm": float(np.linalg.norm(state)),
        "residual_norm": float(residual),
        "mean_X": mean_x,
        "mean_ZZ": mean_zz,
        "runtime_seconds": elapsed,
    }


def diagonalize_grid(
    *,
    fields: Iterable[float] = DEFAULT_FIELDS,
    num_spins: int = 20,
    coupling: float = 1.0,
    tolerance: float = 1e-9,
    max_iterations: int = 1000,
    krylov_vectors: int = 24,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Recover one normalized ground state for every transverse field."""

    results: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    previous_state: np.ndarray | None = None
    for field in fields:
        field = float(field)
        print(f"ED | h={field:.1f}")
        result = diagonalize_ground_state(
            num_spins=num_spins,
            coupling=coupling,
            field=field,
            initial_vector=previous_state,
            tolerance=tolerance,
            max_iterations=max_iterations,
            krylov_vectors=krylov_vectors,
        )
        previous_state = result["state"]
        states.append(previous_state.copy())
        results.append({key: value for key, value in result.items() if key != "state"})
        print(
            f"  E0={result['energy']:.10f} | residual={result['residual_norm']:.3e} "
            f"| norm={result['state_norm']:.8f} | {result['runtime_seconds']:.2f}s"
        )
    return results, np.stack(states, axis=0)


def save_ed_results(
    results: list[dict[str, Any]], states: np.ndarray, output_directory: str | Path
) -> dict[str, Path]:
    """Save metadata, energies, fields, and recovered ground-state vectors."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "ed_ground_states.npz"
    metadata_path = directory / "ed_ground_state_metadata.json"
    fields = np.asarray([result["transverse_field_h"] for result in results])
    energies = np.asarray([result["energy"] for result in results])
    np.savez_compressed(state_path, fields=fields, energies=energies, ground_states=states)
    metadata_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"states": state_path, "metadata": metadata_path}
