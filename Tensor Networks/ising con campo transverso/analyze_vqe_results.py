"""Observables and ED fidelities for the completed TFIM VQE sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exact_diagonalization import DEFAULT_FIELDS
from ising_tfim_gpu import ansatz_state, select_device, tfim_energy


def little_endian_z_matrix(num_spins: int, device: torch.device) -> torch.Tensor:
    """Return a [spin, basis-index] matrix of Z eigenvalues."""

    basis = torch.arange(2**num_spins, device=device, dtype=torch.int64)
    bit_positions = torch.arange(num_spins, device=device, dtype=torch.int64)
    bits = (basis[:, None] >> bit_positions[None, :]) & 1
    return (1.0 - 2.0 * bits.to(torch.float32)).T.contiguous()


def vqe_vector_little_endian(state: torch.Tensor) -> torch.Tensor:
    """Flatten the tensor-network state in the same basis convention as ED."""

    axes = tuple(reversed(range(state.ndim)))
    return state.permute(axes).reshape(-1).contiguous()


def global_x_flip_indices(num_spins: int, size: int) -> np.ndarray:
    """Basis indices produced by applying P_X = product_i X_i."""

    if size != 2**num_spins:
        raise ValueError("The state size must be 2**num_spins.")
    return np.arange(size, dtype=np.int64) ^ (size - 1)


def parity_projected_fidelity(
    target_state: np.ndarray,
    state: np.ndarray,
    sector: int,
) -> float:
    """Fidelity with target after normalizing the +/- parity projection."""

    if sector not in (-1, 1):
        raise ValueError("sector must be +1 or -1.")
    state = np.asarray(state, dtype=np.complex128)
    target_state = np.asarray(target_state, dtype=np.complex128)
    flip = global_x_flip_indices(
        int(round(np.log2(state.size))), state.size
    )
    projected = state + sector * state[flip]
    projected_norm = np.linalg.norm(projected)
    if projected_norm < 1e-12:
        return 0.0
    projected /= projected_norm
    return float(abs(np.vdot(target_state, projected)) ** 2)


def parity_projected_state(state: np.ndarray, sector: int) -> np.ndarray:
    """Return a normalized state in the requested global-X parity sector."""

    if sector not in (-1, 1):
        raise ValueError("sector must be +1 or -1.")
    state = np.asarray(state, dtype=np.complex128)
    flip = global_x_flip_indices(
        int(round(np.log2(state.size))), state.size
    )
    projected = state + sector * state[flip]
    projected_norm = np.linalg.norm(projected)
    if projected_norm < 1e-12:
        raise ValueError("The requested parity projection has negligible norm.")
    return projected / projected_norm


def vqe_observables(
    state: torch.Tensor,
    num_spins: int,
    coupling: float,
    field: float,
    z_matrix: torch.Tensor,
) -> dict[str, Any]:
    """Compute magnetizations, ZZ correlations, and reconstructed energy."""

    vector = vqe_vector_little_endian(state)
    probabilities = vector.abs().square().real
    z_expectations = z_matrix @ probabilities
    correlation_matrix = (z_matrix * probabilities.unsqueeze(0)) @ z_matrix.T
    x_expectations = torch.stack(
        [
            torch.real(
                torch.sum(
                    torch.conj(vector)
                    * vector[
                        torch.arange(
                            vector.numel(), device=vector.device, dtype=torch.int64
                        )
                        ^ (1 << spin)
                    ]
                )
            )
            for spin in range(num_spins)
        ]
    )
    global_flip = torch.arange(
        vector.numel(), device=vector.device, dtype=torch.int64
    ) ^ (vector.numel() - 1)
    parity_x = torch.real(torch.vdot(vector, vector[global_flip]))
    magnetization_per_basis = z_matrix.sum(dim=0) / num_spins
    zz_by_distance = [
        torch.diagonal(correlation_matrix, offset=distance).mean()
        for distance in range(num_spins)
    ]
    connected_matrix = correlation_matrix - z_expectations[:, None] * z_expectations[None, :]
    connected_by_distance = [
        torch.diagonal(connected_matrix, offset=distance).mean()
        for distance in range(num_spins)
    ]
    energy = (
        -coupling * torch.diagonal(correlation_matrix, offset=1).sum()
        - field * x_expectations.sum()
    )
    return {
        "energy_reconstructed": float(energy.detach().cpu()),
        "M_x": float(x_expectations.mean().detach().cpu()),
        "M_z": float(z_expectations.mean().detach().cpu()),
        "abs_M_z": float(
            torch.sum(probabilities * magnetization_per_basis.abs()).detach().cpu()
        ),
        "M_z_squared": float(
            torch.sum(probabilities * magnetization_per_basis.square()).detach().cpu()
        ),
        "parity_X": float(parity_x.detach().cpu()),
        "parity_variance": float((1.0 - parity_x.square()).detach().cpu()),
        "parity_plus_weight": float(((1.0 + parity_x) / 2.0).detach().cpu()),
        "parity_minus_weight": float(((1.0 - parity_x) / 2.0).detach().cpu()),
        "mean_ZZ_nearest": float(
            torch.diagonal(correlation_matrix, offset=1).mean().detach().cpu()
        ),
        "zz_correlation_by_distance": [
            float(value.detach().cpu()) for value in zz_by_distance
        ],
        "connected_zz_by_distance": [
            float(value.detach().cpu()) for value in connected_by_distance
        ],
        "state_norm": float(torch.linalg.vector_norm(vector).detach().cpu()),
        "state_vector": vector.detach().cpu().numpy(),
    }


def _numpy_z_matrix(num_spins: int) -> np.ndarray:
    basis = np.arange(2**num_spins, dtype=np.int64)
    bit_positions = np.arange(num_spins, dtype=np.int64)
    bits = (basis[:, None] >> bit_positions[None, :]) & 1
    return (1.0 - 2.0 * bits.astype(np.float32)).T


def ed_observables(
    state: np.ndarray,
    num_spins: int,
    coupling: float,
    field: float,
    z_matrix: np.ndarray,
) -> dict[str, Any]:
    """Compute the same observables directly from an ED ground state."""

    state = np.asarray(state, dtype=np.float64)
    probabilities = state**2
    basis = np.arange(state.size, dtype=np.int64)
    z_expectations = z_matrix @ probabilities
    correlation_matrix = (z_matrix * probabilities[None, :]) @ z_matrix.T
    x_expectations = np.asarray(
        [np.dot(state, state[basis ^ (1 << spin)]) for spin in range(num_spins)]
    )
    global_flip = global_x_flip_indices(num_spins, state.size)
    parity_x = float(np.real(np.vdot(state, state[global_flip])))
    magnetization_per_basis = z_matrix.sum(axis=0) / num_spins
    zz_by_distance = [
        np.diagonal(correlation_matrix, offset=distance).mean()
        for distance in range(num_spins)
    ]
    connected_matrix = correlation_matrix - z_expectations[:, None] * z_expectations[None, :]
    connected_by_distance = [
        np.diagonal(connected_matrix, offset=distance).mean()
        for distance in range(num_spins)
    ]
    energy = (
        -coupling * np.diagonal(correlation_matrix, offset=1).sum()
        - field * x_expectations.sum()
    )
    return {
        "energy_reconstructed": float(energy),
        "M_x": float(x_expectations.mean()),
        "M_z": float(z_expectations.mean()),
        "abs_M_z": float(np.dot(probabilities, np.abs(magnetization_per_basis))),
        "M_z_squared": float(np.dot(probabilities, magnetization_per_basis**2)),
        "parity_X": parity_x,
        "parity_variance": float(1.0 - parity_x**2),
        "parity_plus_weight": float((1.0 + parity_x) / 2.0),
        "parity_minus_weight": float((1.0 - parity_x) / 2.0),
        "mean_ZZ_nearest": float(np.diagonal(correlation_matrix, offset=1).mean()),
        "zz_correlation_by_distance": [float(value) for value in zz_by_distance],
        "connected_zz_by_distance": [float(value) for value in connected_by_distance],
        "state_norm": float(np.linalg.norm(state)),
    }


def analyze_vqe_results(
    *,
    vqe_path: str | Path,
    ed_state_path: str | Path,
    ed_metadata_path: str | Path,
    output_path: str | Path,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct all VQE states, compute observables, and save a JSON table."""

    if device is None:
        device = select_device(require_cuda=True)
    vqe_rows = json.loads(Path(vqe_path).read_text(encoding="utf-8"))
    ed_rows = json.loads(Path(ed_metadata_path).read_text(encoding="utf-8"))
    ed_archive = np.load(ed_state_path)
    ed_states = ed_archive["ground_states"]
    ed_fields = ed_archive["fields"]
    num_spins = int(vqe_rows[0]["num_spins"])
    coupling = float(vqe_rows[0]["coupling_J"])
    z_matrix_torch = little_endian_z_matrix(num_spins, device)
    z_matrix_numpy = _numpy_z_matrix(num_spins)
    ed_index = {round(float(field), 1): index for index, field in enumerate(ed_fields)}
    ed_metadata = {
        round(float(row["transverse_field_h"]), 1): row for row in ed_rows
    }

    analysis_rows: list[dict[str, Any]] = []
    for run_index, vqe_row in enumerate(vqe_rows, start=1):
        field = float(vqe_row["transverse_field_h"])
        field_key = round(field, 1)
        parameters = torch.tensor(
            vqe_row["final_parameters"], device=device, dtype=torch.float32
        )
        with torch.no_grad():
            state = ansatz_state(parameters, num_spins)
            vqe_values = vqe_observables(
                state, num_spins, coupling, field, z_matrix_torch
            )
        vqe_vector = vqe_values.pop("state_vector")
        ed_state = ed_states[ed_index[field_key]]
        ed_values = ed_observables(
            ed_state, num_spins, coupling, field, z_matrix_numpy
        )
        fidelity = float(abs(np.vdot(ed_state, vqe_vector)) ** 2)
        ed_even_state = parity_projected_state(ed_state, sector=1)
        vqe_even_fidelity = parity_projected_fidelity(
            ed_even_state, vqe_vector, sector=1
        )
        vqe_odd_fidelity = parity_projected_fidelity(
            ed_state, vqe_vector, sector=-1
        )
        fidelity_with_even_ed = float(abs(np.vdot(ed_even_state, vqe_vector)) ** 2)
        fidelity_plus_against_raw_ed = parity_projected_fidelity(
            ed_state, vqe_vector, sector=1
        )
        ed_energy = float(ed_metadata[field_key]["energy"])
        reconstructed_energy = float(vqe_values["energy_reconstructed"])
        row: dict[str, Any] = {
            "run_index": run_index,
            "field": field,
            "seed": int(vqe_row["seed"]),
            "gpu_name": vqe_row["gpu_name"],
            "vqe_energy_reported": float(vqe_row["final_energy"]),
            "vqe_energy_reconstructed": reconstructed_energy,
            "ed_energy": ed_energy,
            "energy_error_vs_ed": reconstructed_energy - ed_energy,
            "absolute_energy_error_vs_ed": abs(reconstructed_energy - ed_energy),
            "fidelity_with_ed": fidelity,
            "fidelity_with_even_ed_reference": fidelity_with_even_ed,
            "fidelity_after_even_projection": vqe_even_fidelity,
            "fidelity_after_even_projection_against_raw_ed": fidelity_plus_against_raw_ed,
            "fidelity_after_odd_projection": vqe_odd_fidelity,
            "ed_even_projection_weight": float(ed_values["parity_plus_weight"]),
            "vqe": vqe_values,
            "ed": ed_values,
        }
        analysis_rows.append(row)
        print(
            f"{run_index:02d}/30 | h={field:.1f} | seed={vqe_row['seed']} | "
            f"F_ED={fidelity:.6f} | dE={row['energy_error_vs_ed']:.6e}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis_rows, indent=2), encoding="utf-8")
    return analysis_rows
