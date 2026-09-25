"""GPU tensor-network VQE for the one-dimensional transverse-field Ising model.

The state is stored as an order-N tensor with shape (2,) * N. Every one- or
two-qubit gate is a small tensor contracted with the corresponding state
indices, so the simulation stays in tensor-network form while using PyTorch's
CUDA kernels for the contractions. Twenty spins require only 2**20 complex
amplitudes and fit comfortably on the requested RTX 4050.

Hamiltonian (open boundary conditions):

    H(J, h) = -J sum_i Z_i Z_(i+1) - h sum_i X_i

The ansatz has four rotation blocks. Each block applies RX, RY and RZ to every
spin; the first three blocks are followed by a periodic CNOT ring. It
therefore contains four rotation layers and exactly three CNOT rings, with
4 * 20 * 3 = 240 trainable parameters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


DEFAULT_FIELDS = tuple(round(value, 1) for value in np.arange(0.2, 2.01, 0.2))
DEFAULT_SEEDS = (11, 22, 33)
ROTATION_LAYERS = 4
ENTANGLING_RINGS = 3


def select_device(require_cuda: bool = False) -> torch.device:
    """Select CUDA when present and optionally fail if it is unavailable."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError(
            "CUDA no está disponible. Activa el entorno tensor-networks y "
            "verifica la instalación de PyTorch CUDA."
        )
    return torch.device("cpu")


def device_report(device: torch.device) -> dict[str, Any]:
    """Return a small reproducible report of the selected execution device."""

    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1024**3, 3
        )
    else:
        report["gpu_name"] = None
        report["gpu_memory_gb"] = None
    return report


def _pauli(axis: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a Pauli matrix in the complex dtype used by the state tensor."""

    complex_dtype = dtype if dtype.is_complex else torch.complex64
    if axis == "x":
        values = [[0, 1], [1, 0]]
    elif axis == "y":
        values = [[0, -1j], [1j, 0]]
    elif axis == "z":
        values = [[1, 0], [0, -1]]
    else:
        raise ValueError(f"Unknown Pauli axis: {axis}")
    return torch.tensor(values, device=device, dtype=complex_dtype)


def rotation_gate(theta: torch.Tensor, axis: str) -> torch.Tensor:
    """Construct the differentiable single-qubit rotation exp(-i theta P / 2)."""

    identity = torch.eye(2, device=theta.device, dtype=torch.complex64)
    pauli = _pauli(axis, theta.device, torch.complex64)
    half_theta = theta / 2
    return torch.cos(half_theta) * identity - 1j * torch.sin(half_theta) * pauli


def cnot_gate(device: torch.device) -> torch.Tensor:
    """Return a CNOT tensor whose first wire is the control."""

    gate = torch.zeros((2, 2, 2, 2), device=device, dtype=torch.complex64)
    # Tensor indices are [control_out, target_out, control_in, target_in].
    gate[0, 0, 0, 0] = 1
    gate[0, 1, 0, 1] = 1
    gate[1, 1, 1, 0] = 1
    gate[1, 0, 1, 1] = 1
    return gate


def apply_gate_tensor_network(
    state: torch.Tensor, gate: torch.Tensor, wires: Iterable[int]
) -> torch.Tensor:
    """Contract a one- or two-qubit gate into an order-N state tensor.

    state has one physical index per spin. The requested gate indices are
    moved to the front, contracted as a small matrix multiplication, and then
    moved back to their original positions. This is the tensor-network core
    of the simulator and works for adjacent and wrap-around ring gates.
    """

    selected_wires = tuple(int(wire) for wire in wires)
    if len(selected_wires) not in (1, 2):
        raise ValueError("Only one- and two-qubit gates are supported")
    if len(set(selected_wires)) != len(selected_wires):
        raise ValueError("Gate wires must be distinct")
    if any(wire < 0 or wire >= state.ndim for wire in selected_wires):
        raise ValueError("Gate wire is outside the state tensor")

    remaining_wires = [wire for wire in range(state.ndim) if wire not in selected_wires]
    permutation = list(selected_wires) + remaining_wires
    inverse_permutation = [permutation.index(axis) for axis in range(state.ndim)]
    local_dimension = 2 ** len(selected_wires)

    state_matrix = state.permute(permutation).reshape(local_dimension, -1)
    gate_matrix = gate.reshape(local_dimension, local_dimension)
    updated_matrix = gate_matrix @ state_matrix
    updated = updated_matrix.reshape((2,) * state.ndim)
    return updated.permute(inverse_permutation).contiguous()


def initial_zero_state(num_spins: int, device: torch.device) -> torch.Tensor:
    """Create |00...0> as an order-N tensor on the selected device."""

    state = torch.zeros((2,) * num_spins, device=device, dtype=torch.complex64)
    state[(0,) * num_spins] = 1.0 + 0.0j
    return state


def ansatz_state(
    parameters: torch.Tensor,
    num_spins: int,
    rotation_layers: int = ROTATION_LAYERS,
    entangling_rings: int = ENTANGLING_RINGS,
) -> torch.Tensor:
    """Build the variational state using four rotation blocks and three rings."""

    if tuple(parameters.shape) != (rotation_layers, num_spins, 3):
        raise ValueError(
            "parameters must have shape "
            f"({rotation_layers}, {num_spins}, 3), got {tuple(parameters.shape)}"
        )
    if entangling_rings > rotation_layers - 1:
        raise ValueError("There cannot be more CNOT rings than gaps between rotations")

    state = initial_zero_state(num_spins, parameters.device)
    cnot = cnot_gate(parameters.device)

    for layer in range(rotation_layers):
        for spin in range(num_spins):
            state = apply_gate_tensor_network(
                state, rotation_gate(parameters[layer, spin, 0], "x"), (spin,)
            )
            state = apply_gate_tensor_network(
                state, rotation_gate(parameters[layer, spin, 1], "y"), (spin,)
            )
            state = apply_gate_tensor_network(
                state, rotation_gate(parameters[layer, spin, 2], "z"), (spin,)
            )

        if layer < entangling_rings:
            for spin in range(num_spins - 1):
                state = apply_gate_tensor_network(state, cnot, (spin, spin + 1))
            # The last CNOT closes the periodic ansatz ring.
            state = apply_gate_tensor_network(state, cnot, (num_spins - 1, 0))

    # Unitary gates preserve the norm; this protects the energy against tiny
    # accumulated round-off errors during long optimization runs.
    return state / torch.linalg.vector_norm(state)


def _marginal(probabilities: torch.Tensor, wires: tuple[int, ...]) -> torch.Tensor:
    remaining = tuple(axis for axis in range(probabilities.ndim) if axis not in wires)
    if remaining:
        return probabilities.sum(dim=remaining)
    return probabilities


def expectation_zz(probabilities: torch.Tensor, first: int, second: int) -> torch.Tensor:
    """Compute <Z_first Z_second> from the probability tensor."""

    marginal = _marginal(probabilities, (first, second))
    return marginal[0, 0] - marginal[0, 1] - marginal[1, 0] + marginal[1, 1]


def expectation_x(state: torch.Tensor, wire: int) -> torch.Tensor:
    """Compute <X_wire> by contracting the state with a flipped index."""

    flipped = torch.flip(state, dims=(wire,))
    return torch.real(torch.sum(torch.conj(state) * flipped))


def tfim_energy(
    state: torch.Tensor, coupling: float, field: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return energy, mean <X>, and mean nearest-neighbour <ZZ>."""

    probabilities = state.abs().square()
    num_spins = state.ndim
    zz_values = torch.stack(
        [expectation_zz(probabilities, spin, spin + 1) for spin in range(num_spins - 1)]
    )
    x_values = torch.stack([expectation_x(state, spin) for spin in range(num_spins)])
    energy = -coupling * zz_values.sum() - field * x_values.sum()
    return energy.real, x_values.mean(), zz_values.mean()


def run_vqe(
    *,
    num_spins: int = 20,
    coupling: float = 1.0,
    field: float = 0.2,
    seed: int = 11,
    adam_steps: int = 500,
    lbfgs_steps: int = 80,
    learning_rate: float = 0.03,
    initial_parameters: np.ndarray | torch.Tensor | None = None,
    device: torch.device | None = None,
    rotation_layers: int = ROTATION_LAYERS,
    entangling_rings: int = ENTANGLING_RINGS,
) -> dict[str, Any]:
    """Run one VQE using Adam followed by LBFGS."""

    if num_spins < 2:
        raise ValueError("num_spins must be at least 2")
    if coupling <= 0:
        raise ValueError("coupling must be positive")
    if field <= 0:
        raise ValueError("field must be positive")
    if adam_steps < 0 or lbfgs_steps < 0:
        raise ValueError("adam_steps and lbfgs_steps must be nonnegative")
    if device is None:
        device = select_device()

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    expected_shape = (rotation_layers, num_spins, 3)
    warm_start_used = initial_parameters is not None
    if initial_parameters is None:
        parameters = 0.08 * torch.randn(
            expected_shape, device=device, dtype=torch.float32
        )
    else:
        parameters = torch.as_tensor(
            initial_parameters, device=device, dtype=torch.float32
        ).clone()
        if tuple(parameters.shape) != expected_shape:
            raise ValueError(
                f"initial_parameters must have shape {expected_shape}, "
                f"got {tuple(parameters.shape)}"
            )
    parameters = parameters.requires_grad_()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    with torch.no_grad():
        initial_state = ansatz_state(
            parameters.detach(), num_spins, rotation_layers, entangling_rings
        )
        initial_energy, _, _ = tfim_energy(initial_state, coupling, field)

    history: list[float] = []
    gradient_history: list[float] = []
    adam_optimizer = torch.optim.Adam([parameters], lr=learning_rate)
    for _ in range(adam_steps):
        adam_optimizer.zero_grad(set_to_none=True)
        state = ansatz_state(parameters, num_spins, rotation_layers, entangling_rings)
        energy, _, _ = tfim_energy(state, coupling, field)
        energy.backward()
        gradient_norm = float(parameters.grad.detach().norm().cpu())
        torch.nn.utils.clip_grad_norm_([parameters], max_norm=5.0)
        adam_optimizer.step()
        history.append(float(energy.detach().cpu()))
        gradient_history.append(gradient_norm)

    lbfgs_evaluations = 0
    if lbfgs_steps:
        lbfgs_optimizer = torch.optim.LBFGS(
            [parameters],
            lr=1.0,
            max_iter=lbfgs_steps,
            max_eval=max(2 * lbfgs_steps, 1),
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal lbfgs_evaluations
            lbfgs_optimizer.zero_grad(set_to_none=True)
            state = ansatz_state(
                parameters, num_spins, rotation_layers, entangling_rings
            )
            energy, _, _ = tfim_energy(state, coupling, field)
            energy.backward()
            gradient_norm = float(parameters.grad.detach().norm().cpu())
            history.append(float(energy.detach().cpu()))
            gradient_history.append(gradient_norm)
            lbfgs_evaluations += 1
            return energy

        lbfgs_optimizer.step(closure)

    with torch.no_grad():
        final_state = ansatz_state(
            parameters.detach(), num_spins, rotation_layers, entangling_rings
        )
        final_energy, mean_x, mean_zz = tfim_energy(final_state, coupling, field)
        final_norm = torch.linalg.vector_norm(final_state)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    maximum_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )

    result: dict[str, Any] = {
        "num_spins": num_spins,
        "coupling_J": coupling,
        "transverse_field_h": field,
        "hamiltonian": "H = -J sum_i Z_i Z_(i+1) - h sum_i X_i (open chain)",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "rotation_layers": rotation_layers,
        "entangling_cnot_rings": entangling_rings,
        "trainable_parameters": int(parameters.numel()),
        "optimizer": "Adam + LBFGS",
        "seed": seed,
        "adam_steps": adam_steps,
        "lbfgs_steps": lbfgs_steps,
        "lbfgs_evaluations": lbfgs_evaluations,
        "optimization_steps": adam_steps + lbfgs_steps,
        "learning_rate": learning_rate,
        "warm_start_used": warm_start_used,
        "initial_energy": float(initial_energy.cpu()),
        "final_energy": float(final_energy.cpu()),
        "best_energy": min(history),
        "energy_drop": float((initial_energy - final_energy).cpu()),
        "mean_X_final": float(mean_x.cpu()),
        "mean_ZZ_final": float(mean_zz.cpu()),
        "final_state_norm": float(final_norm.cpu()),
        "runtime_seconds": elapsed,
        "maximum_gpu_memory_mb": float(maximum_memory_mb),
        "history": history,
        "gradient_norm_history": gradient_history,
        "final_parameters": parameters.detach().cpu().numpy().tolist(),
    }
    return result


def run_grid(
    *,
    fields: Iterable[float] = DEFAULT_FIELDS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    num_spins: int = 20,
    coupling: float = 1.0,
    first_field_adam_steps: int = 500,
    later_field_adam_steps: int = 150,
    lbfgs_steps: int = 80,
    learning_rate: float = 0.03,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Run a warm-start field sweep with three VQEs per field.

    The first field gets 500 Adam steps. Each following field gets 150 Adam
    steps and reuses the final parameters from the same seed at the previous
    field. Every run then receives up to 80 LBFGS iterations.
    """

    field_values = [float(field) for field in fields]
    seed_values = [int(seed) for seed in seeds]
    if len(seed_values) != 3:
        raise ValueError("Provide exactly three seeds for the requested three VQEs")

    results: list[dict[str, Any]] = []
    previous_parameters: dict[int, list[list[list[float]]]] = {}
    for field_index, field in enumerate(field_values):
        adam_steps = (
            first_field_adam_steps
            if field_index == 0
            else later_field_adam_steps
        )
        for run_index, seed in enumerate(seed_values, start=1):
            warm_start = previous_parameters.get(seed)
            source_field = field_values[field_index - 1] if warm_start is not None else None
            print(
                f"h={field:.1f} | VQE {run_index}/3 | seed={seed} | "
                f"Adam={adam_steps} | LBFGS={lbfgs_steps} | "
                f"warm_start={source_field}"
            )
            result = run_vqe(
                num_spins=num_spins,
                coupling=coupling,
                field=field,
                seed=seed,
                adam_steps=adam_steps,
                lbfgs_steps=lbfgs_steps,
                learning_rate=learning_rate,
                initial_parameters=warm_start,
                device=device,
            )
            result["field_sweep_index"] = field_index
            result["warm_start_source_field"] = source_field
            previous_parameters[seed] = result["final_parameters"]
            results.append(result)
    return results


def save_results(results: Any, path: str | Path) -> Path:
    """Write one result or a grid of results as formatted JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output_path
