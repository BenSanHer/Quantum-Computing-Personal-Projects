"""VQE for the critical 1D transverse-field Ising model using an MPS backend.

The requested alternating state ``|0101010101>`` is the natural ordered state
for the antiferromagnetic convention used here:

    H = J * sum(Z_i Z_{i+1}) - h * sum(X_i), with J = h = 1.

Open boundary conditions are used.  The equality h / J = 1 is the critical
point in the thermodynamic limit; for ten sites it is a finite-size crossover.
"""

from __future__ import annotations

import os

# The execution sandbox cannot write Numba's cache inside the Conda
# environment.  Quimb itself works normally without that optional cache.
os.environ.setdefault("QUIMB_NUMBA_CACHE", "OFF")

# Small tensor contractions are usually faster without nested thread pools.
os.environ.setdefault("QUIMB_NUM_THREAD_WORKERS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh


# PennyLane 0.45.1 emits this cosmetic warning while converting BasisState's
# integer bits into Quimb's internal binary string.
warnings.filterwarnings(
    "ignore",
    category=np.exceptions.ComplexWarning,
    module=r"pennylane\.devices\.default_tensor",
)


def alternating_state(num_qubits: int) -> np.ndarray:
    """Return |0101...> as a computational-basis bit array."""
    return np.array([site % 2 for site in range(num_qubits)], dtype=int)


def tfim_hamiltonian(num_qubits: int, coupling: float, field: float) -> qml.Hamiltonian:
    """Construct the open-boundary antiferromagnetic TFIM Hamiltonian."""
    coefficients = [coupling] * (num_qubits - 1) + [-field] * num_qubits
    observables = [
        qml.PauliZ(site) @ qml.PauliZ(site + 1)
        for site in range(num_qubits - 1)
    ]
    observables.extend(qml.PauliX(site) for site in range(num_qubits))
    return qml.Hamiltonian(coefficients, observables)


def apply_ansatz(parameters: np.ndarray, initial_state: np.ndarray) -> None:
    """Apply a shallow, period-two tensor-network-friendly ansatz.

    Every layer has three parameters: one RY angle for even sites, one for odd
    sites, and one nearest-neighbour IsingZZ angle.  Zero parameters leave the
    requested basis state unchanged.
    """
    num_qubits = len(initial_state)
    qml.BasisState(initial_state, wires=range(num_qubits))

    for even_angle, odd_angle, entangling_angle in parameters:
        for site in range(num_qubits):
            angle = even_angle if site % 2 == 0 else odd_angle
            qml.RY(angle, wires=site)

        # Brickwork ordering keeps every applied gate local in the MPS chain.
        for first_site in (0, 1):
            for site in range(first_site, num_qubits - 1, 2):
                qml.IsingZZ(entangling_angle, wires=[site, site + 1])


def exact_ground_energy(num_qubits: int, coupling: float, field: float) -> float:
    """Compute a small sparse-matrix reference value for validation."""
    identity = sparse.identity(2, dtype=complex, format="csr")
    pauli_x = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    pauli_z = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
    dimension = 2**num_qubits
    hamiltonian = sparse.csr_matrix((dimension, dimension), dtype=complex)

    def kron_term(operators: dict[int, sparse.csr_matrix]) -> sparse.csr_matrix:
        result = sparse.csr_matrix([[1.0 + 0.0j]])
        for site in range(num_qubits):
            result = sparse.kron(result, operators.get(site, identity), format="csr")
        return result

    for site in range(num_qubits - 1):
        hamiltonian += coupling * kron_term({site: pauli_z, site + 1: pauli_z})
    for site in range(num_qubits):
        hamiltonian -= field * kron_term({site: pauli_x})

    return float(eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False)[0].real)


def save_convergence_plot(history: list[float], exact_energy: float, path: Path) -> None:
    """Save the best energy reached as a function of circuit evaluations."""
    best_so_far = np.minimum.accumulate(history)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(range(1, len(history) + 1), best_so_far, label="Best MPS-VQE energy")
    axis.axhline(exact_energy, color="tab:red", linestyle="--", label="Exact energy")
    axis.set_xlabel("Circuit evaluations")
    axis.set_ylabel("Energy")
    axis.set_title("Critical 10-qubit TFIM: MPS-VQE convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_vqe(
    *,
    num_qubits: int,
    layers: int,
    coupling: float,
    field: float,
    max_iterations: int,
    max_bond_dimension: int,
    cutoff: float,
    seed: int,
    output_directory: Path,
) -> dict[str, object]:
    initial_state = alternating_state(num_qubits)
    hamiltonian = tfim_hamiltonian(num_qubits, coupling, field)
    device = qml.device(
        "default.tensor",
        wires=num_qubits,
        method="mps",
        max_bond_dim=max_bond_dimension,
        cutoff=cutoff,
        contract="auto-mps",
        contraction_optimizer="greedy",
    )

    @qml.qnode(device, interface=None)
    def energy_circuit(parameters: np.ndarray) -> float:
        apply_ansatz(parameters, initial_state)
        return qml.expval(hamiltonian)

    transverse_magnetization = qml.sum(
        *(qml.PauliX(site) / num_qubits for site in range(num_qubits))
    )
    staggered_magnetization = qml.sum(
        *(
            ((-1) ** site) * qml.PauliZ(site) / num_qubits
            for site in range(num_qubits)
        )
    )

    @qml.qnode(device, interface=None)
    def diagnostic_circuit(parameters: np.ndarray) -> tuple[float, float, float]:
        apply_ansatz(parameters, initial_state)
        return (
            qml.expval(hamiltonian),
            qml.expval(transverse_magnetization),
            qml.expval(staggered_magnetization),
        )

    exact_energy = exact_ground_energy(num_qubits, coupling, field)
    zero_parameters = np.zeros((layers, 3), dtype=float)
    initial_energy = float(energy_circuit(zero_parameters))

    generator = np.random.default_rng(seed)
    starting_parameters = generator.normal(loc=0.0, scale=0.08, size=(layers, 3))
    history: list[float] = []
    start_time = time.perf_counter()

    def objective(flat_parameters: np.ndarray) -> float:
        energy = float(energy_circuit(flat_parameters.reshape(layers, 3)))
        history.append(energy)
        evaluation = len(history)
        if evaluation == 1 or evaluation % 20 == 0:
            print(f"evaluation={evaluation:4d} energy={energy:.10f} best={min(history):.10f}")
        return energy

    optimization = minimize(
        objective,
        starting_parameters.ravel(),
        method="COBYLA",
        options={
            "maxiter": max_iterations,
            "rhobeg": 0.30,
            "tol": 1e-6,
            "catol": 1e-6,
            "disp": False,
        },
    )
    elapsed = time.perf_counter() - start_time
    optimized_parameters = optimization.x.reshape(layers, 3)
    final_energy, transverse_x, staggered_z = map(
        float, diagnostic_circuit(optimized_parameters)
    )
    energy_error = final_energy - exact_energy
    relative_error = abs(energy_error / exact_energy)
    maximum_bond = int(device._quimb_circuit.psi.max_bond())

    results: dict[str, object] = {
        "model": "open-boundary antiferromagnetic transverse-field Ising model",
        "hamiltonian": "H = J sum_i Z_i Z_(i+1) - h sum_i X_i",
        "num_qubits": num_qubits,
        "initial_state": "".join(map(str, initial_state)),
        "coupling_J": coupling,
        "transverse_field_h": field,
        "field_to_coupling_ratio": field / coupling,
        "critical_point_note": "h/J = 1 is critical in the thermodynamic limit",
        "backend": "PennyLane default.tensor, method=mps",
        "layers": layers,
        "variational_parameters": int(optimized_parameters.size),
        "max_bond_dimension_allowed": max_bond_dimension,
        "maximum_bond_dimension_observed": maximum_bond,
        "svd_cutoff": cutoff,
        "optimizer": "COBYLA",
        "optimizer_success": bool(optimization.success),
        "optimizer_message": str(optimization.message),
        "circuit_evaluations": len(history),
        "runtime_seconds": elapsed,
        "initial_energy": initial_energy,
        "final_vqe_energy": final_energy,
        "exact_ground_energy": exact_energy,
        "absolute_energy_error": abs(energy_error),
        "relative_energy_error": relative_error,
        "mean_transverse_magnetization_X": transverse_x,
        "mean_staggered_magnetization_Z": staggered_z,
        "optimized_parameters": optimized_parameters.tolist(),
        "energy_history": history,
        "software": {
            "pennylane": qml.__version__,
        },
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "vqe_results.json"
    plot_path = output_directory / "vqe_convergence.png"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_convergence_plot(history, exact_energy, plot_path)

    print("\nMPS-VQE result")
    print(f"Initial state: |{results['initial_state']}>")
    print(f"Critical parameters: J={coupling:.6g}, h={field:.6g}, h/J={field / coupling:.6g}")
    print(f"Initial energy: {initial_energy:.10f}")
    print(f"Final VQE energy: {final_energy:.10f}")
    print(f"Exact ground energy: {exact_energy:.10f}")
    print(f"Absolute error: {abs(energy_error):.10f}")
    print(f"Relative error: {100 * relative_error:.6f}%")
    print(f"Mean <X>: {transverse_x:.10f}")
    print(f"Mean staggered <Z>: {staggered_z:.10f}")
    print(f"Maximum observed MPS bond dimension: {maximum_bond}")
    print(f"Circuit evaluations: {len(history)}")
    print(f"Optimization runtime: {elapsed:.3f} s")
    print(f"Optimizer status: {optimization.message}")
    print(f"Results: {results_path}")
    print(f"Convergence plot: {plot_path}")
    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=10)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--max-bond", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-directory", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.qubits < 2:
        raise ValueError("At least two qubits are required")
    if arguments.layers < 1:
        raise ValueError("At least one variational layer is required")
    if arguments.coupling == 0:
        raise ValueError("The coupling must be nonzero")

    run_vqe(
        num_qubits=arguments.qubits,
        layers=arguments.layers,
        coupling=arguments.coupling,
        field=arguments.field,
        max_iterations=arguments.max_iterations,
        max_bond_dimension=arguments.max_bond,
        cutoff=arguments.cutoff,
        seed=arguments.seed,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
