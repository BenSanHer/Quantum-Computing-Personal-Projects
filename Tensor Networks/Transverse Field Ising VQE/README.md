# Critical transverse-field Ising VQE with an MPS backend

This experiment runs a ten-qubit variational quantum eigensolver using
PennyLane's `default.tensor` device and Quimb's matrix-product-state method.

The open-boundary Hamiltonian is

```text
H = J sum_i Z_i Z_(i+1) - h sum_i X_i,
```

with `J = h = 1`. This is the antiferromagnetic convention, chosen because the
requested initial state `|0101010101>` is its ordered product state. The ratio
`h/J = 1` is the critical point in the thermodynamic limit; a ten-site chain
shows a finite-size crossover rather than a mathematically sharp transition.

The ansatz alternates period-two `RY` rotations and nearest-neighbour `IsingZZ`
gates. Its gates follow the one-dimensional MPS ordering. COBYLA optimizes the
parameters without requiring a dense state vector or differentiating through
the tensor-network contractions.

## Run

```powershell
conda activate tensor-networks
python vqe_tfim_mps.py
```

The defaults use six layers, 18 parameters, at most 400 energy evaluations, an
MPS bond-dimension ceiling of 64, and an SVD cutoff of `1e-12`.

The script writes `results/vqe_results.json` and
`results/vqe_convergence.png`. It also calculates the exact ten-qubit ground
energy with a sparse matrix as a validation reference; the VQE energy itself is
always evaluated using the MPS backend.

## Didactic notebook

Open `vqe_tfim_mps_tutorial.ipynb` from JupyterLab in the `tensor-networks`
environment. The Spanish-language lesson explains the model, critical point,
Néel initial state, variational ansatz, MPS representation, convergence, and
observables. It loads the validated 400-evaluation run by default; set
`RUN_FULL_OPTIMIZATION = True` in the notebook to repeat the optimization.

Run the unit tests with:

```powershell
python -m unittest -v test_vqe_tfim_mps.py
```
