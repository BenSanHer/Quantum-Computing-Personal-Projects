# Papers and Resources

## Tensor Networks

### Biamonte (2020)
- File: Biamonte-2020-Lectures-on-Quantum-Tensor-Networks.pdf
- Name: Lectures on Quantum Tensor Networks: A Pathway to Modern Diagrammatic Reasoning
- Link: https://arxiv.org/pdf/1912.10049

**Why it matters**
Provides a self-contained theoretical foundation for tensor networks in quantum information, from tensor diagrams and contractions to matrix product states and numerical tensor-network methods.

**Key ideas**
- Penrose graphical notation and tensor contractions
- Matrix product states and matrix product factorization of quantum states
- Tensor-network representations of quantum circuits and quantum processes
- Numerical tensor-network algorithms and software packages

**My notes**
- Chapters I and II are the most relevant starting point for the current MPS-based VQE experiment.
- It connects the diagrammatic language with the linear-algebra operations used in tensor-network simulations.


### Godinez (2025)
- File: Online resource (PennyLane demo)
- Name: Introducing tensor networks for quantum practitioners
- Link: https://pennylane.ai/demos/tutorial_tensor_network_basics#arad2010

**Why it matters**
Offers an accessible, implementation-oriented introduction to tensor networks and connects their basic concepts directly to quantum-circuit simulation with PennyLane's `default.tensor` device.

**Key ideas**
- Tensors as multidimensional generalizations of vectors and matrices
- Tensor contractions, contraction cost, and contraction paths
- Quantum circuits as a special class of tensor networks
- Expectation values, sampling, MPS/MPO representations, and the Quimb backend

**My notes**
- Useful as the practical companion to Biamonte's theoretical lecture notes.
- The examples are directly relevant to the current PennyLane and Quimb MPS workflow.
