# ZGraph

[![Unit Tests](https://github.com/themintlab/ZGraph/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/themintlab/ZGraph/actions/workflows/unit-tests.yml)

**A hardware-accelerated, fully differentiable, physics-free soft-tropical sum-product graph engine.**

`zgraph` is a low-level, high-performance numerical kernel engine built natively on JAX and Equinox (`jax.jit`, `jax.vmap`, `jax.grad`). It compiles hierarchical factor graph contractions, soft-tropical marginalizations, and convex optimization problems into homogeneous chains of parameterized $\text{LogSumExp}$ ($\text{LSE}$) matrix operations.

---

## Core Philosophy: Physics-Free Pure Math

`zgraph` is **strictly domain-agnostic and physics-free**.

The engine operates exclusively on unlabeled numerical tensors (`jax.Array`). It contains zero domain-specific logic, zero unit conversions, and zero Python object overhead (such as string matching or dictionary lookups inside loops).

All physical nomenclature, thermodynamic ensembles, chemical potential mappings ($\mu$), state variables ($T, P$), phase definitions, and human-readable metadata are relegated to higher-level application layers such as [**ThermoGraph**](https://github.com/themintlab/ThermoGraph).

### Separation of Concerns

| Feature / Responsibility | **`zgraph`** (Math Engine) | **`thermograph`** (Application Layer) |
| :--- | :--- | :--- |
| **Domain Scope** | Abstract factor graphs, soft-tropical semirings | Thermodynamics, statistical mechanics, CALPHAD |
| **Data Types** | Unlabeled JAX Arrays (`jax.Array`) | Named variables, species, phases, physical units |
| **Core Operations** | Incidence projection, parameterized $\text{LogSumExp}$ | Ensemble definitions $(T, P, \mu)$, driving forces |
| **Optimization Target** | `jax.jit`, XLA compilation, `jax.vmap` vectorization | Phase equilibrium, convex hulls, phase-field coupling |
| **Graph Mutations** | Structurally immutable execution graphs | Dynamic model assembly & configuration |

---

## Key Pillars

### 1. Physics-Free Pure Tensor Math
By enforcing pure tensor-only signatures (`__call__(signals: Array) -> Array`), `zgraph` guarantees zero graph breaks during JIT compilation. Nodes do not store string names, metadata dicts, or conditional Python logic.

### 2. Soft-Tropical Sum-Product Algebra
Every node maps incoming child log-potentials $\mathbf{z} \in \mathbb{R}^N$ via incidence matrix $M \in \mathbb{R}^{K \times N}$, log-degeneracy bias $\boldsymbol{g} \in \mathbb{R}^K$, and scale parameter $\tau \in [0, \infty)$ into a scalar outgoing log-potential $z^+$:
$$z^+(\mathbf{z}) = \tau \, \text{LogSumExp}\!\left( \tau^{-1} M \mathbf{z} + \boldsymbol{g} \right)$$
where $\tau$ controls the smoothing transition between exact hard-tropical selection ($\tau \to 0^+$, $\max_i (M\mathbf{z})_i$) and smooth statistical marginalization ($\tau > 0$).

### 3. Hardware-Native Execution
Engine nodes (`FactorNode`, `ConstantNode`, `SignalNode`) are engineered to compile seamlessly via `jax.jit` into fused XLA kernels. Vectorization across spatial grids or multi-dimensional parameter ensembles (`Ensemble`) is handled automatically via `vmap`.

### 4. End-to-End Differentiability & Legendre-Fenchel Duality
Built for direct integration with JAX's autodiff engine, `zgraph` provides exact analytical marginals and susceptibilities ($\mathbf{n} = \nabla_{\mathbf{z}} z^+$, $\mathbf{H}_{\mathbf{z}} = \nabla^2_{\mathbf{z}} z^+$) as well as both forward parametric (`legendre_transform`) and iterative inverse (`legendre_solve`, `legendre_dual`) Legendre-Fenchel duality:
$$z^*(\mathbf{n}) = \sup_{\mathbf{z}} \left\{ \mathbf{z}^T \mathbf{n} - z^+(\mathbf{z}) \right\}$$

---

## Engine Usage Example

At the `zgraph` level, models are built using purely mathematical node abstractions:

```python
import jax.numpy as jnp
import zgraph as zg

# 1. Define incidence matrix M (K branches x N child factors)
M_matrix = jnp.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5]
])

# 2. Define child subgraphs (Constant / Signal nodes)
subgraphs = [
    zg.ConstantNode(init_val=1.5),
    zg.SignalNode(0)
]

# 3. Instantiate the FactorNode engine with scale tau and optional log-degeneracy g
factor_node = zg.FactorNode(M_matrix=M_matrix, subgraph_list=subgraphs, tau=1.0)

# 4. Pure tensor input evaluation
signals = jnp.array([0.2, 1.0])  # Unlabeled numerical input tensor
potential = factor_node(signals) # Fully differentiable scalar tensor output
```

## Interactive Examples

Explore interactive Jupyter notebooks running natively on Google Colab:

- **2D Binary Equilibrium** (`binary.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ZGraph/blob/main/examples/binary.ipynb)
- **3D Extruded Dimension** (`Extruded_dimension.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ZGraph/blob/main/examples/Extruded_dimension.ipynb)
- **Stoichiometric Equilibrium** (`stoichiometric.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ZGraph/blob/main/examples/stoichiometric.ipynb)
- **Simple Phase Diagram** (`phase_diagram.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ZGraph/blob/main/examples/phase_diagram.ipynb)

---

## Application Layer

For thermodynamic potential evaluation, materials modeling, phase diagram construction, and user-facing APIs, please see the [**ThermoGraph**](https://github.com/themintlab/ThermoGraph) repository.
