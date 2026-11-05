# ZGraph Architecture & Design Philosophy

`zgraph` is a **Soft-Tropical Sum-Product Network (Factor Graph)** engine designed for differentiable belief propagation and convex optimization. It executes fused matrix-vector `LogSumExp` contractions, exact higher-order automatic differentiation, and Legendre-Fenchel duality using JAX and Equinox.

To ensure speed and portability across platforms and hardware architectures, all code within this package MUST adhere to the following strict architectural directives. This ensures robust compilation via `jax.jit` and seamless compatibility with `jax`.

`zgraph` is purely mathematical and has no knowledge of the application domain. All labelling of variables, modules, and physical significance (e.g., mapping gradients to CALPHAD composition tie-lines) are relegated to independent domain packages (like `thermograph`).

---

## Architectural Division

The `zgraph` codebase is strictly divided into three distinct categories based on their mathematical role:

1. **`core/` (The Soft-Tropical Forward Pass)**
   Contains the structural math primitives (`FactorNode`, `ProductNode`, `marginalize`). These define the topology of the graphical model:
   $$z^+(\mathbf{z}) = \tau \, \text{LSE}\!\left(\tau^{-1} M \mathbf{z} + \boldsymbol{g}\right)$$
   They execute the `evaluate()` pass (wrapped by `__call__()`) to compute either the collapsed scalar potential $z^+$ or the uncollapsed branch potentials $z_i^+ + \tau g_i$ via `.logits()`.

2. **`transforms/` (Graph-to-Graph Operators)**
   Contains functional operators (`legendre_transform`, `legendre_dual`) that reshape the topology or coordinates. A transform takes a graph (`eqx.Module`) and returns a *new* `ZGraphNode` graph.

3. **`solvers/` (Execution Routines)**
   Contains numerical and analytic search functions (`extract_decision_boundary`, `gauge_fix`, `legendre_solve`). Solvers take a **callable function** (e.g., a model or `.logits` method) and an **input domain tensor**, and execute searches or projections (finding roots, branch crossovers, conjugate potentials, or projecting onto level-sets $z^+(\mathbf{z}) = 0$).

---

## Core Design Principles

To prevent breaking JAX/Equinox native `vmap` and `jax.jit` compatibility, all contributors must strictly adhere to the following rules:

1. **Pure Math, No Python Objects:**
   **No standard Python objects (strings, lists of strings, dictionaries) or Python control flow (`if` statements based on string matching or traced values) may exist inside `zgraph` `eqx.Module` classes or their `evaluate()` passes.**
   - *Reason:* `jax.jit` requires strict static typing. Dictionaries or string parsing cause graph breaks and kernel compilation failures. All domain knowledge (names, metadata) must remain in the application layer.

2. **Structural Immutability:**
   `zgraph` routing nodes (e.g., `FactorNode`) are structurally immutable after creation. You may **NOT** include methods that mutate the underlying `list` or `tuple` (within `eqx.Module`) or swap subgraphs in-place (e.g., `self.subgraphs[2] = new_model`).
   - *Reason:* Swapping nodes in-place corrupts JAX's tracer graph and invalidates fused XLA kernels.

3. **GPU & Device Management (The Buffer Rule):**
   - **Static indices and configuration flags** MUST be registered using `eqx.field(static=True)` or as integer/float JAX arrays.
   - **Learnable constants** must use standard JAX arrays (Equinox treats all floating-point JAX arrays as trainable parameters unless marked static).
   - **Never** store lists of integers or floats as raw attributes (e.g., `self.indices = [0, 1]`) if they are indexed inside the `evaluate()` pass.

4. **Tensor-Only Communication:**
   All inputs and outputs between `zgraph` modules must be `jax.Array` types (or tuples of `jax.Array` for multi-output transforms like `LegendreTransform`).

5. **Computational Efficiency (No Python Loops over Batches/States):**
   **Never** use Python `for` or `while` loops over spatial dimensions, batches, or branches inside an `evaluate()` pass. All operations must be vectorized using native JAX tensor operations, broadcasting, or `vmap`.

6. **Clean Code & Strict Typing:**
   All functions and methods must use explicit type hints from the built-in `typing` module.

---

## 4. The Functor Pattern (`ZGraphNode`)

To maximize both code readability and hardware performance, `zgraph` implements a strict Template Method design pattern via the `ZGraphNode` base class.

* **Pristine Subclasses:** Subclasses (`FactorNode`, `ProductNode`, `LegendreTransform`, `LegendreDualNode`) must NEVER implement `__call__`. Instead, they implement their pure, unvectorized scalar mathematical logic inside `evaluate(self, signals)`.
* **The Interceptor:** The `ZGraphNode` base class takes absolute ownership of `__call__`. When a user evaluates `system(signals)`, the base class automatically intercepts the execution, detects batch dimensions or uncertainty parameters (`Ensemble`), constructs the necessary JAX masks, wraps `evaluate` in `jax.vmap` and `eqx.filter_vmap`, and executes the hardware-optimized kernel.

## 5. Universal Uncertainty Quantification (UQ)

ZGraph treats aleatoric fluctuations (scale $\tau$) and epistemic parameter uncertainty (`Ensemble`) distinctly:
* **Aleatoric / Entropic:** Handled natively inside the core scalar logic via exact `LogSumExp` partition reductions at scale $\tau$.
* **Epistemic:** Executed externally via vectorized Monte Carlo parameter ensembles mapped over the graph via JAX.

### The `Ensemble` PyTree Wrapper
To distinguish parameter uncertainty distributions from vector constants (such as incidence weights $M$ or degeneracy biases $\boldsymbol{g}$), uncertainty parameters are wrapped in `Ensemble(eqx.Module)` with leading sample dimension $S$.
* The `__call__` interceptor recursively scans the PyTree for `Ensemble` objects and dynamically unwraps them inside `eqx.filter_vmap` before passing raw arrays into the pure `evaluate` method.

### The `.ensemble` BatchProxy
For secondary node methods (such as `.logits(signals)` or `.probabilities(signals)`), the base class provides a `BatchProxy` via the `.ensemble` property:
* **Usage:** `system.ensemble.logits(signals)`
