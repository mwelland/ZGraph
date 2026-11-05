# ZGraph — Paper 1 Alignment Status & Open TODOs

This document tracks the reconciliation between `Zgraph` and the foundational theory manuscript (`paper1_zgraph_engine`) so work can be seamlessly resumed by any local or remote agent.

---

## Completed in Phase 1

1. **Unified Scale Parameter (`tau`)**:
   - Replaced `beta` with `tau` everywhere across `src/zgraph/`, `tests/`, `examples/`, and docs. No `beta` aliases remain.
2. **Log-Degeneracy Vector $\boldsymbol{g}$ (`g` in `FactorNode`)**:
   - Added `g: Optional[jax.Array] = None` to `TropicalPolynomialNode` (`FactorNode`), implementing Paper Eq. (2.2):
     $$z^+(\mathbf{z}) = \tau \, \text{LSE}\!\left(\tau^{-1} M \mathbf{z} + \boldsymbol{g}\right)$$
   - Guarded `+`, `-`, `*`, and `|` operator fusion in `core/base.py` so nodes with $g \neq \text{None}$ or non-unit $\tau$ are never flattened improperly.
3. **Branch-Free Zero-Scale ($\tau = 0$) Limit in `marginalize`**:
   - Removed `_MIN_BETA = 1e-4` pre-clamping on `TropicalPolynomialNode` and `TropicalMatMulNode`.
   - Implemented the 6-line branch-free shifted LSE in `core/functional.py:marginalize` with `safe_tau = jnp.maximum(tau, MIN_SAFE_TAU)` (`MIN_SAFE_TAU = 1e-12`) in the denominator and exact `tau` multiplying the outer `jnp.log(sum_exp)`.
   - *Technical note:* `MIN_SAFE_TAU = 1e-12` is chosen so $1 / \tau_{\text{safe}}^3 = 10^{36} < 3.4 \times 10^{38}$ (`float32` max), ensuring 1st- and 2nd-order autodiff wrt `tau` at $\tau = 0$ stay finite (`0.0` rather than `0.0 * inf = NaN`).
4. **Tropical Zero Barrier (`TROPICAL_ZERO_BARRIER = -1e30`)**:
   - Changed `TROPICAL_ZERO_BARRIER` in `core/constants.py` from `+1e9` to `-1e30` (and added `-inf` guard in `marginalize`).
   - *Technical note:* Using a finite barrier `-1e30` rather than IEEE `float("-inf")` avoids `0.0 * (-inf) = NaN` when multiplying sparse incidence matrices `self.M @ x` in forward and reverse-mode autodiff.
5. **Standard Convex Fenchel Sign Convention**:
   - Updated `transforms/functional.py:apply_legendre` to compute $z^*(\mathbf{n}) = \mathbf{z}^T \mathbf{n} - z^+(\mathbf{z})$ (Paper Eq. 3.9–3.10).
   - `LegendreTransform` now inherits directly from `ZGraphNode`.
6. **Removed Legacy `graph_to_function` (`transforms/compiler.py`)**:
   - Because all nodes (including `LegendreTransform` and `LegendreDualNode`) now inherit from `ZGraphNode`, `ZGraphNode.__call__` automatically handles arbitrary batch dimensions and `Ensemble` dimensions.
   - To JIT-compile a node or list of nodes, use `eqx.filter_jit(node)` or `[eqx.filter_jit(f) for f in fcns]`. (Note: do not call `jax.jit(node)` directly on an `eqx.Module` without `eqx.filter_jit`, as `jax.jit` attempts to hash `jax.Array` fields).
7. **Inverse Legendre-Fenchel Solver (`legendre_solve` & `LegendreDualNode`)**:
   - Added `solvers/dual.py:legendre_solve` and `transforms/algebraic.py:LegendreDualNode` (`legendre_dual`), using damped SVD-pseudoinverse (`jnp.linalg.pinv`) Newton-Raphson inside `jax.lax.while_loop` with `@jax.custom_jvp` implicit differentiation and optional `anchor_gauge=True`.
   - *Technical note:* All dynamic index scatter updates `.at[idx].set(...)` inside custom JVP rules must pass `unique_indices=True` so JAX can transpose them during `jax.grad`.
8. **General Directional Gauge Projection (`gauge_fix`)**:
   - Updated `solvers/gauge.py:gauge_fix` and `solvers/functional.py:apply_gauge_shift` to compute directional coupling $c_u = \mathbf{u}^T \nabla_{\mathbf{z}} z^+(\mathbf{z})$ in a single `jax.jvp` pass and shift by $(z_{\text{target}} - z^+(\mathbf{z})) / c_u$, supporting both `shift_indices` and an explicit `direction` vector $\mathbf{u}$.
9. **Precision Preservation & Serialization**:
   - Removed hardcoded `.astype(jnp.float32)` casts across `core/` so `jax_enable_x64` is respected.
   - Fixed Windows drive-letter colon parsing (`os.path.splitdrive`) in `io.py:load` and kept `pyproject.toml` dependencies strictly `["jax", "jaxlib", "equinox"]`.

---

## Open Design Items & Loose Ends (Phase 2)

### 1. Continuous Sub-Grid Decision Boundary Refinement (Paper Sec. 3.4 & 4.3.2)
- **Files:** `src/zgraph/solvers/boundaries.py`, `src/zgraph/solvers/functional.py`
- **Context:** `extract_decision_boundary` currently finds the discrete sweep grid index `boundary_idx = jnp.argmin(jnp.abs(delta_omega), axis=-1)` where the top two branches cross. Because `argmin` is piecewise constant, the boundary is quantized to the grid spacing and its gradient wrt model parameters (`jax.grad`) is zero.
- **Paper Formulation:**
  - Given the discrete bracket along sweep direction $\mathbf{d}$, refine the root of $f(\alpha) = z^+_{i_1}(\mathbf{z}_0 + \alpha \mathbf{d}) - z^+_{i_2}(\mathbf{z}_0 + \alpha \mathbf{d}) = 0$ via 1D Newton-Raphson inside `jax.lax.while_loop`:
    $$\alpha_{t+1} = \alpha_t - \frac{z^+_{i_1}(\mathbf{z}_t) - z^+_{i_2}(\mathbf{z}_t)}{(\mathbf{n}_{i_1}(\mathbf{z}_t) - \mathbf{n}_{i_2}(\mathbf{z}_t))^T \mathbf{d}}$$
  - Attach a `@jax.custom_jvp` rule applying the Implicit Function Theorem on the equal-potential manifold (Paper Eq. 3.16):
    $$\frac{\partial \alpha^*}{\partial \theta} = -\frac{\frac{\partial z^+_{i_1}}{\partial \theta} - \frac{\partial z^+_{i_2}}{\partial \theta}}{(\mathbf{n}_{i_1} - \mathbf{n}_{i_2})^T \mathbf{d}}$$
- **Decision Needed:** Whether to add `refine: bool = True` to `extract_decision_boundary` and/or expose a standalone `project_to_boundary` solver.

### 2. Vector `SignalNode` vs. Direct Vector Evaluation (`subgraph_list=None`) (Paper Sec. 4.1.3)
- **Files:** `src/zgraph/core/leaf_nodes.py`, `src/zgraph/core/tropical_nodes.py`
- **Context:** Currently, constructing a leaf `FactorNode` over an $N$-dimensional input vector $\mathbf{z}$ requires creating $N$ scalar `SignalNode(0), ..., SignalNode(N-1)` objects, and `TropicalPolynomialNode.logits` runs a Python list comprehension `[subgraph(signals) for subgraph in self.subgraphs]` followed by `jnp.stack`.
- **Options to Discuss:**
  - **Option A:** Allow `SignalNode` to accept a slice/list of indices (or a vector `SignalNode` exception to the scalar node pattern) so a single node can pass a vector of signals.
  - **Option B:** Allow `subgraph_list=None` on `FactorNode(M_matrix)`, which directly evaluates `self.M @ signals` without any `SignalNode` wrapper.
  - **Option C:** Support both or keep explicit scalar `SignalNodes`.

### 3. Constrained Parameterization Wrappers for Training (Paper Sec. 4.1.4)
- **Context:** During gradient-based parameter estimation (e.g., with Optax), $M$ and $\boldsymbol{g}$ are unconstrained real arrays, whereas convexity requires $\tau \ge 0$ (and an Input-Convex Neural Network / monotonicity prior requires $M_{ij} \ge 0$). Paper Sec. 4.1.4 describes parameterizing trainable $\tau = \text{Softplus}(\tilde{\tau})$ or $\exp(\tilde{\tau})$.
- **Decision Needed:** Should constrained parameterization helpers live inside `zgraph` (e.g., a `zgraph.nn` / `zgraph.optim` module or a `FactorNode` option), or belong in the downstream training/application package?

### 4. Multi-Subgraph `Ensemble` Resampling (Paper Sec. 4.4.3)
- **Files:** `src/zgraph/core/base.py`
- **Context:** `ZGraphNode._execute_vectorized` applies a single `eqx.filter_vmap` over axis `0` of every `Ensemble` leaf in a DAG, requiring all `Ensemble` leaves to have the same sample count $S$. When combining independently calibrated subgraphs with different ensemble sizes ($S_A \neq S_B$), each subgraph's `Ensemble` leaves must be resampled to a common $S$ (sharing the same random index draw within a subgraph to preserve intra-subgraph parameter covariance, while drawing independent indices across distinct subgraphs).
- **Decision Needed:** Should this be an explicit utility `zg.align_ensembles(graph, num_samples=None, key=...)` (defaulting `num_samples` to `max(S_i)` across the graph), an automatic behavior in `ZGraphNode` / compiler, or part of the application layer?

### 5. Paper Section 5 Benchmarks (Deferred)
- **Context:** Paper Section 5 defines four verification benchmarks (`P1_sec5_draft.md`):
  1. Benchmark 1 (Sec. 5.1): 3-level soft-tropical DAG verifying machine-precision gradients ($\nabla_{\mathbf{z}} z^+$, $\partial z^+/\partial \tau$), PSD Hessians ($\mathbf{H}_{\mathbf{z}} \succeq 0$), Euler homogeneity residual $< 10^{-14}$, and exact convergence to $\tau = 0$.
  2. Benchmark 2 (Sec. 5.2): High-dimensional ($N=50, K=200$) Legendre-Fenchel inversion via `legendre_solve` and validation of implicit derivatives ($\nabla_{\mathbf{n}} z^* = \mathbf{z}^*$, $\nabla^2_{\mathbf{n}} z^* = \mathbf{H}_{\mathbf{z}}^\dagger$) against finite differences.
  3. Benchmark 3 (Sec. 5.3): Inverse parameter recovery (`M`, `g`, `tau`) from noisy marginal observations $\hat{\mathbf{n}}$ and decision-boundary points via IFT gradients.
  4. Benchmark 4 (Sec. 5.4): Forward epistemic `Ensemble` propagation vs. analytical linear covariance $\Sigma_n = \mathbf{J}_\theta \Sigma_\theta \mathbf{J}_\theta^T$.
- **Status:** Intentionally deferred until Phase 2 design items are finalized.
