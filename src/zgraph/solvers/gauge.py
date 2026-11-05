import jax
import jax.numpy as jnp
from typing import Callable, List, Optional, Sequence, Union
import zgraph.solvers.functional as F


def gauge_fix(
    compiled_model_fn: Callable[[jax.Array], jax.Array],
    sweep_signals: jax.Array,
    shift_indices: Optional[Sequence[int]] = None,
    target_val: float = 0.0,
    *,
    direction: Optional[jax.Array] = None,
    c_u: Optional[Union[float, jax.Array]] = None,
) -> jax.Array:
    """
    Projects input states onto a target potential isosurface (default z^+(z) = 0.0)
    in a single pass via Eq. (3.14):
        alpha^*(z) = (target_val - z^+(z)) / c_u,   z_proj = z + alpha^*(z) u

    When c_u is not explicitly provided, both z^+(z) and the directional coupling
    c_u = u^T grad_z z^+(z) are evaluated in a single forward-mode AD (jax.jvp) pass.

    Args:
        compiled_model_fn: A compiled graph or forward callable.
        sweep_signals: Tensor of input coordinates. Shape: (..., D)
        shift_indices: Indices of coordinates to shift uniformly (u_j = 1 for j in shift_indices).
        target_val: Target potential manifold value (default: 0.0).
        direction: Optional explicit shift direction vector u in R^D (overrides shift_indices).
        c_u: Optional known shift-equivariance gain (M u = c_u 1_K). Computed via JVP if None.

    Returns:
        jax.Array: The shifted coordinates on the target potential manifold.
    """
    dim = sweep_signals.shape[-1]

    if direction is not None:
        u = jnp.asarray(direction, dtype=sweep_signals.dtype)
        u_batched = jnp.broadcast_to(u, sweep_signals.shape)
        if c_u is None:
            batched_phi, c_u_val = jax.jvp(compiled_model_fn, (sweep_signals,), (u_batched,))
        else:
            batched_phi = compiled_model_fn(sweep_signals)
            c_u_val = jnp.asarray(c_u, dtype=batched_phi.dtype)
        target_tensor = jnp.asarray(target_val, dtype=batched_phi.dtype).squeeze()
        safe_c_u = jnp.where(c_u_val == 0.0, 1.0, c_u_val)
        alpha_star = (target_tensor - batched_phi) / safe_c_u
        return sweep_signals + jnp.expand_dims(alpha_star, -1) * u

    if shift_indices is None:
        raise ValueError("Either shift_indices or direction must be provided to gauge_fix.")

    shift_idx = jnp.atleast_1d(jnp.asarray(shift_indices, dtype=jnp.int32))
    if shift_idx.size == 0:
        return sweep_signals

    if c_u is None:
        u = jnp.zeros((dim,), dtype=sweep_signals.dtype).at[shift_idx].set(1.0)
        u_batched = jnp.broadcast_to(u, sweep_signals.shape)
        batched_phi, c_u_val = jax.jvp(compiled_model_fn, (sweep_signals,), (u_batched,))
    else:
        batched_phi = compiled_model_fn(sweep_signals)
        c_u_val = jnp.asarray(c_u, dtype=batched_phi.dtype)

    target_tensor = jnp.asarray(target_val, dtype=batched_phi.dtype).squeeze()
    return F.apply_gauge_shift(sweep_signals, batched_phi, shift_idx, target_tensor, c_u=c_u_val)
