import jax
import jax.numpy as jnp
from typing import Tuple, Union


def apply_gauge_shift(
    primal_x: jax.Array,
    raw_phi: jax.Array,
    shift_idx: jax.Array,
    target_val: jax.Array,
    c_u: Union[float, jax.Array] = 1.0,
) -> jax.Array:
    """
    Pure subfunction to project state coordinates onto the target potential manifold
    z^+(z + alpha^* u) = target_val along shift_idx with affine gain c_u:
        alpha^* = (target_val - raw_phi) / c_u

    Args:
        primal_x (jax.Array): Input coordinates. Shape: (..., D)
        raw_phi (jax.Array): Evaluated potential z^+(primal_x). Shape: (...,)
        shift_idx (jax.Array): 1D integer tensor of coordinate indices to shift.
        target_val (jax.Array): Target potential value (default 0.0).
        c_u (Union[float, jax.Array]): Directional shift-equivariance gain u^T n.

    Returns:
        jax.Array: The projected coordinates on the target isosurface.
    """
    if shift_idx.size == 0:
        return primal_x

    safe_c_u = jnp.where(c_u == 0.0, 1.0, c_u)
    shift_amount = (target_val - raw_phi) / safe_c_u
    shifted_x = primal_x.at[..., shift_idx].add(
        jnp.expand_dims(shift_amount, -1), unique_indices=True
    )

    return shifted_x


def find_crossover_indices(
    batched_logits: jax.Array,
    ranks: Tuple[int, int] = (0, 1),
) -> Tuple[jax.Array, jax.Array]:
    """
    Pure subfunction to find the tensor indices where two rank-ordered uncollapsed
    branches cross via jax.lax.top_k.

    Args:
        batched_logits (jax.Array): Evaluated uncollapsed branch potentials across a sweep.
            Shape: [..., Sweep_Steps, K]
        ranks (Tuple[int, int]): Which two rank-ordered branches to find the crossover for.

    Returns:
        Tuple[jax.Array, jax.Array]:
            - The integer index in Sweep_Steps where the intersection occurs.
            - The actual branch indices [2] of the states at the intersection.
    """
    k_needed = max(ranks) + 1
    top_values, top_indices = jax.lax.top_k(batched_logits, k_needed)

    delta_omega = top_values[..., ranks[0]] - top_values[..., ranks[1]]
    boundary_idx = jnp.argmin(jnp.abs(delta_omega), axis=-1)

    boundary_idx_expanded = jnp.broadcast_to(
        jnp.expand_dims(boundary_idx, (-1, -2)),
        (*boundary_idx.shape, 1, k_needed),
    )
    selected_top_indices = jnp.take_along_axis(top_indices, boundary_idx_expanded, axis=-2).squeeze(-2)

    active_phases = jnp.stack(
        [
            selected_top_indices[..., ranks[0]],
            selected_top_indices[..., ranks[1]],
        ],
        axis=-1,
    )

    return boundary_idx, active_phases
