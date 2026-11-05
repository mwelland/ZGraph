import jax
import jax.numpy as jnp
from typing import Tuple


def apply_legendre(
    primal_x: jax.Array,
    raw_phi: jax.Array,
    full_grad: jax.Array,
    lt_idx: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Pure subfunction to compute the multivariate Legendre-Fenchel conjugate
    evaluated parametrically at primal_x using the standard convex Fenchel sign
    (Paper Eq. 3.9 - 3.10):
        z^*(n) = z^T n - z^+(z)

    Args:
        primal_x (jax.Array): Input primal coordinates.
        raw_phi (jax.Array): Evaluated primal potential z^+(primal_x).
        full_grad (jax.Array): Gradient vector n = grad z^+(primal_x).
        lt_idx (jax.Array): 1D integer tensor of indices for the Legendre transform.

    Returns:
        Tuple[jax.Array, jax.Array]: The dual potential z^* and dual coordinates.
    """
    if lt_idx.size == 0:
        return -raw_phi, primal_x

    psi = jnp.dot(primal_x[lt_idx], full_grad[lt_idx]) - raw_phi
    dual_x = primal_x.at[lt_idx].set(full_grad[lt_idx], unique_indices=True)
    return psi, dual_x
