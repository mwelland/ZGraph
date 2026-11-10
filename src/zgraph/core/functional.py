import jax
import jax.numpy as jnp
from typing import Union
from .constants import MIN_SAFE_TAU


def marginalize(
    branch_potentials: jax.Array,
    tau: Union[float, jax.Array] = 1.0,
) -> jax.Array:
    """
    Stateless mathematical core of the ZGraph engine.
    Executes the shifted soft-tropical LogSumExp reduction:
        z^+ = tau * LSE(branch_potentials / tau)
            = z_max^+ + tau * ln(sum_i exp((branch_potentials_i - z_max^+) / tau))

    Args:
        branch_potentials (jax.Array): Uncollapsed branch potentials (M z + tau g).
            Shape: (..., K)
        tau (Union[float, jax.Array]): Non-negative scale parameter tau in [0, inf).
            tau -> 0^+ recovers the exact hard-tropical max_i branch_potentials_i.
            Shape: () (Scalar)

    Returns:
        jax.Array: The scalar outgoing log-potential z^+. Shape: (...,)
    """
    # Static compile-time bypass for K=1 (resolved at JIT trace time)
    if branch_potentials.shape[-1] == 1:
        return jnp.squeeze(branch_potentials, axis=-1)

    # Shifted LogSumExp: subtracting z_max^+ before dividing by safe_tau bounds all
    # exponential arguments in (-inf, 0], preventing overflow and 0 * inf = NaN for tau >= 0.
    z_max = jnp.max(branch_potentials, axis=-1, keepdims=True)
    all_neg_inf = jnp.isneginf(z_max)
    safe_z_max = jnp.where(all_neg_inf, 0.0, z_max)

    safe_tau = jnp.maximum(tau, MIN_SAFE_TAU)

    centered = (branch_potentials - safe_z_max) / safe_tau
    sum_exp = jnp.sum(jnp.exp(centered), axis=-1, keepdims=True)

    res = jnp.where(all_neg_inf, -jnp.inf, safe_z_max + tau * jnp.log(sum_exp))

    return jnp.squeeze(res, axis=-1)
