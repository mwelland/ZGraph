import jax
from typing import Tuple, Callable, List
import zgraph.solvers.functional as F

def extract_decision_boundary(batched_logits_fn: Callable[[jax.Array], jax.Array], 
                              sweep_signals: jax.Array,
                              ranks: Tuple[int, int] = (0, 1)) -> Tuple[jax.Array, jax.Array]:
    """
    Extracts the decision boundary B_{ij} where two competing uncollapsed branch
    potentials cross ((z_i^+ + tau g_i) - (z_j^+ + tau g_j) = 0) using
    hardware-accelerated jax.lax.top_k selection.
    
    By default (ranks=(0, 1)), it finds the **Dominant Decision Boundary** where 
    the #1 and #2 branches swap. Sub-dominant crossovers can be queried via `ranks`.
    
    Args:
        batched_logits_fn: A batched callable that takes a tensor of signals and returns logits.
        sweep_signals: A 2D batched tensor of signals [Sweep_Steps, N_signals].
                       The decision boundary is searched along the Sweep_Steps dimension.
        ranks: Tuple indicating which two rank-ordered states to find the crossover between. 
               (0-indexed. 0 is lowest energy / highest probability). Defaults to (0, 1).
                       
    Returns:
        boundary_index: The integer index in Sweep_Steps where the intersection occurs.
        top_indices: The actual state indices [2] of the states at the intersection.
    """
    # Evaluate logits over the sweep using the provided batched callable
    batched_logits = batched_logits_fn(sweep_signals) # [Sweep_Steps, N_states]
    
    # Delegate pure tensor math to the core layer
    return F.find_crossover_indices(batched_logits, ranks)
