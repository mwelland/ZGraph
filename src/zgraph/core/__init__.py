from .base import ZGraphNode, Ensemble
from .tropical_nodes import (
    TropicalPolynomialNode,
    TropicalMatMulNode,
)
from .utility_nodes import PiecewiseNode, StandardProductNode
from .leaf_nodes import ConstantNode, SignalNode, SignalNodes
from .functional import marginalize

import jax.numpy as jnp

# Convenience Aliases for standard probability/math naming
FactorNode = TropicalPolynomialNode
ProductNode = StandardProductNode
MatMulNode = TropicalMatMulNode


def AdditionNode(subgraph_list, weights=None):
    if weights is None:
        weights = jnp.ones(len(subgraph_list), dtype=float)
    elif not (isinstance(weights, jnp.ndarray) and jnp.issubdtype(weights.dtype, jnp.floating)):
        weights = jnp.asarray(weights, dtype=float)
    M = jnp.expand_dims(weights, 0)
    return TropicalPolynomialNode(M_matrix=M, subgraph_list=subgraph_list)


def DivisionNode(numerator, denominator):
    return TropicalPolynomialNode(
        M_matrix=jnp.asarray([[1.0, -1.0]], dtype=float),
        subgraph_list=[numerator, denominator],
    )


from .constants import TROPICAL_ZERO_BARRIER


def TropicalZeroNode(barrier_value=TROPICAL_ZERO_BARRIER):
    """Tropical Zero Element (for ⊕_tau). Represented as -inf (additive identity of LSE)."""
    return ConstantNode(barrier_value)


def TropicalIdentityNode():
    """Tropical Identity Element (for ⊗). Represented as constant 0.0."""
    return ConstantNode(0.0)


__all__ = [
    "ZGraphNode",
    "Ensemble",
    "FactorNode",
    "ProductNode",
    "AdditionNode",
    "DivisionNode",
    "MatMulNode",
    "TropicalPolynomialNode",
    "StandardProductNode",
    "TropicalMatMulNode",
    "TropicalZeroNode",
    "TropicalIdentityNode",
    "PiecewiseNode",
    "ConstantNode",
    "SignalNode",
    "SignalNodes",
    "marginalize",
]
