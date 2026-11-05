import jax
import jax.numpy as jnp
import equinox as eqx


class Ensemble(eqx.Module):
    """
    A lightweight wrapper that tags a JAX array as an uncertainty distribution (ensemble)
    rather than a physical vector constant.
    The _execute_vectorized interceptor recursively scans ZGraph PyTrees for this wrapper
    and automatically builds the in_axes mask to execute jax.vmap.
    """
    array: jax.Array

    @classmethod
    def normal(cls, mean, std, N: int, key: jax.Array):
        import jax.random as jr
        samples = mean + std * jr.normal(key, (N,) + jnp.shape(mean))
        return cls(samples)

    @classmethod
    def uniform(cls, low, high, N: int, key: jax.Array):
        import jax.random as jr
        samples = jr.uniform(key, (N,) + jnp.shape(low), minval=low, maxval=high)
        return cls(samples)


class BatchProxy:
    """
    A proxy object that intercepts method calls and automatically applies
    Uncertainty Quantification (UQ) and batch vectorization.
    """
    def __init__(self, node):
        self._node = node

    def __getattr__(self, name):
        def proxy_method(signals, signal_ndim=1, **kwargs):
            return self._node._execute_vectorized(name, signals, signal_ndim=signal_ndim, **kwargs)
        return proxy_method


class ZGraphNode(eqx.Module):
    """
    The universal base class for all nodes in the ZGraph framework.
    """

    def _execute_vectorized(self, method_name: str, signals: jax.Array, signal_ndim: int = 1, **kwargs):
        """
        The core interceptor that dynamically applies eqx.filter_vmap and jax.vmap.
        Inside the mapped function, Ensemble wrappers are stripped away, allowing
        the underlying method to remain purely scalar and pristine.
        """
        is_batched = getattr(signals, "ndim", 0) > signal_ndim

        has_ensemble = False
        def check_ensemble(x):
            nonlocal has_ensemble
            if isinstance(x, Ensemble):
                has_ensemble = True
            return None

        jax.tree_util.tree_map(check_ensemble, self, is_leaf=lambda x: isinstance(x, Ensemble))

        def _inner(sys, sig):
            def unwrap(x):
                return x.array if isinstance(x, Ensemble) else x
            unwrapped_sys = jax.tree_util.tree_map(unwrap, sys, is_leaf=lambda x: isinstance(x, Ensemble))

            unbound_method = getattr(unwrapped_sys, method_name)
            return unbound_method(sig, **kwargs)

        mapped_func = _inner

        if is_batched:
            num_batch_dims = signals.ndim - signal_ndim
            for _ in range(num_batch_dims):
                mapped_func = jax.vmap(mapped_func, in_axes=(None, 0))

        if has_ensemble:
            mask = jax.tree_util.tree_map(
                lambda x: 0 if isinstance(x, Ensemble) else None,
                self,
                is_leaf=lambda x: isinstance(x, Ensemble)
            )
            mapped_func = eqx.filter_vmap(mapped_func, in_axes=(mask, None))

        return mapped_func(self, signals)

    def __call__(self, signals, **kwargs):
        """
        The primary execution interface. Automatically vectorizes `evaluate`
        over parameter ensembles and signal batches.
        """
        return self._execute_vectorized("evaluate", signals, signal_ndim=1, **kwargs)

    def evaluate(self, signals, **kwargs):
        raise NotImplementedError("ZGraphNodes must implement pure scalar math in evaluate.")

    @property
    def ensemble(self):
        """
        Returns a BatchProxy. Any method called on this proxy will be automatically
        vectorized over the uncertainty ensemble and signal batches.
        Example: system.ensemble.logits(signals)
        """
        return BatchProxy(self)

    # --- Operator Overloading (Graph Flattening) ---
    def _is_1row_tpoly(self):
        from .tropical_nodes import TropicalPolynomialNode
        from .leaf_nodes import ConstantNode
        # Determine if this node is a flat linear combination (1-row M matrix with no g bias)
        if isinstance(self, TropicalPolynomialNode) and self.M.shape[0] == 1 and self.g is None:
            if isinstance(self.tau, ConstantNode) and self.tau.value == 1.0:
                return True
        return False

    def _is_unit_identity_tpoly(self):
        from .tropical_nodes import TropicalPolynomialNode
        from .leaf_nodes import ConstantNode
        # Determine if this node is an unshifted identity selector with unit scale tau=1.0
        if (
            isinstance(self, TropicalPolynomialNode)
            and self.g is None
            and isinstance(self.tau, ConstantNode)
            and self.tau.value == 1.0
            and self.M.shape[0] == len(self.subgraphs)
            and not isinstance(self.M, Ensemble)
        ):
            return bool(jnp.allclose(self.M, jnp.eye(len(self.subgraphs), dtype=self.M.dtype)))
        return False

    def __add__(self, other):
        from .tropical_nodes import TropicalPolynomialNode
        from .leaf_nodes import ConstantNode

        self_subgraphs = self.subgraphs if self._is_1row_tpoly() else [self]
        self_M = self.M[0] if self._is_1row_tpoly() else jnp.asarray([1.0], dtype=float)

        if isinstance(other, ZGraphNode) and other._is_1row_tpoly():
            other_subgraphs = other.subgraphs
            other_M = other.M[0]
        elif isinstance(other, (int, float, jax.Array)):
            other_subgraphs = [ConstantNode(other)]
            other_M = jnp.asarray([1.0], dtype=float)
        elif isinstance(other, ZGraphNode):
            other_subgraphs = [other]
            other_M = jnp.asarray([1.0], dtype=float)
        else:
            raise TypeError(f"Unsupported operand type for +: 'ZGraphNode' and '{type(other)}'")

        new_M = jnp.expand_dims(jnp.concatenate([self_M, other_M]), 0)
        new_subgraphs = self_subgraphs + other_subgraphs
        return TropicalPolynomialNode(M_matrix=new_M, subgraph_list=new_subgraphs)

    def __radd__(self, other):
        if other == 0:
            return self
        from .leaf_nodes import ConstantNode
        if isinstance(other, (int, float, jax.Array)):
            return ConstantNode(other).__add__(self)
        raise TypeError(f"Unsupported operand type for +: '{type(other)}' and 'ZGraphNode'")

    def __sub__(self, other):
        from .tropical_nodes import TropicalPolynomialNode
        from .leaf_nodes import ConstantNode

        self_subgraphs = self.subgraphs if self._is_1row_tpoly() else [self]
        self_M = self.M[0] if self._is_1row_tpoly() else jnp.asarray([1.0], dtype=float)

        if isinstance(other, ZGraphNode) and other._is_1row_tpoly():
            other_subgraphs = other.subgraphs
            other_M = -other.M[0]
        elif isinstance(other, (int, float, jax.Array)):
            other_subgraphs = [ConstantNode(other)]
            other_M = jnp.asarray([-1.0], dtype=float)
        elif isinstance(other, ZGraphNode):
            other_subgraphs = [other]
            other_M = jnp.asarray([-1.0], dtype=float)
        else:
            raise TypeError(f"Unsupported operand type for -: 'ZGraphNode' and '{type(other)}'")

        new_M = jnp.expand_dims(jnp.concatenate([self_M, other_M]), 0)
        new_subgraphs = self_subgraphs + other_subgraphs
        return TropicalPolynomialNode(M_matrix=new_M, subgraph_list=new_subgraphs)

    def __rsub__(self, other):
        from .leaf_nodes import ConstantNode
        if isinstance(other, (int, float, jax.Array)):
            return ConstantNode(other).__sub__(self)
        raise TypeError(f"Unsupported operand type for -: '{type(other)}' and 'ZGraphNode'")

    def __mul__(self, other):
        from .tropical_nodes import TropicalPolynomialNode
        if isinstance(other, (int, float, jax.Array)):
            scalar = other if (isinstance(other, jax.Array) and jnp.issubdtype(other.dtype, jnp.floating)) else jnp.asarray(other, dtype=float)
            if self._is_1row_tpoly():
                new_M = self.M * scalar
                return TropicalPolynomialNode(M_matrix=new_M, subgraph_list=self.subgraphs)
            else:
                return TropicalPolynomialNode(M_matrix=jnp.reshape(scalar, (1, 1)), subgraph_list=[self])
        raise TypeError(f"Unsupported operand type for *: 'ZGraphNode' and '{type(other)}'")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __or__(self, other):
        """Soft-Tropical Addition (LogSumExp selector with M = I)."""
        from .tropical_nodes import TropicalPolynomialNode
        if not isinstance(other, ZGraphNode):
            raise TypeError(f"Unsupported operand type for |: 'ZGraphNode' and '{type(other)}'")

        self_subgraphs = self.subgraphs if self._is_unit_identity_tpoly() else [self]
        other_subgraphs = other.subgraphs if other._is_unit_identity_tpoly() else [other]

        new_subgraphs = self_subgraphs + other_subgraphs
        return TropicalPolynomialNode(M_matrix=None, subgraph_list=new_subgraphs)
