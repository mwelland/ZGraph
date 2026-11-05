# Numerical barrier used to represent the tropical additive identity (-infinity)
# in max-plus soft-tropical algebra while avoiding IEEE 0.0 * (-inf) = NaN
# during matrix multiplication M @ x.
TROPICAL_ZERO_BARRIER = -1e30

# Minimum denominator floor for scale parameter tau in shifted LogSumExp:
# tau * log(sum(exp((z_i - z_max) / max(tau, MIN_SAFE_TAU)))).
# Using 1e-12 ensures 1 / safe_tau^3 stays below float32 max (~3.4e38)
# so 1st and 2nd derivatives at tau = 0.0 never overflow to NaN.
MIN_SAFE_TAU = 1e-12
