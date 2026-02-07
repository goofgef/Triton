"""
Triton Neural - Core Base Classes
==================================

Base classes used across the library.
"""

import jax
import jax.numpy as jnp
from jax import random


# ============================================================================
# BASE MODULE
# ============================================================================

class Module:
    """Base class for all neural network modules."""
    
    def __call__(self, x, params, rng=None, training=False):
        """Forward pass through the module."""
        raise NotImplementedError
    
    def init(self, rng, input_shape):
        """Initialize parameters."""
        raise NotImplementedError
