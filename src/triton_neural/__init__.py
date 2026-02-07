"""
Triton Neural - Core Components
================================

Core neural network layers, activations, containers, and latent space models.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import List, Tuple, Optional

# Import submodules for PyTorch-style access
from . import train
from . import util
from . import transformer

# Re-export commonly used items for convenience
from .train import (
    SGD, Adam, RMSprop,
    train_step, eval_step, fit, accuracy,
    save_params, load_params
)

from .util import (
    print_model_summary, plot_history, print_attention_guide,
    ATTENTION_GUIDE
)

from .transformer import (
    SelfAttention, MaskedSelfAttention, CrossAttention,
    SparseAttention, FlashAttention, RoPEAttention,
    RotaryPositionalEmbedding,
    FeedForward, TransformerLayer, TransformerDecoderLayer,
    PositionalEncoding, LearnedPositionalEmbedding,
    PremadeTransformer, PremadeTransformerDecoder
)


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


# ============================================================================
# CORE LAYERS
# ============================================================================

class Linear(Module):
    """Fully connected linear layer."""
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
    
    def init(self, rng, input_shape):
        w_key, b_key = random.split(rng)
        scale = jnp.sqrt(2.0 / self.in_features)
        w = random.normal(w_key, (self.in_features, self.out_features)) * scale
        b = jnp.zeros((self.out_features,))
        return {'w': w, 'b': b}
    
    def __call__(self, x, params, rng=None, training=False):
        return jnp.dot(x, params['w']) + params['b']


class Conv2D(Module):
    """2D Convolutional layer."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: str = 'SAME'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def init(self, rng, input_shape):
        w_key, b_key = random.split(rng)
        scale = jnp.sqrt(2.0 / (self.kernel_size * self.kernel_size * self.in_channels))
        w = random.normal(w_key, (self.kernel_size, self.kernel_size, 
                                   self.in_channels, self.out_channels)) * scale
        b = jnp.zeros((self.out_channels,))
        return {'w': w, 'b': b}
    
    def __call__(self, x, params, rng=None, training=False):
        out = jax.lax.conv_general_dilated(
            x, params['w'],
            window_strides=(self.stride, self.stride),
            padding=self.padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        return out + params['b']


class BatchNorm(Module):
    """Batch normalization layer."""
    
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
    
    def init(self, rng, input_shape):
        return {
            'gamma': jnp.ones((self.num_features,)),
            'beta': jnp.zeros((self.num_features,)),
            'running_mean': jnp.zeros((self.num_features,)),
            'running_var': jnp.ones((self.num_features,))
        }
    
    def __call__(self, x, params, rng=None, training=False):
        if training:
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0)
            x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        else:
            x_norm = (x - params['running_mean']) / jnp.sqrt(params['running_var'] + self.eps)
        
        return params['gamma'] * x_norm + params['beta']


class Dropout(Module):
    """Dropout layer."""
    
    def __init__(self, rate: float = 0.5):
        self.rate = rate
    
    def init(self, rng, input_shape):
        return {}
    
    def __call__(self, x, params, rng=None, training=False):
        if training and rng is not None:
            keep_prob = 1.0 - self.rate
            mask = random.bernoulli(rng, keep_prob, x.shape)
            return jnp.where(mask, x / keep_prob, 0)
        return x


class Flatten(Module):
    """Flatten layer."""
    
    def init(self, rng, input_shape):
        return {}
    
    def __call__(self, x, params, rng=None, training=False):
        return x.reshape((x.shape[0], -1))


class Reshape(Module):
    """Reshape layer."""
    
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
    
    def init(self, rng, input_shape):
        return {}
    
    def __call__(self, x, params, rng=None, training=False):
        return x.reshape((x.shape[0],) + self.shape)


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class ReLU(Module):
    """ReLU activation."""
    
    def init(self, rng, input_shape):
        return {}
    
    def __call__(self, x, params, rng=None, training=False):
        return jax.nn.relu(x)


class Sigmoid(Module):
    """Sigmoid activation."""
    
    def init(self, rng, input_shape):
        return {}
    
    def __call__(self, x, params, rng=None, training=False):
        return jax.nn.sigmoid(x)


class Tanh(Module):
    """Tanh activation."""
    
    def init(self, rng, input_shape):
        return {}
    
    def __call__(self, x, params, rng=None, training=False):
        return jnp.tanh(x)


class Softmax(Module):
    """Softmax activation."""
    
    def init(self, rng, input_shape):
        return {}
    
    def __call__(self, x, params, rng=None, training=False):
        return jax.nn.softmax(x, axis=-1)


class GELU(Module):
    """GELU activation."""
    
    def init(self, rng, input_shape):
        return {}
    
    def __call__(self, x, params, rng=None, training=False):
        return jax.nn.gelu(x)


class LeakyReLU(Module):
    """Leaky ReLU activation."""
    
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope
    
    def init(self, rng, input_shape):
        return {}
    
    def __call__(self, x, params, rng=None, training=False):
        return jax.nn.leaky_relu(x, self.negative_slope)


# ============================================================================
# NORMALIZATION LAYERS
# ============================================================================

class LayerNorm(Module):
    """Layer normalization for transformers."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
    
    def init(self, rng, input_shape):
        return {
            'gamma': jnp.ones((self.normalized_shape,)),
            'beta': jnp.zeros((self.normalized_shape,))
        }
    
    def __call__(self, x, params, rng=None, training=False):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        return params['gamma'] * x_norm + params['beta']


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (more efficient variant)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
    
    def init(self, rng, input_shape):
        return {'gamma': jnp.ones((self.dim,))}
    
    def __call__(self, x, params, rng=None, training=False):
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return params['gamma'] * (x / rms)


# ============================================================================
# SEQUENTIAL CONTAINER
# ============================================================================

class Sequential:
    """Sequential container for chaining modules."""
    
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def init(self, rng, input_shape):
        params = []
        shape = input_shape
        
        for i, layer in enumerate(self.layers):
            rng, layer_rng = random.split(rng)
            layer_params = layer.init(layer_rng, shape)
            params.append(layer_params)
            
            dummy_input = jnp.zeros((1,) + shape)
            dummy_output = layer(dummy_input, layer_params)
            shape = dummy_output.shape[1:]
        
        return params
    
    def __call__(self, x, params, rng=None, training=False):
        for i, layer in enumerate(self.layers):
            if rng is not None:
                rng, layer_rng = random.split(rng)
            else:
                layer_rng = None
            x = layer(x, params[i], layer_rng, training)
        return x
    
    def add(self, layer):
        self.layers.append(layer)


# ============================================================================
# LATENT SPACE COMPONENTS (VAE, Encoders, Decoders)
# ============================================================================

class LatentEncoder(Module):
    """Encoder that maps input to latent space (mean, log-variance)."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128]
    
    def init(self, rng, input_shape):
        keys = random.split(rng, len(self.hidden_dims) + 2)
        params = {}
        
        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            scale = jnp.sqrt(2.0 / prev_dim)
            params[f'h{i}_w'] = random.normal(keys[i], (prev_dim, hidden_dim)) * scale
            params[f'h{i}_b'] = jnp.zeros((hidden_dim,))
            prev_dim = hidden_dim
        
        scale = jnp.sqrt(2.0 / prev_dim)
        params['mu_w'] = random.normal(keys[-2], (prev_dim, self.latent_dim)) * scale
        params['mu_b'] = jnp.zeros((self.latent_dim,))
        params['logvar_w'] = random.normal(keys[-1], (prev_dim, self.latent_dim)) * scale
        params['logvar_b'] = jnp.zeros((self.latent_dim,))
        
        return params
    
    def __call__(self, x, params, rng=None, training=False):
        h = x
        for i in range(len(self.hidden_dims)):
            h = jnp.dot(h, params[f'h{i}_w']) + params[f'h{i}_b']
            h = jax.nn.relu(h)
        
        mu = jnp.dot(h, params['mu_w']) + params['mu_b']
        logvar = jnp.dot(h, params['logvar_w']) + params['logvar_b']
        
        return mu, logvar


class LatentDecoder(Module):
    """Decoder that maps from latent space to output."""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list = None):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [128, 256]
    
    def init(self, rng, input_shape):
        keys = random.split(rng, len(self.hidden_dims) + 1)
        params = {}
        
        prev_dim = self.latent_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            scale = jnp.sqrt(2.0 / prev_dim)
            params[f'h{i}_w'] = random.normal(keys[i], (prev_dim, hidden_dim)) * scale
            params[f'h{i}_b'] = jnp.zeros((hidden_dim,))
            prev_dim = hidden_dim
        
        scale = jnp.sqrt(2.0 / prev_dim)
        params['out_w'] = random.normal(keys[-1], (prev_dim, self.output_dim)) * scale
        params['out_b'] = jnp.zeros((self.output_dim,))
        
        return params
    
    def __call__(self, z, params, rng=None, training=False):
        h = z
        for i in range(len(self.hidden_dims)):
            h = jnp.dot(h, params[f'h{i}_w']) + params[f'h{i}_b']
            h = jax.nn.relu(h)
        
        out = jnp.dot(h, params['out_w']) + params['out_b']
        out = jax.nn.sigmoid(out)
        
        return out


class VAE(Module):
    """
    Variational Autoencoder.
    
    Learns probabilistic latent representations of data.
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 encoder_hidden: list = None, decoder_hidden: list = None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = LatentEncoder(input_dim, latent_dim, encoder_hidden)
        self.decoder = LatentDecoder(latent_dim, input_dim, decoder_hidden)
    
    def init(self, rng, input_shape):
        enc_rng, dec_rng = random.split(rng, 2)
        
        return {
            'encoder': self.encoder.init(enc_rng, input_shape),
            'decoder': self.decoder.init(dec_rng, (self.latent_dim,))
        }
    
    def reparameterize(self, mu, logvar, rng):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, mu.shape)
        return mu + std * eps
    
    def encode(self, x, params, rng=None):
        return self.encoder(x, params['encoder'], rng)
    
    def decode(self, z, params, rng=None):
        return self.decoder(z, params['decoder'], rng)
    
    def __call__(self, x, params, rng=None, training=False):
        mu, logvar = self.encode(x, params, rng)
        
        if training and rng is not None:
            z = self.reparameterize(mu, logvar, rng)
        else:
            z = mu
        
        reconstruction = self.decode(z, params, rng)
        
        return reconstruction, mu, logvar


class ConditionalVAE(Module):
    """Conditional VAE - conditions generation on class labels."""
    
    def __init__(self, input_dim: int, latent_dim: int, num_classes: int,
                 encoder_hidden: list = None, decoder_hidden: list = None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.encoder = LatentEncoder(
            input_dim + num_classes,
            latent_dim,
            encoder_hidden
        )
        
        self.decoder = LatentDecoder(
            latent_dim + num_classes,
            input_dim,
            decoder_hidden
        )
    
    def init(self, rng, input_shape):
        enc_rng, dec_rng = random.split(rng, 2)
        
        return {
            'encoder': self.encoder.init(enc_rng, (self.input_dim + self.num_classes,)),
            'decoder': self.decoder.init(dec_rng, (self.latent_dim + self.num_classes,))
        }
    
    def encode(self, x, labels, params, rng=None):
        one_hot = jax.nn.one_hot(labels, self.num_classes)
        x_cond = jnp.concatenate([x, one_hot], axis=-1)
        return self.encoder(x_cond, params['encoder'], rng)
    
    def decode(self, z, labels, params, rng=None):
        one_hot = jax.nn.one_hot(labels, self.num_classes)
        z_cond = jnp.concatenate([z, one_hot], axis=-1)
        return self.decoder(z_cond, params['decoder'], rng)
    
    def reparameterize(self, mu, logvar, rng):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, mu.shape)
        return mu + std * eps
    
    def __call__(self, x, labels, params, rng=None, training=False):
        mu, logvar = self.encode(x, labels, params, rng)
        
        if training and rng is not None:
            z = self.reparameterize(mu, logvar, rng)
        else:
            z = mu
        
        reconstruction = self.decode(z, labels, params, rng)
        
        return reconstruction, mu, logvar


class LatentSpace:
    """Utility class for latent space manipulation."""
    
    def __init__(self, vae: VAE):
        self.vae = vae
    
    def read(self, x, params, rng=None):
        """Encode data into latent space."""
        mu, logvar = self.vae.encode(x, params, rng)
        
        result = {
            'mean': mu,
            'logvar': logvar,
            'std': jnp.exp(0.5 * logvar)
        }
        
        if rng is not None:
            result['sample'] = self.vae.reparameterize(mu, logvar, rng)
        else:
            result['sample'] = mu
        
        return result
    
    def write(self, z, params, rng=None):
        """Decode from latent space to data space."""
        return self.vae.decode(z, params, rng)
    
    def interpolate(self, z1, z2, steps=10):
        """Linear interpolation between latent vectors."""
        alphas = jnp.linspace(0, 1, steps)
        interpolated = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolated.append(z_interp)
        
        return jnp.array(interpolated)
    
    def sample_random(self, rng, batch_size, latent_dim):
        """Sample random points from latent space."""
        return random.normal(rng, (batch_size, latent_dim))
    
    def add_noise(self, z, noise_scale=0.1, rng=None):
        """Add noise to latent vectors."""
        if rng is None:
            return z
        
        noise = random.normal(rng, z.shape) * noise_scale
        return z + noise
    
    def spherical_interpolate(self, z1, z2, steps=10):
        """SLERP interpolation (better for normalized spaces)."""
        z1_norm = z1 / (jnp.linalg.norm(z1) + 1e-8)
        z2_norm = z2 / (jnp.linalg.norm(z2) + 1e-8)
        
        dot = jnp.sum(z1_norm * z2_norm)
        dot = jnp.clip(dot, -1.0, 1.0)
        omega = jnp.arccos(dot)
        
        sin_omega = jnp.sin(omega)
        interpolated = []
        
        for i in range(steps):
            t = i / (steps - 1)
            if sin_omega < 1e-6:
                z_interp = (1 - t) * z1 + t * z2
            else:
                coeff1 = jnp.sin((1 - t) * omega) / sin_omega
                coeff2 = jnp.sin(t * omega) / sin_omega
                z_interp = coeff1 * z1 + coeff2 * z2
            
            interpolated.append(z_interp)
        
        return jnp.array(interpolated)
    
    def get_attribute_vector(self, z_positive, z_negative):
        """Compute attribute direction for latent arithmetic."""
        mean_positive = jnp.mean(z_positive, axis=0)
        mean_negative = jnp.mean(z_negative, axis=0)
        return mean_positive - mean_negative
    
    def apply_attribute(self, z, attribute_vector, strength=1.0):
        """Apply attribute vector to latent representations."""
        return z + strength * attribute_vector


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def mse_loss(predictions, targets):
    """Mean squared error loss."""
    return jnp.mean((predictions - targets) ** 2)


def cross_entropy_loss(logits, labels):
    """Cross entropy loss for classification."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def binary_cross_entropy_loss(predictions, targets):
    """Binary cross entropy loss."""
    predictions = jnp.clip(predictions, 1e-7, 1 - 1e-7)
    return -jnp.mean(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions))


def vae_loss(reconstruction, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence.
    beta: Weight for KL term (beta-VAE)
    """
    recon_loss = -jnp.sum(
        x * jnp.log(reconstruction + 1e-8) +
        (1 - x) * jnp.log(1 - reconstruction + 1e-8),
        axis=-1
    )
    
    kl_loss = -0.5 * jnp.sum(
        1 + logvar - mu**2 - jnp.exp(logvar),
        axis=-1
    )
    
    return jnp.mean(recon_loss + beta * kl_loss)


def vae_mse_loss(reconstruction, x, mu, logvar, beta=1.0):
    """VAE loss with MSE reconstruction (for continuous data)."""
    recon_loss = jnp.sum((reconstruction - x)**2, axis=-1)
    
    kl_loss = -0.5 * jnp.sum(
        1 + logvar - mu**2 - jnp.exp(logvar),
        axis=-1
    )
    
    return jnp.mean(recon_loss + beta * kl_loss)


# ============================================================================
# PACKAGE METADATA
# ============================================================================

__version__ = "2.0.3"
__all__ = [
    # Modules
    'Module', 'Sequential',
    
    # Core layers
    'Linear', 'Conv2D', 'BatchNorm', 'Dropout', 'Flatten', 'Reshape',
    
    # Activations
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'GELU', 'LeakyReLU',
    
    # Normalization
    'LayerNorm', 'RMSNorm',
    
    # VAE components
    'VAE', 'ConditionalVAE', 'LatentEncoder', 'LatentDecoder', 'LatentSpace',
    
    # Loss functions
    'mse_loss', 'cross_entropy_loss', 'binary_cross_entropy_loss',
    'vae_loss', 'vae_mse_loss',
    
    # Training components (from train module)
    'SGD', 'Adam', 'RMSprop',
    'train_step', 'eval_step', 'fit', 'accuracy',
    'save_params', 'load_params',
    
    # Utilities (from util module)
    'print_model_summary', 'plot_history', 'print_attention_guide',
    
    # Transformer components (from transformer module)
    'SelfAttention', 'MaskedSelfAttention', 'CrossAttention',
    'SparseAttention', 'FlashAttention', 'RoPEAttention',
    'RotaryPositionalEmbedding',
    'FeedForward', 'TransformerLayer', 'TransformerDecoderLayer',
    'PositionalEncoding', 'LearnedPositionalEmbedding',
    'PremadeTransformer', 'PremadeTransformerDecoder',
    
    # Submodules for PyTorch-style access
    'train', 'util', 'transformer'
]
