"""
Triton Neural - Training Components
===================================

Optimizers, training loops, and model persistence.
"""

import jax
import jax.numpy as jnp
from jax import random


# ============================================================================
# OPTIMIZERS
# ============================================================================

class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def init(self, params):
        if self.momentum > 0:
            self.velocity = jax.tree.map(jnp.zeros_like, params)
        return self
    
    def update(self, params, grads):
        if self.momentum > 0:
            self.velocity = jax.tree.map(
                lambda v, g: self.momentum * v + g,
                self.velocity, grads
            )
            return jax.tree.map(
                lambda p, v: p - self.learning_rate * v,
                params, self.velocity
            )
        else:
            return jax.tree.map(
                lambda p, g: p - self.learning_rate * g,
                params, grads
            )


class Adam:
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def init(self, params):
        self.m = jax.tree.map(jnp.zeros_like, params)
        self.v = jax.tree.map(jnp.zeros_like, params)
        return self
    
    def update(self, params, grads):
        self.t += 1
        
        self.m = jax.tree.map(
            lambda m, g: self.beta1 * m + (1 - self.beta1) * g,
            self.m, grads
        )
        self.v = jax.tree.map(
            lambda v, g: self.beta2 * v + (1 - self.beta2) * (g ** 2),
            self.v, grads
        )
        
        m_hat = jax.tree.map(lambda m: m / (1 - self.beta1 ** self.t), self.m)
        v_hat = jax.tree.map(lambda v: v / (1 - self.beta2 ** self.t), self.v)
        
        return jax.tree.map(
            lambda p, m, v: p - self.learning_rate * m / (jnp.sqrt(v) + self.eps),
            params, m_hat, v_hat
        )


class RMSprop:
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, decay: float = 0.9, eps: float = 1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.eps = eps
        self.v = None
    
    def init(self, params):
        self.v = jax.tree.map(jnp.zeros_like, params)
        return self
    
    def update(self, params, grads):
        self.v = jax.tree.map(
            lambda v, g: self.decay * v + (1 - self.decay) * (g ** 2),
            self.v, grads
        )
        
        return jax.tree.map(
            lambda p, g, v: p - self.learning_rate * g / (jnp.sqrt(v) + self.eps),
            params, grads, self.v
        )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_step(model, params, optimizer, x, y, loss_fn, rng=None):
    """Single training step."""
    def loss_wrapper(params):
        preds = model(x, params, rng, training=True)
        return loss_fn(preds, y)
    
    loss, grads = jax.value_and_grad(loss_wrapper)(params)
    params = optimizer.update(params, grads)
    
    return params, loss


def eval_step(model, params, x, y, loss_fn):
    """Single evaluation step."""
    preds = model(x, params, training=False)
    loss = loss_fn(preds, y)
    return loss


def fit(model, params, optimizer, train_data, epochs, loss_fn, 
        val_data=None, batch_size=32, rng=None, verbose=True):
    """Complete training loop."""
    history = {'train_loss': [], 'val_loss': []}
    
    x_train, y_train = train_data
    n_samples = x_train.shape[0]
    
    for epoch in range(epochs):
        if rng is not None:
            rng, shuffle_rng = random.split(rng)
            perm = random.permutation(shuffle_rng, n_samples)
            x_train = x_train[perm]
            y_train = y_train[perm]
        
        epoch_losses = []
        for i in range(0, n_samples, batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            if rng is not None:
                rng, step_rng = random.split(rng)
            else:
                step_rng = None
            
            params, loss = train_step(model, params, optimizer, 
                                     batch_x, batch_y, loss_fn, step_rng)
            epoch_losses.append(loss)
        
        train_loss = jnp.mean(jnp.array(epoch_losses))
        history['train_loss'].append(float(train_loss))
        
        if val_data is not None:
            x_val, y_val = val_data
            val_loss = eval_step(model, params, x_val, y_val, loss_fn)
            history['val_loss'].append(float(val_loss))
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}")
    
    return params, history


def accuracy(predictions, targets):
    """Calculate classification accuracy."""
    pred_classes = jnp.argmax(predictions, axis=-1)
    return jnp.mean(pred_classes == targets)


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_params(params, filepath):
    """Save model parameters."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)


def load_params(filepath):
    """Load model parameters."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)
