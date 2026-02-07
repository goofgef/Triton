# Triton Neural - Import Style Migration Guide

## Overview

Triton Neural now supports PyTorch-style imports! You can use either the old direct import style or the new organized module style.

## Import Styles Comparison

### Old Style (Still Supported)
```python
from triton_neural import *

# Create model
model = Sequential(Linear(784, 128), ReLU())

# Create optimizer
optimizer = Adam(learning_rate=0.001)

# Train
params, history = fit(model, params, optimizer, train_data, epochs=10, loss_fn=cross_entropy_loss)

# Utilities
plot_history(history)
print_model_summary(model, params, (784,))

# Transformers
transformer = PremadeTransformer(num_layers=6, embed_dim=512)
```

### New Style (Recommended)
```python
import triton_neural as tn

# Create model
model = tn.Sequential(tn.Linear(784, 128), tn.ReLU())

# Create optimizer
optimizer = tn.train.Adam(learning_rate=0.001)

# Train
params, history = tn.train.fit(model, params, optimizer, train_data, epochs=10, loss_fn=tn.cross_entropy_loss)

# Utilities
tn.util.plot_history(history)
tn.util.print_model_summary(model, params, (784,))

# Transformers
transformer = tn.transformer.PremadeTransformer(num_layers=6, embed_dim=512)
```

## Module Organization

### Core Module (`tn.*`)
Direct access to layers, activations, and basic components:

| Component | Import |
|-----------|--------|
| Layers | `tn.Linear`, `tn.Conv2D`, `tn.BatchNorm`, `tn.Dropout` |
| Activations | `tn.ReLU`, `tn.GELU`, `tn.Sigmoid`, `tn.Tanh`, `tn.Softmax` |
| Normalization | `tn.LayerNorm`, `tn.RMSNorm` |
| Container | `tn.Sequential` |
| VAE | `tn.VAE`, `tn.ConditionalVAE`, `tn.LatentSpace` |
| Losses | `tn.mse_loss`, `tn.cross_entropy_loss`, `tn.vae_loss` |

### Train Module (`tn.train.*`)
Training utilities, optimizers, and model persistence:

| Component | Import |
|-----------|--------|
| Optimizers | `tn.train.SGD`, `tn.train.Adam`, `tn.train.RMSprop` |
| Training | `tn.train.fit`, `tn.train.train_step`, `tn.train.eval_step` |
| Metrics | `tn.train.accuracy` |
| Persistence | `tn.train.save_params`, `tn.train.load_params` |

### Util Module (`tn.util.*`)
Visualization and model inspection:

| Component | Import |
|-----------|--------|
| Visualization | `tn.util.plot_history` |
| Inspection | `tn.util.print_model_summary` |
| Guides | `tn.util.print_attention_guide` |

### Transformer Module (`tn.transformer.*`)
All transformer components and attention mechanisms:

| Component | Import |
|-----------|--------|
| Attention | `tn.transformer.SelfAttention`, `tn.transformer.MaskedSelfAttention` |
| Advanced | `tn.transformer.FlashAttention`, `tn.transformer.RoPEAttention`, `tn.transformer.SparseAttention` |
| Models | `tn.transformer.PremadeTransformer`, `tn.transformer.PremadeTransformerDecoder` |
| Layers | `tn.transformer.TransformerLayer`, `tn.transformer.FeedForward` |
| Position | `tn.transformer.PositionalEncoding`, `tn.transformer.RotaryPositionalEmbedding` |

## Complete Examples

### Example 1: Training a Simple Network

**Old Style:**
```python
from triton_neural import *
from jax import random

model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
rng = random.PRNGKey(0)
params = model.init(rng, (784,))

optimizer = Adam(learning_rate=0.001)
optimizer.init(params)

params, history = fit(model, params, optimizer, (x_train, y_train), epochs=10, loss_fn=cross_entropy_loss)
plot_history(history)
```

**New Style:**
```python
import triton_neural as tn
from jax import random

model = tn.Sequential(tn.Linear(784, 128), tn.ReLU(), tn.Linear(128, 10))
rng = random.PRNGKey(0)
params = model.init(rng, (784,))

optimizer = tn.train.Adam(learning_rate=0.001)
optimizer.init(params)

params, history = tn.train.fit(model, params, optimizer, (x_train, y_train), epochs=10, loss_fn=tn.cross_entropy_loss)
tn.util.plot_history(history)
```

### Example 2: Building a Transformer

**Old Style:**
```python
from triton_neural import *

transformer = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    attention_type='flash'
)
```

**New Style:**
```python
import triton_neural as tn

transformer = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    attention_type='flash'
)
```

### Example 3: VAE with Latent Space

**Old Style:**
```python
from triton_neural import *

vae = VAE(input_dim=784, latent_dim=32)
params = vae.init(rng, (784,))
reconstruction, mu, logvar = vae(x, params, rng, training=True)
loss = vae_loss(reconstruction, x, mu, logvar)

latent_space = LatentSpace(vae)
z_interp = latent_space.interpolate(z1, z2, steps=10)
```

**New Style:**
```python
import triton_neural as tn

vae = tn.VAE(input_dim=784, latent_dim=32)
params = vae.init(rng, (784,))
reconstruction, mu, logvar = vae(x, params, rng, training=True)
loss = tn.vae_loss(reconstruction, x, mu, logvar)

latent_space = tn.LatentSpace(vae)
z_interp = latent_space.interpolate(z1, z2, steps=10)
```

## Benefits of New Style

1. **Better Organization**: Clear separation between core, train, util, and transformer
2. **Namespace Clarity**: Easy to see where each component comes from
3. **IDE Support**: Better autocomplete and documentation hints
4. **PyTorch Familiarity**: Similar to `torch.nn`, `torch.optim`, etc.
5. **No Pollution**: Doesn't clutter your namespace with 40+ names

## Migration Tips

1. **Both styles work**: No need to change existing code
2. **Gradual migration**: Update files one at a time
3. **Mix and match**: Can use both styles in same project
4. **Choose what fits**: Use the style that matches your workflow

## Quick Reference Table

| Old Import | New Import | Notes |
|------------|------------|-------|
| `from triton_neural import *` | `import triton_neural as tn` | Main import |
| `Adam(...)` | `tn.train.Adam(...)` | Optimizers in train module |
| `fit(...)` | `tn.train.fit(...)` | Training functions in train module |
| `plot_history(...)` | `tn.util.plot_history(...)` | Utilities in util module |
| `PremadeTransformer(...)` | `tn.transformer.PremadeTransformer(...)` | Transformers in transformer module |
| `Linear(...)` | `tn.Linear(...)` | Core components directly on tn |
| `cross_entropy_loss(...)` | `tn.cross_entropy_loss(...)` | Losses directly on tn |

---

**Choose the style that works best for you - both are fully supported!**
