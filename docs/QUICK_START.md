# Triton Neural - Quick Reference

## Installation
```bash
pip install triton-neural
```

## Import Styles

Triton Neural supports two import styles:

### Style 1: Direct Import (Simple)
```python
from triton_neural import *

# Use components directly
model = Sequential(Linear(784, 128), ReLU())
optimizer = Adam(learning_rate=0.001)
```

### Style 2: PyTorch-Style (Organized)
```python
import triton_neural as tn

# Use with module prefixes
model = tn.Sequential(tn.Linear(784, 128), tn.ReLU())
optimizer = tn.train.Adam(learning_rate=0.001)

# Access utilities
tn.util.print_model_summary(model, params, (784,))
tn.util.plot_history(history)

# Access transformers
transformer = tn.transformer.PremadeTransformer(
    num_layers=6,
    embed_dim=512
)
```

Both styles work identically - choose what you prefer!

## Basic Usage

### 1. Simple Neural Network
```python
import triton_neural as tn
from jax import random

# Create model
model = tn.Sequential(
    tn.Linear(784, 128),
    tn.ReLU(),
    tn.Dropout(0.2),
    tn.Linear(128, 10)
)

# Initialize & train
rng = random.PRNGKey(0)
params = model.init(rng, (784,))
optimizer = tn.train.Adam(learning_rate=0.001)
optimizer.init(params)

params, history = tn.train.fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    epochs=10,
    loss_fn=tn.cross_entropy_loss
)
```

### 2. Transformer (Choose Your Attention)
```python
import triton_neural as tn

# Standard attention
model = tn.transformer.PremadeTransformer(
    num_layers=6,
    embed_dim=512,
    attention_type='self'
)

# Flash (memory-efficient)
model = tn.transformer.PremadeTransformer(
    attention_type='flash'  # O(n) memory!
)

# RoPE (modern LLMs)
model = tn.transformer.PremadeTransformer(
    attention_type='rope'  # No length limit
)

# Sparse (long sequences)
model = tn.transformer.PremadeTransformer(
    attention_type='sparse'  # O(n√n)
)
```

### 3. VAE
```python
import triton_neural as tn

vae = tn.VAE(input_dim=784, latent_dim=32)
params = vae.init(rng, (784,))

# Forward pass
reconstruction, mu, logvar = vae(x, params, rng, training=True)

# Loss
loss = tn.vae_loss(reconstruction, x, mu, logvar, beta=1.0)

# Latent space utils
latent_space = tn.LatentSpace(vae)
latent = latent_space.read(x, params, rng)
interpolated = latent_space.interpolate(z1, z2, steps=10)
```

## Attention Types

| Type | When to Use |
|------|-------------|
| `self` | Standard (n < 512) |
| `masked` | GPT-style generation |
| `sparse` | Long docs (512-2048) |
| `flash` | Memory-limited (2048+) ⭐ |
| `rope` | Modern LLMs |
| `cross` | Encoder-decoder |

## Core Components

### Layers
```python
import triton_neural as tn

tn.Linear(in_features, out_features)
tn.Conv2D(in_channels, out_channels, kernel_size=3)
tn.BatchNorm(num_features)
tn.Dropout(rate=0.5)
tn.LayerNorm(dim)
tn.RMSNorm(dim)
```

### Activations
```python
tn.ReLU()
tn.GELU()
tn.Sigmoid()
tn.Tanh()
tn.Softmax()
tn.LeakyReLU(negative_slope=0.01)
```

### Loss Functions
```python
tn.mse_loss(predictions, targets)
tn.cross_entropy_loss(logits, labels)
tn.binary_cross_entropy_loss(predictions, targets)
tn.vae_loss(reconstruction, x, mu, logvar, beta=1.0)
```

### Optimizers
```python
tn.train.SGD(learning_rate=0.01, momentum=0.0)
tn.train.Adam(learning_rate=0.001)
tn.train.RMSprop(learning_rate=0.01, decay=0.9)
```

## Examples

See `triton_neural_examples.py` for:
1. MLP Classifier
2. CNN
3. BERT Encoder
4. GPT Decoder
5. Sparse Attention
6. Flash Attention
7. RoPE Attention
8. Seq2Seq
9. VAE
10. Latent Space
11. Conditional VAE
12. Hybrid Models
13. Full Training

## Common Patterns

### Train a Model
```python
import triton_neural as tn

params, history = tn.train.fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
    epochs=10,
    loss_fn=tn.cross_entropy_loss,
    batch_size=32,
    rng=rng
)
tn.util.plot_history(history)
```

### Build GPT
```python
import triton_neural as tn

gpt = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='rope',
    max_len=2048
)
```

### Build BERT
```python
import triton_neural as tn

bert = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='self',
    max_len=512
)
```

### Memory-Efficient Model
```python
import triton_neural as tn

efficient = tn.transformer.PremadeTransformer(
    attention_type='flash',  # O(n) memory
    max_len=2048
)
```

## Module Organization

```
triton_neural/
├── Core layers & activations (tn.Linear, tn.ReLU, etc.)
├── train module (tn.train.Adam, tn.train.fit, etc.)
├── util module (tn.util.plot_history, etc.)
└── transformer module (tn.transformer.PremadeTransformer, etc.)
```

---

**Full documentation in README.md**
