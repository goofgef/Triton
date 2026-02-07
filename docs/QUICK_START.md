# Triton Neural - Quick Reference

## Installation
```bash
pip install jax jaxlib
```

## Basic Usage

### 1. Simple Neural Network
```python
from triton_neural import *
from jax import random

# Create model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 10)
)

# Initialize & train
rng = random.PRNGKey(0)
params = model.init(rng, (784,))
optimizer = Adam(learning_rate=0.001)
optimizer.init(params)

params, history = fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    epochs=10,
    loss_fn=cross_entropy_loss
)
```

### 2. Transformer (Choose Your Attention)
```python
# Standard attention
model = PremadeTransformer(
    num_layers=6,
    embed_dim=512,
    attention_type='self'
)

# Flash (memory-efficient)
model = PremadeTransformer(
    attention_type='flash'  # O(n) memory!
)

# RoPE (modern LLMs)
model = PremadeTransformer(
    attention_type='rope'  # No length limit
)

# Sparse (long sequences)
model = PremadeTransformer(
    attention_type='sparse'  # O(n√n)
)
```

### 3. VAE
```python
vae = VAE(input_dim=784, latent_dim=32)
params = vae.init(rng, (784,))

# Forward pass
reconstruction, mu, logvar = vae(x, params, rng, training=True)

# Loss
loss = vae_loss(reconstruction, x, mu, logvar, beta=1.0)

# Latent space utils
latent_space = LatentSpace(vae)
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
Linear(in_features, out_features)
Conv2D(in_channels, out_channels, kernel_size=3)
BatchNorm(num_features)
Dropout(rate=0.5)
LayerNorm(dim)
RMSNorm(dim)
```

### Activations
```python
ReLU()
GELU()
Sigmoid()
Tanh()
Softmax()
LeakyReLU(negative_slope=0.01)
```

### Loss Functions
```python
mse_loss(predictions, targets)
cross_entropy_loss(logits, labels)
binary_cross_entropy_loss(predictions, targets)
vae_loss(reconstruction, x, mu, logvar, beta=1.0)
```

### Optimizers
```python
SGD(learning_rate=0.01, momentum=0.0)
Adam(learning_rate=0.001)
RMSprop(learning_rate=0.01, decay=0.9)
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
params, history = fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
    epochs=10,
    loss_fn=cross_entropy_loss,
    batch_size=32,
    rng=rng
)
plot_history(history)
```

### Build GPT
```python
gpt = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='rope',
    max_len=2048
)
```

### Build BERT
```python
bert = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='self',
    max_len=512
)
```

### Memory-Efficient Model
```python
efficient = PremadeTransformer(
    attention_type='flash',  # O(n) memory
    max_len=2048
)
```

---

**Full documentation in README.md**
