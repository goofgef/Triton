# Triton Neural - Import Styles Visual Comparison

## Side-by-Side Comparison

### Training a Simple Network

<table>
<tr>
<th>Old Style (from triton_neural import *)</th>
<th>New Style (import triton_neural as tn)</th>
</tr>
<tr>
<td>

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

# Initialize
rng = random.PRNGKey(0)
params = model.init(rng, (784,))

# Create optimizer
optimizer = Adam(learning_rate=0.001)
optimizer.init(params)

# Train
params, history = fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
    epochs=10,
    loss_fn=cross_entropy_loss
)

# Visualize
plot_history(history)
print_model_summary(model, params, (784,))
```

</td>
<td>

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

# Initialize
rng = random.PRNGKey(0)
params = model.init(rng, (784,))

# Create optimizer
optimizer = tn.train.Adam(learning_rate=0.001)
optimizer.init(params)

# Train
params, history = tn.train.fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
    epochs=10,
    loss_fn=tn.cross_entropy_loss
)

# Visualize
tn.util.plot_history(history)
tn.util.print_model_summary(model, params, (784,))
```

</td>
</tr>
</table>

### Building a Transformer

<table>
<tr>
<th>Old Style</th>
<th>New Style</th>
</tr>
<tr>
<td>

```python
from triton_neural import *

# BERT-style encoder
bert = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='self'
)

# GPT with Flash Attention
gpt = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='flash'
)

# Modern LLM with RoPE
llm = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='rope'
)
```

</td>
<td>

```python
import triton_neural as tn

# BERT-style encoder
bert = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='self'
)

# GPT with Flash Attention
gpt = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='flash'
)

# Modern LLM with RoPE
llm = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='rope'
)
```

</td>
</tr>
</table>

### VAE with Latent Space

<table>
<tr>
<th>Old Style</th>
<th>New Style</th>
</tr>
<tr>
<td>

```python
from triton_neural import *

# Create VAE
vae = VAE(
    input_dim=784,
    latent_dim=32
)

params = vae.init(rng, (784,))

# Forward pass
reconstruction, mu, logvar = vae(
    x, params, rng, training=True
)

# Compute loss
loss = vae_loss(
    reconstruction, x, mu, logvar, beta=1.0
)

# Latent space operations
latent_space = LatentSpace(vae)
latent = latent_space.read(x, params, rng)

# Interpolation
z1 = latent['sample'][0]
z2 = latent['sample'][1]
interpolated = latent_space.interpolate(
    z1, z2, steps=10
)

# Attribute vector
smile_vec = latent_space.get_attribute_vector(
    z_positive=smiling,
    z_negative=neutral
)
```

</td>
<td>

```python
import triton_neural as tn

# Create VAE
vae = tn.VAE(
    input_dim=784,
    latent_dim=32
)

params = vae.init(rng, (784,))

# Forward pass
reconstruction, mu, logvar = vae(
    x, params, rng, training=True
)

# Compute loss
loss = tn.vae_loss(
    reconstruction, x, mu, logvar, beta=1.0
)

# Latent space operations
latent_space = tn.LatentSpace(vae)
latent = latent_space.read(x, params, rng)

# Interpolation
z1 = latent['sample'][0]
z2 = latent['sample'][1]
interpolated = latent_space.interpolate(
    z1, z2, steps=10
)

# Attribute vector
smile_vec = latent_space.get_attribute_vector(
    z_positive=smiling,
    z_negative=neutral
)
```

</td>
</tr>
</table>

### CNN Model

<table>
<tr>
<th>Old Style</th>
<th>New Style</th>
</tr>
<tr>
<td>

```python
from triton_neural import *

model = Sequential(
    Conv2D(1, 32, kernel_size=3),
    ReLU(),
    BatchNorm(32),
    Conv2D(32, 64, kernel_size=3),
    ReLU(),
    Flatten(),
    Linear(64 * 7 * 7, 128),
    ReLU(),
    Dropout(0.5),
    Linear(128, 10)
)

optimizer = Adam(learning_rate=0.001)
```

</td>
<td>

```python
import triton_neural as tn

model = tn.Sequential(
    tn.Conv2D(1, 32, kernel_size=3),
    tn.ReLU(),
    tn.BatchNorm(32),
    tn.Conv2D(32, 64, kernel_size=3),
    tn.ReLU(),
    tn.Flatten(),
    tn.Linear(64 * 7 * 7, 128),
    tn.ReLU(),
    tn.Dropout(0.5),
    tn.Linear(128, 10)
)

optimizer = tn.train.Adam(learning_rate=0.001)
```

</td>
</tr>
</table>

## Quick Reference

### Core Components (Both Styles)

| Component | Old Style | New Style |
|-----------|-----------|-----------|
| Linear Layer | `Linear(10, 5)` | `tn.Linear(10, 5)` |
| ReLU | `ReLU()` | `tn.ReLU()` |
| Sequential | `Sequential(...)` | `tn.Sequential(...)` |
| VAE | `VAE(...)` | `tn.VAE(...)` |

### Training (Different Modules)

| Component | Old Style | New Style |
|-----------|-----------|-----------|
| Adam | `Adam(lr=0.001)` | `tn.train.Adam(lr=0.001)` |
| Fit | `fit(...)` | `tn.train.fit(...)` |
| Save | `save_params(...)` | `tn.train.save_params(...)` |
| Load | `load_params(...)` | `tn.train.load_params(...)` |

### Utilities (Different Modules)

| Component | Old Style | New Style |
|-----------|-----------|-----------|
| Plot | `plot_history(...)` | `tn.util.plot_history(...)` |
| Summary | `print_model_summary(...)` | `tn.util.print_model_summary(...)` |
| Guide | `print_attention_guide()` | `tn.util.print_attention_guide()` |

### Transformers (Different Modules)

| Component | Old Style | New Style |
|-----------|-----------|-----------|
| Transformer | `PremadeTransformer(...)` | `tn.transformer.PremadeTransformer(...)` |
| Self Attn | `SelfAttention(...)` | `tn.transformer.SelfAttention(...)` |
| Flash Attn | `FlashAttention(...)` | `tn.transformer.FlashAttention(...)` |
| RoPE Attn | `RoPEAttention(...)` | `tn.transformer.RoPEAttention(...)` |

## Advantages of Each Style

### Old Style (Direct Import)
- ✓ Shorter code
- ✓ Faster to type
- ✓ Good for notebooks/scripts
- ✓ Familiar if coming from simple imports

### New Style (PyTorch-Style)
- ✓ Clear organization
- ✓ Better IDE support
- ✓ Familiar if coming from PyTorch
- ✓ No namespace pollution
- ✓ Easy to see where things come from
- ✓ More professional/maintainable

## Recommendation

For **quick experiments and notebooks**: Use old style
```python
from triton_neural import *
```

For **production code and larger projects**: Use new style
```python
import triton_neural as tn
```

Both styles are fully supported and can be mixed in the same project!
