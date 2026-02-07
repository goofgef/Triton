# Triton Neural - Unified Deep Learning API

**Version 2.0.5** - Complete neural network framework built on JAX

A comprehensive, modular deep learning library featuring state-of-the-art transformer architectures, latent space models, and memory-efficient attention mechanisms.

## Installation

```bash
pip install triton-neural
```

## Import Styles

Triton Neural supports two import styles for maximum flexibility:

### Direct Import (Simple)
```python
from triton_neural import *

model = Sequential(Linear(784, 128), ReLU())
optimizer = Adam(learning_rate=0.001)
```

### PyTorch-Style Import (Organized)
```python
import triton_neural as tn

# Core components
model = tn.Sequential(tn.Linear(784, 128), tn.ReLU())

# Training utilities
optimizer = tn.train.Adam(learning_rate=0.001)
params, history = tn.train.fit(model, params, optimizer, ...)

# Utilities
tn.util.print_model_summary(model, params, (784,))
tn.util.plot_history(history)

# Transformers
transformer = tn.transformer.PremadeTransformer(num_layers=6, embed_dim=512)
```

Both styles work identically - choose what fits your workflow!

## Features

### Core Components
- **Basic Layers**: Linear, Conv2D, BatchNorm, Dropout, Flatten, Reshape
- **Activations**: ReLU, GELU, Sigmoid, Tanh, Softmax, LeakyReLU
- **Container**: Sequential for easy model building

### Advanced Transformers
- **6 Attention Types**:
  - `self` - Standard multi-head attention
  - `masked` - Causal attention for GPT-style models
  - `cross` - Encoder-decoder attention
  - `sparse` - O(n√n) for long sequences
  - `flash` - O(n) memory for very long contexts
  - `rope` - Rotary Position Embeddings (modern LLMs)
- **Normalization**: LayerNorm, RMSNorm
- **Position Encodings**: Sinusoidal, Learned, RoPE
- **Complete Models**: PremadeTransformer, PremadeTransformerDecoder

### Latent Spaces
- **VAE**: Variational Autoencoder
- **CVAE**: Conditional VAE
- **Utilities**: Interpolation, attribute vectors, latent manipulation

### Training & Optimization
- **Optimizers**: SGD, Adam, RMSprop
- **Loss Functions**: MSE, CrossEntropy, VAE losses
- **Training Loop**: Complete fit() with validation
- **Utilities**: Model summary, plotting, save/load

## Quick Start

### Basic Neural Network

```python
import triton_neural as tn
import numpy as np

# Create sample data (MNIST-like)
# x_train: flattened 28x28 images, shape (num_samples, 784)
# y_train: labels 0-9, shape (num_samples,)
x_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# Create a simple MLP
model = tn.Sequential(
    tn.Linear(784, 256),
    tn.ReLU(),
    tn.Dropout(0.2),
    tn.Linear(256, 10),
    tn.Softmax()
)

# Initialize (using built-in rng, no need to import JAX!)
key = tn.rng.PRNGKey(0)
params = model.init(key, (784,))

# Train
optimizer = tn.train.Adam(learning_rate=0.001)
optimizer.init(params)

params, history = tn.train.fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    epochs=10,
    loss_fn=tn.cross_entropy_loss
)
```

### Modern Transformer (Flash Attention)

```python
import triton_neural as tn

# Memory-efficient transformer for long sequences
transformer = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='flash',  # O(n) memory!
    max_len=2048
)

key = tn.rng.PRNGKey(0)
params = transformer.init(key, (2048, 768))

# Process long sequence
x = tn.rng.normal(key, (1, 1024, 768))
output = transformer(x, params, key, training=False)
```

### GPT-style Language Model

```python
import triton_neural as tn

# Autoregressive decoder with RoPE
gpt = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='rope',  # Modern position encoding
    max_len=2048,
    use_learned_pos=False
)

key = tn.rng.PRNGKey(0)
params = gpt.init(key, (2048, 768))
```

### Variational Autoencoder

```python
import triton_neural as tn
import numpy as np

# VAE for latent space learning
vae = tn.VAE(
    input_dim=784,
    latent_dim=32,
    encoder_hidden=[256, 128],
    decoder_hidden=[128, 256]
)

key = tn.rng.PRNGKey(0)
params = vae.init(key, (784,))

# Sample data
x = np.random.randn(10, 784).astype(np.float32)

# Encode, sample, decode
reconstruction, mu, logvar = vae(x, params, key, training=True)

# Compute VAE loss
loss = tn.vae_loss(reconstruction, x, mu, logvar, beta=1.0)

# Latent space utilities
latent_space = tn.LatentSpace(vae)
latent = latent_space.read(x, params, key)

# Interpolate in latent space
z1 = latent['sample'][0]
z2 = latent['sample'][1]
interpolated = latent_space.interpolate(z1, z2, steps=10)
```

## Attention Type Selection

### When to Use What

| Attention Type | Complexity | Memory | Use Case |
|----------------|------------|--------|----------|
| `self` | O(n²) | O(n²) | Standard, n < 512 |
| `masked` | O(n²) | O(n²) | GPT-style, autoregressive |
| `sparse` | O(n√n) | O(n√n) | Long docs, 512-2048 |
| `flash` | O(n²) | **O(n)** | Memory-limited, 2048+ |
| `rope` | O(n²) | O(n²) | Modern LLMs, no length limit |
| `cross` | O(n²) | O(n²) | Encoder-decoder |

### Recommendations

```python
import triton_neural as tn

# Sequence < 512 tokens
transformer = tn.transformer.PremadeTransformer(attention_type='self')

# GPT-style generation
gpt = tn.transformer.PremadeTransformer(attention_type='masked')

# Long documents (512-2048)
doc_encoder = tn.transformer.PremadeTransformer(attention_type='sparse')

# Very long sequences (2048+) or limited memory
efficient_model = tn.transformer.PremadeTransformer(attention_type='flash')

# Building modern LLM
llm = tn.transformer.PremadeTransformer(attention_type='rope')
```

## Architecture Examples

### BERT-style Encoder

```python
import triton_neural as tn

bert = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    ff_dim=3072,
    max_len=512,
    attention_type='self',
    dropout=0.1
)
```

### GPT-style Decoder

```python
import triton_neural as tn

gpt = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    ff_dim=3072,
    max_len=1024,
    attention_type='masked',
    use_learned_pos=True,
    dropout=0.1
)
```

### Seq2Seq Translation

```python
import triton_neural as tn

# Encoder
encoder = tn.transformer.PremadeTransformer(
    num_layers=6,
    embed_dim=512,
    num_heads=8
)

# Decoder with cross-attention
decoder = tn.transformer.PremadeTransformerDecoder(
    num_layers=6,
    embed_dim=512,
    num_heads=8,
    use_cross_attention=True
)

# Usage
encoder_out = encoder(source, enc_params, rng)
decoder_out = decoder(target, dec_params, rng, 
                     encoder_output=encoder_out)
```

### Long Document Processing

```python
import triton_neural as tn

# Efficient sparse attention for 2048 tokens
long_encoder = tn.transformer.PremadeTransformer(
    num_layers=8,
    embed_dim=512,
    num_heads=8,
    max_len=4096,
    attention_type='sparse'  # O(n√n) complexity
)
```

## Module Organization

Triton Neural is organized into modules for easy access:

### Core Module (triton_neural)
Direct access to layers, activations, and basic components:
```python
import triton_neural as tn

tn.Linear(784, 128)
tn.ReLU()
tn.Sequential(...)
tn.VAE(...)
```

### Train Module (triton_neural.train)
Training utilities, optimizers, and model persistence:
```python
tn.train.Adam(learning_rate=0.001)
tn.train.fit(model, params, optimizer, ...)
tn.train.save_params(params, 'model.pkl')
tn.train.load_params('model.pkl')
```

### Util Module (triton_neural.util)
Visualization and model inspection:
```python
tn.util.print_model_summary(model, params, input_shape)
tn.util.plot_history(history)
tn.util.print_attention_guide()
```

### Transformer Module (triton_neural.transformer)
All transformer components and attention mechanisms:
```python
tn.transformer.PremadeTransformer(...)
tn.transformer.SelfAttention(...)
tn.transformer.FlashAttention(...)
tn.transformer.RoPEAttention(...)
```

## Complete Examples

See `triton_neural_examples.py` for 13 comprehensive examples:

1. MLP Classifier
2. CNN for Images
3. BERT-style Transformer
4. GPT-style Decoder
5. Sparse Attention
6. Flash Attention
7. RoPE Attention
8. Seq2Seq Translation
9. VAE
10. Latent Space Manipulation
11. Conditional VAE
12. Hybrid Models
13. Complete Training Pipeline

## API Reference

### Core Layers

```python
import triton_neural as tn

tn.Linear(in_features, out_features)
tn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding='SAME')
tn.BatchNorm(num_features, momentum=0.9, eps=1e-5)
tn.Dropout(rate=0.5)
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

### Attention Mechanisms

```python
tn.transformer.SelfAttention(embed_dim, num_heads=8, dropout=0.0)
tn.transformer.MaskedSelfAttention(embed_dim, num_heads=8, dropout=0.0)
tn.transformer.CrossAttention(embed_dim, num_heads=8, dropout=0.0)
tn.transformer.SparseAttention(embed_dim, num_heads=8, block_size=64, stride=64)
tn.transformer.FlashAttention(embed_dim, num_heads=8, block_size=64)
tn.transformer.RoPEAttention(embed_dim, num_heads=8, max_len=2048)
```

### Complete Models

```python
tn.transformer.PremadeTransformer(
    num_layers, embed_dim, num_heads=8,
    ff_dim=None, max_len=512, dropout=0.1,
    attention_type='self',  # 'self', 'masked', 'sparse', 'flash', 'rope'
    use_learned_pos=False
)

tn.transformer.PremadeTransformerDecoder(
    num_layers, embed_dim, num_heads=8,
    ff_dim=None, max_len=512, dropout=0.1,
    use_cross_attention=True,
    use_learned_pos=False
)
```

### Latent Space Models

```python
tn.VAE(input_dim, latent_dim, encoder_hidden=None, decoder_hidden=None)
tn.ConditionalVAE(input_dim, latent_dim, num_classes, ...)
tn.LatentSpace(vae)  # Utility class for manipulation
```

### Optimizers

```python
tn.train.SGD(learning_rate=0.01, momentum=0.0)
tn.train.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
tn.train.RMSprop(learning_rate=0.01, decay=0.9)
```

### Loss Functions

```python
tn.mse_loss(predictions, targets)
tn.cross_entropy_loss(logits, labels)
tn.binary_cross_entropy_loss(predictions, targets)
tn.vae_loss(reconstruction, x, mu, logvar, beta=1.0)
tn.vae_mse_loss(reconstruction, x, mu, logvar, beta=1.0)
```

### Training

```python
tn.train.fit(model, params, optimizer, train_data, epochs, loss_fn,
             val_data=None, batch_size=32, rng=None, verbose=True)

tn.train.train_step(model, params, optimizer, x, y, loss_fn, rng=None)
tn.train.eval_step(model, params, x, y, loss_fn)
```

### Utilities

```python
tn.train.accuracy(predictions, targets)
tn.util.plot_history(history, metric='loss')
tn.util.print_model_summary(model, params, input_shape)
tn.train.save_params(params, filepath)
tn.train.load_params(filepath)
```

## Key Benefits

### Modularity
- Small, composable functions
- Mix and match any components
- Clear, explicit data flow
- PyTorch-style module organization

### Performance
- Built on JAX (JIT compilation, GPU support)
- O(n) memory with Flash Attention
- Efficient sparse attention for long sequences

### Modern Features
- 6 attention types
- RoPE for state-of-the-art position encoding
- Flash Attention for memory efficiency
- Complete VAE support with utilities

### Educational
- Clean, readable code
- Comprehensive examples
- Well-documented API
- Two import styles for flexibility

## Advanced Topics

### Memory Optimization

```python
import triton_neural as tn

# Use Flash Attention for long sequences
model = tn.transformer.PremadeTransformer(
    attention_type='flash',  # O(n) memory
    max_len=2048
)

# Use sparse attention for efficiency
model = tn.transformer.PremadeTransformer(
    attention_type='sparse',  # O(n√n)
    max_len=2048
)
```

### Latent Space Arithmetic

```python
import triton_neural as tn

latent_space = tn.LatentSpace(vae)

# Interpolate between points
z_interp = latent_space.interpolate(z1, z2, steps=10)

# Attribute vectors (e.g., "smile" direction)
smile_vec = latent_space.get_attribute_vector(
    z_positive=smiling_faces,
    z_negative=neutral_faces
)

# Apply attribute
z_smiling = latent_space.apply_attribute(z, smile_vec, strength=2.0)
```

### Hybrid Models

```python
import triton_neural as tn

# Combine different attention types
class HybridModel:
    def __init__(self):
        # Early layers: Flash (memory efficient)
        self.flash_layers = tn.transformer.PremadeTransformer(
            num_layers=6,
            attention_type='flash'
        )
        
        # Later layers: RoPE (better positions)
        self.rope_layers = tn.transformer.PremadeTransformer(
            num_layers=6,
            attention_type='rope'
        )
    
    def __call__(self, x, params, rng):
        x = self.flash_layers(x, params['flash'], rng)
        x = self.rope_layers(x, params['rope'], rng)
        return x
```

## Performance Characteristics

| Component | Time | Space | Notes |
|-----------|------|-------|-------|
| Linear | O(nd) | O(nd) | Standard |
| Conv2D | O(n·k²·c) | O(n·c) | k=kernel, c=channels |
| SelfAttention | O(n²d) | O(n²) | Standard |
| SparseAttention | O(n√n·d) | O(n√n) | Efficient |
| FlashAttention | O(n²d) | **O(n)** | Memory-efficient |
| RoPEAttention | O(n²d) | O(n²) | Modern LLMs |
| Full Transformer | O(L·n²d) | O(L·n²) | L=layers |

## Development

### Requirements

- Python 3.8+
- JAX 0.6.0+
- NumPy

### Structure

```
triton_neural/
├── __init__.py          # Core layers, activations, VAE
├── train.py             # Optimizers, training loops
├── util.py              # Visualization, utilities
└── transformer.py       # Attention mechanisms, transformers
```

## Citation

If you use Triton Neural in your research, please cite:

```bibtex
@software{triton_neural,
  title={Triton Neural: Unified Deep Learning API},
  author={Built with JAX},
  year={2026},
  version={2.0.2}
}
```

## Contributing

Contributions welcome! This is a modular, educational framework.

## License

MIT License - Free to use in research and production.

## Acknowledgments

- Built on JAX by Google
- Inspired by PyTorch, Flax, and modern transformer architectures
- Flash Attention algorithm from Dao et al.
- RoPE from Su et al. (used in LLaMA, PaLM)

## What Makes Triton Neural Special

1. **True Modularity** - Every component is independent
2. **PyTorch-Style Organization** - `tn.train.Adam`, `tn.util.plot_history`, etc.
3. **6 Attention Types** - More than most frameworks
4. **Memory-Efficient** - Flash Attention with O(n) memory
5. **Modern LLM Support** - RoPE, modern architectures
6. **Complete VAE Suite** - With latent space utilities
7. **Educational** - Clean, readable, well-documented
8. **Production-Ready** - Full training pipelines, save/load
9. **Flexible** - Two import styles to match your preference

## Learn More

- See `triton_neural_examples.py` for 13 working examples
- Check inline documentation for detailed API info
- Experiment with different attention types
- Build hybrid models combining multiple approaches

---

**Happy Deep Learning with Triton Neural!**

*A complete, modular deep learning framework for modern neural networks.*
