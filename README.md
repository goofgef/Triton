# Triton Neural - Unified Deep Learning API

**Version 2.0** - Complete neural network framework built on JAX

A comprehensive, modular deep learning library featuring state-of-the-art transformer architectures, latent space models, and memory-efficient attention mechanisms.

## To install..
pip install deeptriton
## Features

### Core Components
-   **Basic Layers**: Linear, Conv2D, BatchNorm, Dropout, Flatten, Reshape
-   **Activations**: ReLU, GELU, Sigmoid, Tanh, Softmax, LeakyReLU
-   **Container**: Sequential for easy model building

### Advanced Transformers
-   **6 Attention Types**:
  - `self` - Standard multi-head attention
  - `masked` - Causal attention for GPT-style models
  - `cross` - Encoder-decoder attention
  - `sparse` - O(n√n) for long sequences
  - `flash` - O(n) memory for very long contexts
  - `rope` - Rotary Position Embeddings (modern LLMs)
-   **Normalization**: LayerNorm, RMSNorm
-   **Position Encodings**: Sinusoidal, Learned, RoPE
-   **Complete Models**: PremadeTransformer, PremadeTransformerDecoder

### Latent Spaces
-   **VAE**: Variational Autoencoder
-   **CVAE**: Conditional VAE
-   **Utilities**: Interpolation, attribute vectors, latent manipulation

### Training & Optimization
-   **Optimizers**: SGD, Adam, RMSprop
-   **Loss Functions**: MSE, CrossEntropy, VAE losses
-   **Training Loop**: Complete fit() with validation
-   **Utilities**: Model summary, plotting, save/load

## Dependences

```bash
pip install jax jaxlib
# For GPU support:
# pip install jax[cuda12]
```

##   Quick Start

### Basic Neural Network

```python
from triton_neural import *
import jax.numpy as jnp
from jax import random

# Create a simple MLP
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 10),
    Softmax()
)

# Initialize
rng = random.PRNGKey(0)
params = model.init(rng, (784,))

# Train
optimizer = Adam(learning_rate=0.001)
optimizer.init(params)

params, history = fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    epochs=10,
    loss_fn=cross_entropy_loss
)
```

### Modern Transformer (Flash Attention)

```python
# Memory-efficient transformer for long sequences
transformer = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='flash',  # O(n) memory!
    max_len=2048
)

rng = random.PRNGKey(0)
params = transformer.init(rng, (2048, 768))

# Process long sequence
x = random.normal(rng, (1, 1024, 768))
output = transformer(x, params, rng, training=False)
```

### GPT-style Language Model

```python
# Autoregressive decoder with RoPE
gpt = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='rope',  # Modern position encoding
    max_len=2048,
    use_learned_pos=False
)

params = gpt.init(rng, (2048, 768))
```

### Variational Autoencoder

```python
# VAE for latent space learning
vae = VAE(
    input_dim=784,
    latent_dim=32,
    encoder_hidden=[256, 128],
    decoder_hidden=[128, 256]
)

params = vae.init(rng, (784,))

# Encode, sample, decode
reconstruction, mu, logvar = vae(x, params, rng, training=True)

# Compute VAE loss
loss = vae_loss(reconstruction, x, mu, logvar, beta=1.0)

# Latent space utilities
latent_space = LatentSpace(vae)
latent = latent_space.read(x, params, rng)

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
| `flash` | O(n²) | **O(n)**   | Memory-limited, 2048+ |
| `rope` | O(n²) | O(n²) | Modern LLMs, no length limit |
| `cross` | O(n²) | O(n²) | Encoder-decoder |

### Recommendations

```python
# Sequence < 512 tokens
transformer = PremadeTransformer(attention_type='self')

# GPT-style generation
gpt = PremadeTransformer(attention_type='masked')

# Long documents (512-2048)
doc_encoder = PremadeTransformer(attention_type='sparse')

# Very long sequences (2048+) or limited memory
efficient_model = PremadeTransformer(attention_type='flash')

# Building modern LLM
llm = PremadeTransformer(attention_type='rope')
```

## Architecture Examples

### BERT-style Encoder

```python
bert = PremadeTransformer(
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
gpt = PremadeTransformer(
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
# Encoder
encoder = PremadeTransformer(
    num_layers=6,
    embed_dim=512,
    num_heads=8
)

# Decoder with cross-attention
decoder = PremadeTransformerDecoder(
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
# Efficient sparse attention for 2048 tokens
long_encoder = PremadeTransformer(
    num_layers=8,
    embed_dim=512,
    num_heads=8,
    max_len=4096,
    attention_type='sparse'  # O(n√n) complexity
)
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
Linear(in_features, out_features)
Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding='SAME')
BatchNorm(num_features, momentum=0.9, eps=1e-5)
Dropout(rate=0.5)
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

### Attention Mechanisms

```python
SelfAttention(embed_dim, num_heads=8, dropout=0.0)
MaskedSelfAttention(embed_dim, num_heads=8, dropout=0.0)
CrossAttention(embed_dim, num_heads=8, dropout=0.0)
SparseAttention(embed_dim, num_heads=8, block_size=64, stride=64)
FlashAttention(embed_dim, num_heads=8, block_size=64)
RoPEAttention(embed_dim, num_heads=8, max_len=2048)
```

### Complete Models

```python
PremadeTransformer(
    num_layers, embed_dim, num_heads=8,
    ff_dim=None, max_len=512, dropout=0.1,
    attention_type='self',  # 'self', 'masked', 'sparse', 'flash', 'rope'
    use_learned_pos=False
)

PremadeTransformerDecoder(
    num_layers, embed_dim, num_heads=8,
    ff_dim=None, max_len=512, dropout=0.1,
    use_cross_attention=True,
    use_learned_pos=False
)
```

### Latent Space Models

```python
VAE(input_dim, latent_dim, encoder_hidden=None, decoder_hidden=None)
ConditionalVAE(input_dim, latent_dim, num_classes, ...)
LatentSpace(vae)  # Utility class for manipulation
```

### Optimizers

```python
SGD(learning_rate=0.01, momentum=0.0)
Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
RMSprop(learning_rate=0.01, decay=0.9)
```

### Loss Functions

```python
mse_loss(predictions, targets)
cross_entropy_loss(logits, labels)
binary_cross_entropy_loss(predictions, targets)
vae_loss(reconstruction, x, mu, logvar, beta=1.0)
vae_mse_loss(reconstruction, x, mu, logvar, beta=1.0)
```

### Training

```python
fit(model, params, optimizer, train_data, epochs, loss_fn,
    val_data=None, batch_size=32, rng=None, verbose=True)

train_step(model, params, optimizer, x, y, loss_fn, rng=None)
eval_step(model, params, x, y, loss_fn)
```

### Utilities

```python
accuracy(predictions, targets)
plot_history(history, metric='loss')
print_model_summary(model, params, input_shape)
save_params(params, filepath)
load_params(filepath)
```

## Key Benefits

### Modularity      
- Small, composable functions
- Mix and match any components
- Clear, explicit data flow

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

##  Advanced Topics

### Memory Optimization

```python
# Use Flash Attention for long sequences
model = PremadeTransformer(
    attention_type='flash',  # O(n) memory
    max_len=2048
)

# Use sparse attention for efficiency
model = PremadeTransformer(
    attention_type='sparse',  # O(n√n) 
    max_len=2048
)
```

### Latent Space Arithmetic

```python
latent_space = LatentSpace(vae)

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
# Combine different attention types
class HybridModel:
    def __init__(self):
        # Early layers: Flash (memory efficient)
        self.flash_layers = PremadeTransformer(
            num_layers=6,
            attention_type='flash'
        )
        
        # Later layers: RoPE (better positions)
        self.rope_layers = PremadeTransformer(
            num_layers=6,
            attention_type='rope'
        )
    
    def __call__(self, x, params, rng):
        x = self.flash_layers(x, params['flash'], rng)
        x = self.rope_layers(x, params['rope'], rng)
        return x
```

##  Performance Characteristics

| Component | Time | Space | Notes |
|-----------|------|-------|-------|
| Linear | O(nd) | O(nd) | Standard |
| Conv2D | O(n·k²·c) | O(n·c) | k=kernel, c=channels |
| SelfAttention | O(n²d) | O(n²) | Standard |
| SparseAttention | O(n√n·d) | O(n√n) | Efficient |
| FlashAttention | O(n²d) | **O(n)**   | Memory-efficient |
| RoPEAttention | O(n²d) | O(n²) | Modern LLMs |
| Full Transformer | O(L·n²d) | O(L·n²) | L=layers |

##  Development

### Requirements

- Python 3.8+
- JAX 0.6.0+
- NumPy

### Structure

```
triton_neural/
├── triton_neural.py            # Main API (unified)
├── triton_neural_examples.py   # 13 comprehensive examples
└── README.md                    # This file
```

## Citation

If you use Triton Neural in your research, please cite:

```bibtex
@software{triton_neural,
  title={Triton Neural: Unified Deep Learning API},
  author={Built with JAX},
  year={2026},
  version={2.0}
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

##   What Makes Triton Neural Special

1. **True Modularity** - Every component is independent
2. **6 Attention Types** - With more coming soon!
3. **Memory-Efficient** - Flash Attention with O(n) memory
4. **Modern LLM Support** - RoPE, modern architectures
5. **Complete VAE Suite** - With latent space utilities
6. **Educational** - Clean, readable, well-documented
7. **Production-Ready** - Full training pipelines, save/load

## Learn More

- See `triton_neural_examples.py` for 13 working examples
- Check inline documentation for detailed API info
- Experiment with different attention types
- Build hybrid models combining multiple approaches

---

**Happy Deep Learning with Triton Neural!  **

*A complete, modular deep learning framework for modern neural networks.*
