# Triton Neural - Unified API Summary

## What You're Getting

A complete, production-ready deep learning framework that merges:
- **Core neural network layers** (Linear, Conv, BatchNorm, etc.)
- **6 advanced attention mechanisms** (Self, Masked, Cross, Sparse, Flash, RoPE)
- **Complete transformer architectures** (BERT, GPT, Seq2Seq)
- **Latent space models** (VAE, CVAE with utilities)
- **Training infrastructure** (optimizers, losses, fit loop)

All in a single, clean API built on JAX with PyTorch-style module organization.

## Import Styles

**New in v2.0.2**: PyTorch-style imports!

```python
# Style 1: Direct (simple)
from triton_neural import *
model = Sequential(Linear(784, 128), ReLU())
optimizer = Adam(learning_rate=0.001)

# Style 2: PyTorch-style (organized)
import triton_neural as tn
model = tn.Sequential(tn.Linear(784, 128), tn.ReLU())
optimizer = tn.train.Adam(learning_rate=0.001)
tn.util.plot_history(history)
```

Both styles work identically - choose what fits your workflow!

## Module Organization

```
triton_neural/
├── Core (tn.Linear, tn.ReLU, tn.VAE, etc.)
├── tn.train (optimizers, fit, save/load)
├── tn.util (plotting, summaries, guides)
└── tn.transformer (all attention types)
```

## Key Features

### 6 Attention Types (Most Frameworks Have 1-2)

| Type | Complexity | Memory | Use Case |
|------|-----------|--------|----------|
| `self` | O(n²) | O(n²) | Standard transformers |
| `masked` | O(n²) | O(n²) | GPT-style generation |
| `cross` | O(n²) | O(n²) | Encoder-decoder |
| `sparse` | O(n√n) | O(n√n) | Long sequences (512-2048) |
| **`flash`** | O(n²) | **O(n)** | Memory-efficient (2048+) |
| **`rope`** | O(n²) | O(n²) | Modern LLMs (no length limit) |

### Highlights

1. **Flash Attention**: O(n) memory instead of O(n²) - enables 2048+ token sequences
2. **RoPE**: Rotary Position Embeddings used in LLaMA, PaLM - no length limitation
3. **Complete VAE Suite**: With latent space manipulation utilities
4. **PyTorch-Style Organization**: `tn.train.Adam`, `tn.util.plot_history`, etc.
5. **Fully Modular**: Mix and match any components
6. **Production-Ready**: Complete training loops, save/load, utilities

## Usage Examples

### 1. Memory-Efficient Transformer
```python
import triton_neural as tn

# Flash Attention: O(n) memory!
model = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='flash',
    max_len=2048
)

# Can now process 2048+ tokens!
```

### 2. Modern LLM with RoPE
```python
import triton_neural as tn

# Rotary Position Embeddings (like LLaMA)
gpt = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='rope',  # Modern approach
    max_len=2048
)

# No length limitation!
```

### 3. VAE with Latent Space
```python
import triton_neural as tn

# Variational Autoencoder
vae = tn.VAE(input_dim=784, latent_dim=32)
params = vae.init(rng, (784,))

# Encode to latent space
reconstruction, mu, logvar = vae(x, params, rng, training=True)

# Latent space utilities
latent_space = tn.LatentSpace(vae)
z_interp = latent_space.interpolate(z1, z2, steps=10)
```

### 4. Complete Training Pipeline
```python
import triton_neural as tn

model = tn.Sequential(
    tn.Linear(784, 256), tn.ReLU(),
    tn.Dropout(0.2),
    tn.Linear(256, 10)
)

params = model.init(rng, (784,))
optimizer = tn.train.Adam(learning_rate=0.001)
optimizer.init(params)

params, history = tn.train.fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    val_data=(x_val, y_val),
    epochs=10,
    loss_fn=tn.cross_entropy_loss
)

tn.util.plot_history(history)
```

## What's Supported

### Neural Network Tasks
- Image classification (CNN)
- Text classification (Transformer)
- Language modeling (GPT-style)
- Machine translation (Seq2Seq)
- Long document processing (Sparse/Flash)
- Representation learning (VAE)
- Conditional generation (CVAE)

### Technical Capabilities
- Multi-head attention (all 6 types)
- Causal masking (autoregressive)
- Cross-attention (encoder-decoder)
- Position encodings (3 types)
- Layer normalization (2 types)
- Dropout & regularization
- Batch normalization
- Convolutional layers
- Fully connected layers
- Various activations
- Multiple optimizers
- Different loss functions
- Training & validation loops
- Model save/load
- ASCII plotting

## Performance

- **JAX Backend**: JIT compilation, GPU support
- **Flash Attention**: Enables 2048+ token sequences
- **Sparse Attention**: 512-2048 tokens efficiently
- **Memory-Efficient**: O(n) memory with Flash

## Perfect For

- **Learning**: Clean, educational code
- **Research**: Quick prototyping, experiments
- **Production**: Full training pipelines
- **Modern ML**: PyTorch-style organization

## Next Steps

1. **Read**: `README.md` for complete documentation
2. **Quick Start**: `QUICK_START.md` for common patterns
3. **Examples**: `triton_neural_examples.py` - run 13 examples
4. **Experiment**: Try different attention types
5. **Build**: Create your own models!

## Stats

- **Lines of Code**: ~2000 (main API)
- **Components**: 40+ classes/functions
- **Attention Types**: 6 (industry-leading)
- **Examples**: 13 comprehensive demos
- **Documentation**: Complete
- **Import Styles**: 2 (flexible!)

## Summary

A **complete, unified deep learning API** that includes:

1. All basic neural network layers
2. **6 attention mechanisms** (Flash, RoPE, etc.)
3. Complete transformer architectures
4. VAE and latent space models
5. Full training infrastructure
6. Comprehensive examples & docs
7. **PyTorch-style module organization**

**Happy Deep Learning!**
