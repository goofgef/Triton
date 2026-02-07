# Triton Neural v2.0.2 - Release Notes

## What's New

Triton Neural v2.0.2 introduces PyTorch-style imports while maintaining full backward compatibility with existing code.

## New Features

### 1. PyTorch-Style Module Organization

The package now supports organized imports similar to PyTorch:

```python
import triton_neural as tn

# Use components with module prefixes
optimizer = tn.train.Adam(learning_rate=0.001)
tn.util.plot_history(history)
transformer = tn.transformer.PremadeTransformer(...)
```

### 2. Module Structure

The package is organized into clear modules:

- **Core Module (`tn.*`)**: Layers, activations, VAE, losses
- **Train Module (`tn.train.*`)**: Optimizers, training loops, save/load
- **Util Module (`tn.util.*`)**: Visualization, summaries, guides  
- **Transformer Module (`tn.transformer.*`)**: All attention mechanisms

### 3. Backward Compatibility

The original import style continues to work:

```python
from triton_neural import *

# All existing code remains compatible
optimizer = Adam(learning_rate=0.001)
plot_history(history)
```

## Files Modified

### 1. `src/triton_neural/__init__.py`
- Added imports for submodules (`train`, `util`, `transformer`)
- Re-exported commonly used items from submodules
- Maintained all existing functionality
- Added `__all__` for better namespace control

### 2. `README.md`
- Added section explaining both import styles
- Updated all examples to demonstrate new style
- Added module organization documentation
- Emphasized flexibility of choosing import style

### 3. `docs/QUICK_START.md`
- Added import styles comparison at the top
- Updated all examples to show new style
- Added module organization section
- Maintained backward-compatible examples

### 4. `docs/SUMMARY.md`
- Updated to highlight PyTorch-style imports
- Added module organization information
- Updated usage examples

### 5. `docs/IMPORT_STYLES.md` (NEW)
- Complete migration guide
- Side-by-side comparison of old vs new
- Module organization reference
- Benefits of new style
- Quick reference table

### 6. `test_imports.py` (NEW)
- Automated tests for both import styles
- Verifies compatibility between styles
- Ensures all modules are accessible

## Usage Examples

### Basic Model Training

```python
import triton_neural as tn
from jax import random

# Create model
model = tn.Sequential(
    tn.Linear(784, 128),
    tn.ReLU(),
    tn.Linear(128, 10)
)

# Train
optimizer = tn.train.Adam(learning_rate=0.001)
params, history = tn.train.fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    epochs=10,
    loss_fn=tn.cross_entropy_loss
)

# Visualize
tn.util.plot_history(history)
```

### Transformer Model

```python
import triton_neural as tn

# Create transformer
transformer = tn.transformer.PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    num_heads=12,
    attention_type='flash'
)
```

### VAE

```python
import triton_neural as tn

# Create VAE
vae = tn.VAE(input_dim=784, latent_dim=32)
params = vae.init(rng, (784,))

# Use latent space utilities
latent_space = tn.LatentSpace(vae)
z_interp = latent_space.interpolate(z1, z2, steps=10)
```

## Benefits

### For Users

1. **Better Organization**: Clear separation of concerns
2. **Familiar**: Similar to PyTorch (`torch.nn`, `torch.optim`)
3. **Flexible**: Choose the style that fits your workflow
4. **IDE-Friendly**: Better autocomplete and hints
5. **No Breaking Changes**: All existing code continues to work

### For Code Quality

1. **Modular**: Clear module boundaries
2. **Maintainable**: Easy to find and update components
3. **Extensible**: Easy to add new modules
4. **Professional**: Matches industry standards

## Testing

Run the test script to verify everything works:

```bash
python test_imports.py
```

This tests:
1. PyTorch-style imports work correctly
2. Direct imports still work
3. Both styles reference the same classes
4. All modules are accessible

## Migration Guide

No migration required! Existing code will continue to work without changes.

If you prefer to adopt the new style:

1. Replace `from triton_neural import *` with `import triton_neural as tn`
2. Add `tn.` prefix to core components
3. Use `tn.train.` for training utilities
4. Use `tn.util.` for visualization/inspection
5. Use `tn.transformer.` for transformer components

See `docs/IMPORT_STYLES.md` for detailed examples.

## Version History

- **v2.0.2**: Added PyTorch-style imports, maintained backward compatibility
- **v2.0.0**: Initial unified API with 6 attention types

## Next Steps

1. Try the new import style in your code
2. Read `docs/IMPORT_STYLES.md` for detailed examples
3. Run `test_imports.py` to verify installation
4. Choose the style that fits your workflow best

---

**No action required - all existing code continues to work!**
