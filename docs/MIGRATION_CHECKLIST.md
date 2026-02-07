# Triton Neural - Quick Migration Checklist

## No Migration Required!

Your existing code will continue to work without any changes. This checklist is only if you want to adopt the new PyTorch-style imports.

## Optional Migration Steps

### Step 1: Update Import Statement

**Before:**
```python
from triton_neural import *
```

**After:**
```python
import triton_neural as tn
```

### Step 2: Update Core Components

Add `tn.` prefix to core components:

- [ ] `Sequential` → `tn.Sequential`
- [ ] `Linear` → `tn.Linear`
- [ ] `Conv2D` → `tn.Conv2D`
- [ ] `ReLU` → `tn.ReLU`
- [ ] `GELU` → `tn.GELU`
- [ ] `Dropout` → `tn.Dropout`
- [ ] `BatchNorm` → `tn.BatchNorm`
- [ ] `LayerNorm` → `tn.LayerNorm`
- [ ] `VAE` → `tn.VAE`
- [ ] `ConditionalVAE` → `tn.ConditionalVAE`
- [ ] `LatentSpace` → `tn.LatentSpace`

### Step 3: Update Training Components

Add `tn.train.` prefix to training utilities:

- [ ] `Adam` → `tn.train.Adam`
- [ ] `SGD` → `tn.train.SGD`
- [ ] `RMSprop` → `tn.train.RMSprop`
- [ ] `fit` → `tn.train.fit`
- [ ] `train_step` → `tn.train.train_step`
- [ ] `eval_step` → `tn.train.eval_step`
- [ ] `accuracy` → `tn.train.accuracy`
- [ ] `save_params` → `tn.train.save_params`
- [ ] `load_params` → `tn.train.load_params`

### Step 4: Update Loss Functions

Add `tn.` prefix:

- [ ] `cross_entropy_loss` → `tn.cross_entropy_loss`
- [ ] `mse_loss` → `tn.mse_loss`
- [ ] `binary_cross_entropy_loss` → `tn.binary_cross_entropy_loss`
- [ ] `vae_loss` → `tn.vae_loss`
- [ ] `vae_mse_loss` → `tn.vae_mse_loss`

### Step 5: Update Utilities

Add `tn.util.` prefix:

- [ ] `plot_history` → `tn.util.plot_history`
- [ ] `print_model_summary` → `tn.util.print_model_summary`
- [ ] `print_attention_guide` → `tn.util.print_attention_guide`

### Step 6: Update Transformer Components

Add `tn.transformer.` prefix:

- [ ] `PremadeTransformer` → `tn.transformer.PremadeTransformer`
- [ ] `PremadeTransformerDecoder` → `tn.transformer.PremadeTransformerDecoder`
- [ ] `SelfAttention` → `tn.transformer.SelfAttention`
- [ ] `MaskedSelfAttention` → `tn.transformer.MaskedSelfAttention`
- [ ] `CrossAttention` → `tn.transformer.CrossAttention`
- [ ] `SparseAttention` → `tn.transformer.SparseAttention`
- [ ] `FlashAttention` → `tn.transformer.FlashAttention`
- [ ] `RoPEAttention` → `tn.transformer.RoPEAttention`
- [ ] `TransformerLayer` → `tn.transformer.TransformerLayer`
- [ ] `FeedForward` → `tn.transformer.FeedForward`
- [ ] `PositionalEncoding` → `tn.transformer.PositionalEncoding`

## Example Migration

### Before (Old Style)
```python
from triton_neural import *
from jax import random

model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
optimizer = Adam(learning_rate=0.001)
optimizer.init(model.init(random.PRNGKey(0), (784,)))

params, history = fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    epochs=10,
    loss_fn=cross_entropy_loss
)

plot_history(history)
```

### After (New Style)
```python
import triton_neural as tn
from jax import random

model = tn.Sequential(tn.Linear(784, 128), tn.ReLU(), tn.Linear(128, 10))
optimizer = tn.train.Adam(learning_rate=0.001)
optimizer.init(model.init(random.PRNGKey(0), (784,)))

params, history = tn.train.fit(
    model, params, optimizer,
    train_data=(x_train, y_train),
    epochs=10,
    loss_fn=tn.cross_entropy_loss
)

tn.util.plot_history(history)
```

## Tips

1. **Use Find & Replace**: You can use your editor's find/replace to speed up migration
2. **Migrate Gradually**: Update one file at a time
3. **Test After Each Change**: Run your tests after updating each file
4. **Keep Old Style if Preferred**: There's no requirement to migrate

## Common Find & Replace Patterns

If your code uses the old style, these find/replace patterns can help:

1. `from triton_neural import *` → `import triton_neural as tn`
2. `= Adam(` → `= tn.train.Adam(`
3. `= SGD(` → `= tn.train.SGD(`
4. `= RMSprop(` → `= tn.train.RMSprop(`
5. `fit(` → `tn.train.fit(`
6. `plot_history(` → `tn.util.plot_history(`
7. `print_model_summary(` → `tn.util.print_model_summary(`
8. `PremadeTransformer(` → `tn.transformer.PremadeTransformer(`

**Note**: Be careful with find/replace - make sure you're not replacing function definitions or other code!

## Verification

After migration, verify your code works:

1. Import the module: `import triton_neural as tn`
2. Run your code and check for errors
3. Verify output matches previous version
4. Run `python test_imports.py` to verify installation

## Need Help?

- Check `docs/IMPORT_STYLES.md` for detailed examples
- See `docs/COMPARISON.md` for side-by-side comparisons
- Read `CHANGES.md` for complete change summary

---

**Remember: Migration is optional! Your existing code will continue to work.**
