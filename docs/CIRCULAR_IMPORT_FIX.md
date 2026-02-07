# Circular Import Fix - v2.0.3

## Problem
When importing `triton_neural`, there was a circular import error:
```
ImportError: cannot import name 'Module' from partially initialized module 'triton_neural'
```

## Root Cause
The `transformer.py` module was importing `Module`, `LayerNorm`, and `RMSNorm` from `__init__.py`, but `__init__.py` was also importing from `transformer.py`, creating a circular dependency.

## Solution
1. Created a new `core.py` module containing the base `Module` class
2. Added `LayerNorm` and `RMSNorm` directly to `transformer.py` (since they're only used there)
3. Updated imports:
   - `__init__.py` now imports `Module` from `core.py`
   - `transformer.py` now imports `Module` from `core.py` and defines its own `LayerNorm`/`RMSNorm`

## Files Modified
- Created: `src/triton_neural/core.py`
- Modified: `src/triton_neural/__init__.py`
- Modified: `src/triton_neural/transformer.py`

## Testing
After rebuilding the package, test with:
```python
import triton_neural as tn
from jax import random

# This should work now
model = tn.Sequential(tn.Linear(10, 5), tn.ReLU())
rng = random.PRNGKey(0)
params = model.init(rng, (10,))
print("Success!")
```

## Rebuild Instructions
1. Clean old build:
   ```bash
   Remove-Item -Recurse -Force dist\
   Remove-Item -Recurse -Force src\triton_neural.egg-info\
   ```

2. Rebuild:
   ```bash
   python -m build
   ```

3. Upload:
   ```bash
   python -m twine upload dist/*
   ```
