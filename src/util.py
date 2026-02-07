"""
Triton Neural - Utilities
=========================

Visualization, model inspection, and helper functions.
"""

import jax
import jax.numpy as jnp


def print_model_summary(model, params, input_shape):
    """Print a summary of the model architecture."""
    print("\nModel Summary")
    print("=" * 60)
    
    # Import Sequential here to avoid circular import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from __init__ import Sequential
    
    if not isinstance(model, Sequential):
        print("Model summary only available for Sequential models")
        return
    
    total_params = 0
    shape = input_shape
    
    print(f"{'Layer':<20} {'Output Shape':<20} {'Params':<10}")
    print("-" * 60)
    
    for i, (layer, layer_params) in enumerate(zip(model.layers, params)):
        dummy_input = jnp.zeros((1,) + shape)
        dummy_output = layer(dummy_input, layer_params)
        output_shape = dummy_output.shape[1:]
        
        n_params = sum(p.size for p in jax.tree.leaves(layer_params))
        total_params += n_params
        
        layer_name = layer.__class__.__name__
        print(f"{layer_name:<20} {str(output_shape):<20} {n_params:<10}")
        
        shape = output_shape
    
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print()


def plot_history(history, metric='loss', width=60, height=15):
    """Plot training history using ASCII characters."""
    train_key = f'train_{metric}'
    val_key = f'val_{metric}'
    
    if train_key not in history:
        print(f"No {train_key} found in history")
        return
    
    train_data = history[train_key]
    val_data = history.get(val_key, None)
    
    all_data = train_data + (val_data if val_data else [])
    min_val = min(all_data)
    max_val = max(all_data)
    
    range_val = max_val - min_val
    min_val -= range_val * 0.1
    max_val += range_val * 0.1
    
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    for i, val in enumerate(train_data):
        x = int((i / (len(train_data) - 1)) * (width - 1))
        y = height - 1 - int(((val - min_val) / (max_val - min_val)) * (height - 1))
        y = max(0, min(height - 1, y))
        canvas[y][x] = '●'
    
    if val_data:
        for i, val in enumerate(val_data):
            x = int((i / (len(val_data) - 1)) * (width - 1))
            y = height - 1 - int(((val - min_val) / (max_val - min_val)) * (height - 1))
            y = max(0, min(height - 1, y))
            if canvas[y][x] == '●':
                canvas[y][x] = '◆'
            else:
                canvas[y][x] = '○'
    
    print(f"\n{metric.upper()} History")
    print("─" * (width + 10))
    
    for i, row in enumerate(canvas):
        y_val = max_val - (i / (height - 1)) * (max_val - min_val)
        print(f"{y_val:6.4f} │{''.join(row)}")
    
    print(" " * 7 + "└" + "─" * width)
    print(" " * 8 + f"0{' ' * (width - 10)}{len(train_data)-1}")
    print(f"\n● Train{' ○ Val' if val_data else ''}")
    print()


# ============================================================================
# ATTENTION TYPE GUIDE
# ============================================================================

ATTENTION_GUIDE = """
Triton Neural - Attention Type Selection Guide
==============================================

Choose attention type based on your needs:

1. SelfAttention ('self')
   - Complexity: O(n²)
   - Memory: O(n²)
   - Use when: Sequence length < 512
   - Best for: Standard transformer tasks

2. MaskedSelfAttention ('masked')
   - Complexity: O(n²)
   - Memory: O(n²)
   - Use when: Autoregressive generation (GPT-style)
   - Best for: Language modeling, generation

3. SparseAttention ('sparse')
   - Complexity: O(n√n)
   - Memory: O(n√n)
   - Use when: Sequence length 512-2048
   - Best for: Long documents, efficient processing

4. FlashAttention ('flash')
   - Complexity: O(n²)
   - Memory: O(n) ⭐
   - Use when: Memory-constrained, long sequences (2048+)
   - Best for: Large batches, long contexts

5. RoPEAttention ('rope')
   - Complexity: O(n²)
   - Memory: O(n²)
   - Use when: Building modern LLMs
   - Best for: Better position encoding, no length limit

6. CrossAttention
   - For encoder-decoder architectures
   - Use in seq2seq models, translation

Example Usage:
--------------
# Memory-efficient LLM
from transformer import PremadeTransformer

transformer = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    attention_type='flash'  # O(n) memory!
)

# Modern LLM with RoPE
gpt = PremadeTransformer(
    num_layers=12,
    embed_dim=768,
    attention_type='rope'  # Better positions
)

# Long document processing
doc_encoder = PremadeTransformer(
    num_layers=6,
    embed_dim=512,
    attention_type='sparse'  # Efficient for 512-2048 tokens
)
"""


def print_attention_guide():
    """Print the attention type selection guide."""
    print(ATTENTION_GUIDE)
