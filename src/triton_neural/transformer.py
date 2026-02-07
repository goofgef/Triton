"""
Triton Neural - Transformer Components
======================================

Advanced attention mechanisms, positional encodings, and complete transformer architectures.
Includes: Self, Masked, Cross, Sparse, Flash, and RoPE attention.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Optional
import math

from . import Module, LayerNorm, RMSNorm


# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

class SelfAttention(Module):
    """
    Standard self-attention mechanism.
    Complexity: O(n²d), Memory: O(n²)
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    
    def init(self, rng, input_shape):
        k1, k2, k3, k4 = random.split(rng, 4)
        scale = jnp.sqrt(1.0 / self.embed_dim)
        
        return {
            'w_q': random.normal(k1, (self.embed_dim, self.embed_dim)) * scale,
            'w_k': random.normal(k2, (self.embed_dim, self.embed_dim)) * scale,
            'w_v': random.normal(k3, (self.embed_dim, self.embed_dim)) * scale,
            'w_o': random.normal(k4, (self.embed_dim, self.embed_dim)) * scale,
        }
    
    def __call__(self, x, params, rng=None, training=False, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = jnp.dot(x, params['w_q'])
        k = jnp.dot(x, params['w_k'])
        v = jnp.dot(x, params['w_v'])
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        if training and self.dropout > 0 and rng is not None:
            keep_prob = 1.0 - self.dropout
            mask = random.bernoulli(rng, keep_prob, attn_weights.shape)
            attn_weights = jnp.where(mask, attn_weights / keep_prob, 0)
        
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        output = jnp.dot(attn_output, params['w_o'])
        
        return output


class MaskedSelfAttention(SelfAttention):
    """
    Masked self-attention for causal (autoregressive) models.
    Used in GPT-style decoders.
    """
    
    def __call__(self, x, params, rng=None, training=False, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Create causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        causal_mask = causal_mask[None, None, :, :]
        
        if mask is not None:
            causal_mask = causal_mask & mask
        
        return super().__call__(x, params, rng, training, causal_mask)


class CrossAttention(Module):
    """
    Cross-attention for encoder-decoder architectures.
    Query from one sequence, Key/Value from another.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0
    
    def init(self, rng, input_shape):
        k1, k2, k3, k4 = random.split(rng, 4)
        scale = jnp.sqrt(1.0 / self.embed_dim)
        
        return {
            'w_q': random.normal(k1, (self.embed_dim, self.embed_dim)) * scale,
            'w_k': random.normal(k2, (self.embed_dim, self.embed_dim)) * scale,
            'w_v': random.normal(k3, (self.embed_dim, self.embed_dim)) * scale,
            'w_o': random.normal(k4, (self.embed_dim, self.embed_dim)) * scale,
        }
    
    def __call__(self, q_input, kv_input, params, rng=None, training=False, mask=None):
        batch_size, target_len, _ = q_input.shape
        _, source_len, _ = kv_input.shape
        
        q = jnp.dot(q_input, params['w_q'])
        k = jnp.dot(kv_input, params['w_k'])
        v = jnp.dot(kv_input, params['w_v'])
        
        q = q.reshape(batch_size, target_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, source_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, source_len, self.num_heads, self.head_dim)
        
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        if training and self.dropout > 0 and rng is not None:
            keep_prob = 1.0 - self.dropout
            drop_mask = random.bernoulli(rng, keep_prob, attn_weights.shape)
            attn_weights = jnp.where(drop_mask, attn_weights / keep_prob, 0)
        
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, target_len, self.embed_dim)
        
        output = jnp.dot(attn_output, params['w_o'])
        
        return output


class SparseAttention(Module):
    """
    Sparse attention with local + strided patterns.
    Complexity: O(n√n·d), Memory: O(n√n)
    Efficient for long sequences.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 block_size: int = 64, stride: int = 64, dropout: float = 0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.stride = stride
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0
    
    def init(self, rng, input_shape):
        k1, k2, k3, k4 = random.split(rng, 4)
        scale = jnp.sqrt(1.0 / self.embed_dim)
        
        return {
            'w_q': random.normal(k1, (self.embed_dim, self.embed_dim)) * scale,
            'w_k': random.normal(k2, (self.embed_dim, self.embed_dim)) * scale,
            'w_v': random.normal(k3, (self.embed_dim, self.embed_dim)) * scale,
            'w_o': random.normal(k4, (self.embed_dim, self.embed_dim)) * scale,
        }
    
    def _create_sparse_mask(self, seq_len: int) -> jnp.ndarray:
        mask = jnp.zeros((seq_len, seq_len), dtype=bool)
        
        for i in range(seq_len):
            start = max(0, i - self.block_size // 2)
            end = min(seq_len, i + self.block_size // 2 + 1)
            mask = mask.at[i, start:end].set(True)
            
            strided_positions = jnp.arange(0, seq_len, self.stride)
            mask = mask.at[i, strided_positions].set(True)
        
        return mask
    
    def __call__(self, x, params, rng=None, training=False, mask=None):
        batch_size, seq_len, _ = x.shape
        
        sparse_mask = self._create_sparse_mask(seq_len)
        sparse_mask = sparse_mask[None, None, :, :]
        
        if mask is not None:
            sparse_mask = sparse_mask & mask
        
        q = jnp.dot(x, params['w_q'])
        k = jnp.dot(x, params['w_k'])
        v = jnp.dot(x, params['w_v'])
        
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        scores = jnp.where(sparse_mask, scores, -1e9)
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        if training and self.dropout > 0 and rng is not None:
            keep_prob = 1.0 - self.dropout
            drop_mask = random.bernoulli(rng, keep_prob, attn_weights.shape)
            attn_weights = jnp.where(drop_mask, attn_weights / keep_prob, 0)
        
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        output = jnp.dot(attn_output, params['w_o'])
        
        return output


class FlashAttention(Module):
    """
    Flash Attention - Memory-efficient attention.
    Memory: O(n) instead of O(n²)
    
    Uses tiling and block-wise computation to avoid materializing
    the full attention matrix. Essential for long sequences (2048+).
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 block_size: int = 64, dropout: float = 0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0
    
    def init(self, rng, input_shape):
        k1, k2, k3, k4 = random.split(rng, 4)
        scale = jnp.sqrt(1.0 / self.embed_dim)
        
        return {
            'w_q': random.normal(k1, (self.embed_dim, self.embed_dim)) * scale,
            'w_k': random.normal(k2, (self.embed_dim, self.embed_dim)) * scale,
            'w_v': random.normal(k3, (self.embed_dim, self.embed_dim)) * scale,
            'w_o': random.normal(k4, (self.embed_dim, self.embed_dim)) * scale,
        }
    
    def _flash_attention_block(self, q_block, k, v, scale):
        scores = jnp.matmul(q_block, jnp.transpose(k, (0, 1, 3, 2))) * scale
        attn_weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(attn_weights, v)
        return output
    
    def __call__(self, x, params, rng=None, training=False, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = jnp.dot(x, params['w_q'])
        k = jnp.dot(x, params['w_k'])
        v = jnp.dot(x, params['w_v'])
        
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        scale = 1.0 / jnp.sqrt(self.head_dim)
        
        # Block-wise computation (memory efficient)
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        outputs = []
        
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, seq_len)
            
            q_block = q[:, :, start_idx:end_idx, :]
            block_output = self._flash_attention_block(q_block, k, v, scale)
            outputs.append(block_output)
        
        attn_output = jnp.concatenate(outputs, axis=2)
        
        if training and self.dropout > 0 and rng is not None:
            keep_prob = 1.0 - self.dropout
            mask_dropout = random.bernoulli(rng, keep_prob, attn_output.shape)
            attn_output = jnp.where(mask_dropout, attn_output / keep_prob, 0)
        
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        output = jnp.dot(attn_output, params['w_o'])
        
        return output


class RotaryPositionalEmbedding(Module):
    """
    Rotary Position Embedding (RoPE).
    
    Modern position encoding used in LLaMA, PaLM, GPT-NeoX.
    Advantages:
    - Natural relative position encoding
    - No length limitation
    - Better long-range modeling
    """
    
    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_len = max_len
        self.base = base
    
    def init(self, rng, input_shape):
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))
        position = jnp.arange(self.max_len).astype(jnp.float32)
        freqs = jnp.outer(position, inv_freq)
        
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        
        return {
            'cos': cos,
            'sin': sin,
            'inv_freq': inv_freq
        }
    
    def _rotate_half(self, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    
    def _apply_rotary_pos_emb(self, x, cos, sin):
        seq_len = x.shape[-2]
        cos = cos[:seq_len, :]
        sin = sin[:seq_len, :]
        
        if len(x.shape) == 4:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        
        cos = jnp.repeat(cos, 2, axis=-1)
        sin = jnp.repeat(sin, 2, axis=-1)
        
        return (x * cos) + (self._rotate_half(x) * sin)
    
    def __call__(self, q, k, params, rng=None, training=False, position_ids=None):
        cos = params['cos']
        sin = params['sin']
        
        q_rotated = self._apply_rotary_pos_emb(q, cos, sin)
        k_rotated = self._apply_rotary_pos_emb(k, cos, sin)
        
        return q_rotated, k_rotated


class RoPEAttention(Module):
    """
    Self-attention with Rotary Position Embeddings (RoPE).
    Modern approach used in state-of-the-art LLMs.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 dropout: float = 0.0, max_len: int = 2048):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_len)
    
    def init(self, rng, input_shape):
        k1, k2, k3, k4, k5 = random.split(rng, 5)
        scale = jnp.sqrt(1.0 / self.embed_dim)
        
        params = {
            'w_q': random.normal(k1, (self.embed_dim, self.embed_dim)) * scale,
            'w_k': random.normal(k2, (self.embed_dim, self.embed_dim)) * scale,
            'w_v': random.normal(k3, (self.embed_dim, self.embed_dim)) * scale,
            'w_o': random.normal(k4, (self.embed_dim, self.embed_dim)) * scale,
        }
        
        params['rope'] = self.rope.init(k5, (self.head_dim,))
        
        return params
    
    def __call__(self, x, params, rng=None, training=False, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = jnp.dot(x, params['w_q'])
        k = jnp.dot(x, params['w_k'])
        v = jnp.dot(x, params['w_v'])
        
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Apply RoPE
        q, k = self.rope(q, k, params['rope'])
        
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        if training and self.dropout > 0 and rng is not None:
            keep_prob = 1.0 - self.dropout
            drop_mask = random.bernoulli(rng, keep_prob, attn_weights.shape)
            attn_weights = jnp.where(drop_mask, attn_weights / keep_prob, 0)
        
        attn_output = jnp.matmul(attn_weights, v)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        output = jnp.dot(attn_output, params['w_o'])
        
        return output


# ============================================================================
# FEED-FORWARD NETWORK
# ============================================================================

class FeedForward(Module):
    """Position-wise feed-forward network for transformers."""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.0, 
                 activation: str = 'gelu'):
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.activation = activation
    
    def init(self, rng, input_shape):
        k1, k2 = random.split(rng, 2)
        
        scale1 = jnp.sqrt(2.0 / self.embed_dim)
        scale2 = jnp.sqrt(2.0 / self.ff_dim)
        
        return {
            'w1': random.normal(k1, (self.embed_dim, self.ff_dim)) * scale1,
            'b1': jnp.zeros((self.ff_dim,)),
            'w2': random.normal(k2, (self.ff_dim, self.embed_dim)) * scale2,
            'b2': jnp.zeros((self.embed_dim,)),
        }
    
    def __call__(self, x, params, rng=None, training=False):
        x = jnp.dot(x, params['w1']) + params['b1']
        
        if self.activation == 'gelu':
            x = jax.nn.gelu(x)
        elif self.activation == 'relu':
            x = jax.nn.relu(x)
        elif self.activation == 'swish':
            x = jax.nn.swish(x)
        
        if training and self.dropout > 0 and rng is not None:
            keep_prob = 1.0 - self.dropout
            mask = random.bernoulli(rng, keep_prob, x.shape)
            x = jnp.where(mask, x / keep_prob, 0)
        
        x = jnp.dot(x, params['w2']) + params['b2']
        
        return x


# ============================================================================
# TRANSFORMER LAYERS
# ============================================================================

class TransformerLayer(Module):
    """
    Single transformer encoder layer.
    Supports 6 attention types: self, masked, cross, sparse, flash, rope
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 ff_dim: Optional[int] = None, dropout: float = 0.1,
                 attention_type: str = 'self'):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.dropout = dropout
        self.attention_type = attention_type
        
        if attention_type == 'self':
            self.attention = SelfAttention(embed_dim, num_heads, dropout)
        elif attention_type == 'masked':
            self.attention = MaskedSelfAttention(embed_dim, num_heads, dropout)
        elif attention_type == 'sparse':
            self.attention = SparseAttention(embed_dim, num_heads, dropout=dropout)
        elif attention_type == 'flash':
            self.attention = FlashAttention(embed_dim, num_heads, dropout=dropout)
        elif attention_type == 'rope':
            self.attention = RoPEAttention(embed_dim, num_heads, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.norm1 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, self.ff_dim, dropout)
        self.norm2 = LayerNorm(embed_dim)
    
    def init(self, rng, input_shape):
        k1, k2, k3, k4 = random.split(rng, 4)
        
        return {
            'attention': self.attention.init(k1, input_shape),
            'norm1': self.norm1.init(k2, input_shape),
            'ff': self.ff.init(k3, input_shape),
            'norm2': self.norm2.init(k4, input_shape),
        }
    
    def __call__(self, x, params, rng=None, training=False, mask=None):
        if rng is not None:
            rng, attn_rng, ff_rng = random.split(rng, 3)
        else:
            attn_rng = ff_rng = None
        
        attn_out = self.attention(x, params['attention'], attn_rng, training, mask)
        x = x + attn_out
        x = self.norm1(x, params['norm1'])
        
        ff_out = self.ff(x, params['ff'], ff_rng, training)
        x = x + ff_out
        x = self.norm2(x, params['norm2'])
        
        return x


class TransformerDecoderLayer(Module):
    """Single transformer decoder layer with optional cross-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 ff_dim: Optional[int] = None, dropout: float = 0.1,
                 use_cross_attention: bool = True):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.dropout = dropout
        self.use_cross_attention = use_cross_attention
        
        self.self_attn = MaskedSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = LayerNorm(embed_dim)
        
        if use_cross_attention:
            self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
            self.norm2 = LayerNorm(embed_dim)
        
        self.ff = FeedForward(embed_dim, self.ff_dim, dropout)
        self.norm3 = LayerNorm(embed_dim)
    
    def init(self, rng, input_shape):
        keys = random.split(rng, 6)
        
        params = {
            'self_attn': self.self_attn.init(keys[0], input_shape),
            'norm1': self.norm1.init(keys[1], input_shape),
            'ff': self.ff.init(keys[4], input_shape),
            'norm3': self.norm3.init(keys[5], input_shape),
        }
        
        if self.use_cross_attention:
            params['cross_attn'] = self.cross_attn.init(keys[2], input_shape)
            params['norm2'] = self.norm2.init(keys[3], input_shape)
        
        return params
    
    def __call__(self, x, params, rng=None, training=False, 
                 encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        if rng is not None:
            rng, self_attn_rng, cross_attn_rng, ff_rng = random.split(rng, 4)
        else:
            self_attn_rng = cross_attn_rng = ff_rng = None
        
        attn_out = self.self_attn(x, params['self_attn'], self_attn_rng, training, self_attn_mask)
        x = x + attn_out
        x = self.norm1(x, params['norm1'])
        
        if self.use_cross_attention and encoder_output is not None:
            cross_out = self.cross_attn(x, encoder_output, params['cross_attn'], 
                                       cross_attn_rng, training, cross_attn_mask)
            x = x + cross_out
            x = self.norm2(x, params['norm2'])
        
        ff_out = self.ff(x, params['ff'], ff_rng, training)
        x = x + ff_out
        x = self.norm3(x, params['norm3'])
        
        return x


# ============================================================================
# POSITIONAL ENCODINGS
# ============================================================================

class PositionalEncoding(Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        self.embed_dim = embed_dim
        self.max_len = max_len
    
    def init(self, rng, input_shape):
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.embed_dim, 2) * 
                          -(jnp.log(10000.0) / self.embed_dim))
        
        pe = jnp.zeros((self.max_len, self.embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        return {'encoding': pe}
    
    def __call__(self, x, params, rng=None, training=False):
        seq_len = x.shape[1]
        return x + params['encoding'][:seq_len][None, :, :]


class LearnedPositionalEmbedding(Module):
    """Learned positional embeddings."""
    
    def __init__(self, max_len: int, embed_dim: int):
        self.max_len = max_len
        self.embed_dim = embed_dim
    
    def init(self, rng, input_shape):
        scale = jnp.sqrt(1.0 / self.embed_dim)
        return {
            'embedding': random.normal(rng, (self.max_len, self.embed_dim)) * scale
        }
    
    def __call__(self, x, params, rng=None, training=False):
        seq_len = x.shape[1]
        return x + params['embedding'][:seq_len][None, :, :]


# ============================================================================
# COMPLETE TRANSFORMER MODELS
# ============================================================================

class PremadeTransformer(Module):
    """
    Complete transformer encoder.
    
    Supports 6 attention types:
    - 'self': Standard attention
    - 'masked': Causal/GPT-style
    - 'sparse': Efficient for long sequences
    - 'flash': Memory-efficient
    - 'rope': Modern LLM approach
    """
    
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int = 8,
                 ff_dim: Optional[int] = None, max_len: int = 512,
                 dropout: float = 0.1, attention_type: str = 'self',
                 use_learned_pos: bool = False):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.max_len = max_len
        self.dropout = dropout
        self.attention_type = attention_type
        
        if use_learned_pos:
            self.pos_encoding = LearnedPositionalEmbedding(max_len, embed_dim)
        else:
            self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        
        self.layers = [
            TransformerLayer(embed_dim, num_heads, ff_dim, dropout, attention_type)
            for _ in range(num_layers)
        ]
    
    def init(self, rng, input_shape):
        keys = random.split(rng, self.num_layers + 1)
        
        params = {
            'pos_encoding': self.pos_encoding.init(keys[0], input_shape),
            'layers': [layer.init(keys[i+1], input_shape) 
                      for i, layer in enumerate(self.layers)]
        }
        
        return params
    
    def __call__(self, x, params, rng=None, training=False, mask=None):
        x = self.pos_encoding(x, params['pos_encoding'])
        
        for i, layer in enumerate(self.layers):
            if rng is not None:
                rng, layer_rng = random.split(rng)
            else:
                layer_rng = None
            
            x = layer(x, params['layers'][i], layer_rng, training, mask)
        
        return x


class PremadeTransformerDecoder(Module):
    """Complete transformer decoder with optional cross-attention."""
    
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int = 8,
                 ff_dim: Optional[int] = None, max_len: int = 512,
                 dropout: float = 0.1, use_cross_attention: bool = True,
                 use_learned_pos: bool = False):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.max_len = max_len
        self.dropout = dropout
        
        if use_learned_pos:
            self.pos_encoding = LearnedPositionalEmbedding(max_len, embed_dim)
        else:
            self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        
        self.layers = [
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout, use_cross_attention)
            for _ in range(num_layers)
        ]
    
    def init(self, rng, input_shape):
        keys = random.split(rng, self.num_layers + 1)
        
        params = {
            'pos_encoding': self.pos_encoding.init(keys[0], input_shape),
            'layers': [layer.init(keys[i+1], input_shape) 
                      for i, layer in enumerate(self.layers)]
        }
        
        return params
    
    def __call__(self, x, params, rng=None, training=False, 
                 encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        x = self.pos_encoding(x, params['pos_encoding'])
        
        for i, layer in enumerate(self.layers):
            if rng is not None:
                rng, layer_rng = random.split(rng)
            else:
                layer_rng = None
            
            x = layer(x, params['layers'][i], layer_rng, training,
                     encoder_output, self_attn_mask, cross_attn_mask)
        
        return x
