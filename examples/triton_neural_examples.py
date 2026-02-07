"""
Triton Neural - Comprehensive Examples
======================================

Examples covering all functionality with the new modular structure.
"""

import jax
import jax.numpy as jnp
from jax import random

# Import from modular files
from __init__ import (
    Sequential, Linear, Conv2D, BatchNorm, Dropout, Flatten, Reshape,
    ReLU, GELU, Sigmoid, Tanh, Softmax, LeakyReLU,
    LayerNorm, RMSNorm,
    VAE, ConditionalVAE, LatentSpace,
    mse_loss, cross_entropy_loss, binary_cross_entropy_loss, vae_loss, vae_mse_loss
)

from transformer import (
    SelfAttention, MaskedSelfAttention, CrossAttention, SparseAttention, FlashAttention, RoPEAttention,
    RotaryPositionalEmbedding,
    FeedForward, TransformerLayer, TransformerDecoderLayer,
    PositionalEncoding, LearnedPositionalEmbedding,
    PremadeTransformer, PremadeTransformerDecoder
)

from train import (
    SGD, Adam, RMSprop,
    train_step, eval_step, fit, accuracy,
    save_params, load_params
)

from util import (
    print_model_summary, plot_history, print_attention_guide
)


# ============================================================================
# EXAMPLE 1: Simple MLP Classifier
# ============================================================================

def example_mlp():
    """Basic multi-layer perceptron for classification."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: MLP Classifier")
    print("=" * 70)
    
    # Create model
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Dropout(0.2),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
        Softmax()
    )
    
    # Initialize
    rng = random.PRNGKey(0)
    params = model.init(rng, (784,))
    
    print("\nModel created:")
    print_model_summary(model, params, (784,))
    
    # Generate dummy data
    rng, data_rng = random.split(rng)
    x_train = random.normal(data_rng, (1000, 784))
    y_train = random.randint(data_rng, (1000,), 0, 10)
    
    # Train
    optimizer = Adam(learning_rate=0.001)
    optimizer.init(params)
    
    print("Training...")
    params, history = fit(
        model, params, optimizer,
        (x_train, y_train),
        epochs=5,
        loss_fn=cross_entropy_loss,
        batch_size=32,
        rng=rng,
        verbose=True
    )
    
    plot_history(history)


# ============================================================================
# EXAMPLE 2: CNN for Image Classification
# ============================================================================

def example_cnn():
    """Convolutional neural network."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CNN for Images")
    print("=" * 70)
    
    model = Sequential(
        Conv2D(1, 32, kernel_size=3),
        ReLU(),
        Conv2D(32, 64, kernel_size=3),
        ReLU(),
        Flatten(),
        Linear(64 * 28 * 28, 128),
        ReLU(),
        Linear(128, 10)
    )
    
    rng = random.PRNGKey(0)
    params = model.init(rng, (28, 28, 1))
    
    print("\nCNN architecture:")
    print_model_summary(model, params, (28, 28, 1))


# ============================================================================
# EXAMPLE 3: Standard Transformer Encoder
# ============================================================================

def example_transformer():
    """Standard BERT-style transformer encoder."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Transformer Encoder (BERT-style)")
    print("=" * 70)
    
    transformer = PremadeTransformer(
        num_layers=6,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        max_len=128,
        dropout=0.1,
        attention_type='self'
    )
    
    rng = random.PRNGKey(0)
    params = transformer.init(rng, (128, 512))
    
    # Test forward pass
    x = random.normal(rng, (2, 128, 512))
    output = transformer(x, params, rng, training=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Standard attention working!")


# ============================================================================
# EXAMPLE 4: GPT-style Masked Transformer
# ============================================================================

def example_gpt():
    """GPT-style decoder with masked attention."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: GPT-style Decoder (Masked Attention)")
    print("=" * 70)
    
    gpt = PremadeTransformer(
        num_layers=12,
        embed_dim=768,
        num_heads=12,
        ff_dim=3072,
        max_len=1024,
        dropout=0.1,
        attention_type='masked',  # Causal masking
        use_learned_pos=True
    )
    
    rng = random.PRNGKey(0)
    params = gpt.init(rng, (1024, 768))
    
    # Test
    x = random.normal(rng, (1, 64, 768))
    output = gpt(x, params, rng, training=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Masked (causal) attention working!")
    print("  Perfect for autoregressive generation!")


# ============================================================================
# EXAMPLE 5: Sparse Attention for Long Sequences
# ============================================================================

def example_sparse_attention():
    """Efficient sparse attention for long documents."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Sparse Attention (Long Sequences)")
    print("=" * 70)
    
    long_encoder = PremadeTransformer(
        num_layers=8,
        embed_dim=512,
        num_heads=8,
        max_len=2048,
        attention_type='sparse',  # O(n√n) complexity
        dropout=0.1
    )
    
    rng = random.PRNGKey(0)
    params = long_encoder.init(rng, (2048, 512))
    
    # Test with long sequence
    x = random.normal(rng, (1, 1024, 512))
    output = long_encoder(x, params, rng, training=False)
    
    print(f"Input shape: {x.shape} (long sequence!)")
    print(f"Output shape: {output.shape}")
    print("✓ Sparse attention working!")
    print("  Complexity: O(n√n) instead of O(n²)")


# ============================================================================
# EXAMPLE 6: Flash Attention (Memory-Efficient)
# ============================================================================

def example_flash_attention():
    """Memory-efficient Flash Attention."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Flash Attention (Memory-Efficient)")
    print("=" * 70)
    
    flash_model = PremadeTransformer(
        num_layers=6,
        embed_dim=512,
        num_heads=8,
        max_len=2048,
        attention_type='flash',  # O(n) memory!
        dropout=0.1
    )
    
    rng = random.PRNGKey(0)
    params = flash_model.init(rng, (2048, 512))
    
    # Test with very long sequence
    x = random.normal(rng, (1, 1024, 512))
    output = flash_model(x, params, rng, training=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Flash attention working!")
    print("  Memory: O(n) instead of O(n²) - HUGE savings!")
    print("  Enables sequences of 2048+ tokens!")


# ============================================================================
# EXAMPLE 7: RoPE Attention (Modern LLMs)
# ============================================================================

def example_rope_attention():
    """Rotary Position Embeddings (RoPE) - Modern LLM approach."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: RoPE Attention (Modern LLMs)")
    print("=" * 70)
    
    rope_model = PremadeTransformer(
        num_layers=12,
        embed_dim=768,
        num_heads=12,
        max_len=2048,
        attention_type='rope',  # Rotary positions
        dropout=0.1
    )
    
    rng = random.PRNGKey(0)
    params = rope_model.init(rng, (2048, 768))
    
    # Test
    x = random.normal(rng, (2, 128, 768))
    output = rope_model(x, params, rng, training=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ RoPE attention working!")
    print("  Used in: LLaMA, PaLM, GPT-NeoX")
    print("  Benefits: Natural relative positions, no length limit")


# ============================================================================
# EXAMPLE 8: Seq2Seq Translation Model
# ============================================================================

def example_seq2seq():
    """Sequence-to-sequence model with encoder-decoder."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Seq2Seq Translation")
    print("=" * 70)
    
    # Encoder
    encoder = PremadeTransformer(
        num_layers=3,
        embed_dim=256,
        num_heads=8,
        max_len=128
    )
    
    # Decoder with cross-attention
    decoder = PremadeTransformerDecoder(
        num_layers=3,
        embed_dim=256,
        num_heads=8,
        max_len=128,
        use_cross_attention=True
    )
    
    rng = random.PRNGKey(0)
    enc_params = encoder.init(rng, (128, 256))
    dec_params = decoder.init(rng, (128, 256))
    
    # Test
    source = random.normal(rng, (2, 64, 256))
    target = random.normal(rng, (2, 64, 256))
    
    # Encode source
    encoder_out = encoder(source, enc_params, rng)
    
    # Decode target
    decoder_out = decoder(target, dec_params, rng, 
                         encoder_output=encoder_out)
    
    print(f"Source shape: {source.shape}")
    print(f"Encoder output: {encoder_out.shape}")
    print(f"Decoder output: {decoder_out.shape}")
    print("✓ Seq2Seq working!")


# ============================================================================
# EXAMPLE 9: Variational Autoencoder (VAE)
# ============================================================================

def example_vae():
    """Variational Autoencoder for latent space learning."""
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Variational Autoencoder (VAE)")
    print("=" * 70)
    
    vae = VAE(
        input_dim=784,
        latent_dim=32,
        encoder_hidden=[256, 128],
        decoder_hidden=[128, 256]
    )
    
    rng = random.PRNGKey(0)
    params = vae.init(rng, (784,))
    
    # Generate data
    x = random.normal(rng, (100, 784))
    
    # Forward pass
    reconstruction, mu, logvar = vae(x, params, rng, training=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {mu.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Compute loss
    loss = vae_loss(reconstruction, x, mu, logvar, beta=1.0)
    print(f"VAE loss: {loss:.4f}")
    print("✓ VAE working!")


# ============================================================================
# EXAMPLE 10: Latent Space Manipulation
# ============================================================================

def example_latent_space():
    """Latent space operations: interpolation, arithmetic."""
    print("\n" + "=" * 70)
    print("EXAMPLE 10: Latent Space Manipulation")
    print("=" * 70)
    
    vae = VAE(input_dim=784, latent_dim=32)
    rng = random.PRNGKey(0)
    params = vae.init(rng, (784,))
    
    latent_space = LatentSpace(vae)
    
    # Encode some data
    x = random.normal(rng, (10, 784))
    latent = latent_space.read(x, params, rng)
    
    print(f"Encoded {x.shape[0]} samples to latent space")
    print(f"Latent dimensions: {latent['mean'].shape}")
    
    # Interpolation
    z1 = latent['sample'][0]
    z2 = latent['sample'][1]
    interpolated = latent_space.interpolate(z1, z2, steps=5)
    
    print(f"\nInterpolated between 2 points: {interpolated.shape}")
    
    # Decode
    decoded = latent_space.write(interpolated, params)
    print(f"Decoded interpolation: {decoded.shape}")
    
    # Spherical interpolation (better for normalized spaces)
    slerp = latent_space.spherical_interpolate(z1, z2, steps=5)
    print(f"SLERP interpolation: {slerp.shape}")
    
    print("✓ Latent space operations working!")


# ============================================================================
# EXAMPLE 11: Conditional VAE
# ============================================================================

def example_conditional_vae():
    """Conditional VAE - generate with class labels."""
    print("\n" + "=" * 70)
    print("EXAMPLE 11: Conditional VAE")
    print("=" * 70)
    
    cvae = ConditionalVAE(
        input_dim=784,
        latent_dim=32,
        num_classes=10
    )
    
    rng = random.PRNGKey(0)
    params = cvae.init(rng, (784,))
    
    # Generate data with labels
    x = random.normal(rng, (100, 784))
    labels = random.randint(rng, (100,), 0, 10)
    
    # Forward pass
    reconstruction, mu, logvar = cvae(x, labels, params, rng, training=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Latent shape: {mu.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Compute loss
    loss = vae_loss(reconstruction, x, mu, logvar)
    print(f"CVAE loss: {loss:.4f}")
    print("✓ Conditional VAE working!")
    print("  Can generate specific classes!")


# ============================================================================
# EXAMPLE 12: Complete Training Pipeline
# ============================================================================

def example_complete_training():
    """Complete end-to-end training example."""
    print("\n" + "=" * 70)
    print("EXAMPLE 12: Complete Training Pipeline")
    print("=" * 70)
    
    # Model
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Dropout(0.3),
        Linear(256, 128),
        ReLU(),
        Dropout(0.3),
        Linear(128, 10)
    )
    
    # Data
    rng = random.PRNGKey(42)
    x_train = random.normal(rng, (1000, 784))
    y_train = random.randint(rng, (1000,), 0, 10)
    x_val = random.normal(rng, (200, 784))
    y_val = random.randint(rng, (200,), 0, 10)
    
    # Initialize
    params = model.init(rng, (784,))
    
    # Optimizer
    optimizer = Adam(learning_rate=0.001)
    optimizer.init(params)
    
    # Train
    print("\nTraining...")
    params, history = fit(
        model, params, optimizer,
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        epochs=10,
        loss_fn=cross_entropy_loss,
        batch_size=32,
        rng=rng,
        verbose=True
    )
    
    # Evaluate
    preds = model(x_val, params, training=False)
    acc = accuracy(preds, y_val)
    print(f"\nFinal validation accuracy: {acc:.4f}")
    
    # Plot
    plot_history(history)
    
    print("\n✓ Training complete!")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRITON NEURAL - COMPREHENSIVE EXAMPLES (Modular Structure)")
    print("=" * 70)
    print("\nRunning examples...")
    
    examples = [
        ("MLP Classifier", example_mlp),
        ("CNN Architecture", example_cnn),
        ("Transformer Encoder", example_transformer),
        ("GPT-style Decoder", example_gpt),
        ("Sparse Attention", example_sparse_attention),
        ("Flash Attention", example_flash_attention),
        ("RoPE Attention", example_rope_attention),
        ("Seq2Seq Model", example_seq2seq),
        ("VAE", example_vae),
        ("Latent Space", example_latent_space),
        ("Conditional VAE", example_conditional_vae),
        ("Complete Training", example_complete_training),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nTriton Neural (Modular) is ready for:")
    print("  ✓ Basic neural networks (MLP, CNN)")
    print("  ✓ Advanced transformers (6 attention types)")
    print("  ✓ Latent space models (VAE, CVAE)")
    print("  ✓ Long sequence processing (Flash, Sparse)")
    print("  ✓ Modern LLM architectures (RoPE)")
    print("  ✓ Complete training pipelines")
    print("\nFiles:")
    print("  • __init__.py    - Core layers, VAE, losses")
    print("  • transformer.py - All attention mechanisms")
    print("  • train.py       - Optimizers, training loops")
    print("  • util.py        - Plotting, model summary")
    print("\nHappy deep learning! 🚀")
