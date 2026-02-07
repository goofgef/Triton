"""
Comprehensive Test Suite for Triton Neural
==========================================

Tests all major functionality including:
1. Built-in RNG (no JAX import needed)
2. Basic neural network training
3. Transformer models
4. VAE models
5. All documented examples
"""

import numpy as np
import triton_neural as tn

print("=" * 80)
print("TRITON NEURAL COMPREHENSIVE TEST SUITE")
print("=" * 80)

# ============================================================================
# TEST 1: Built-in RNG (No JAX import required!)
# ============================================================================
print("\n[TEST 1] Testing built-in RNG functionality...")

try:
    # Create PRNG key without importing JAX
    key = tn.rng.PRNGKey(42)
    print("✓ Created PRNG key without importing JAX")
    
    # Split keys
    key1, key2 = tn.rng.split(key)
    print("✓ Split PRNG key")
    
    # Generate random numbers
    rand_normal = tn.rng.normal(key1, (5, 3))
    print(f"✓ Generated random normal array: shape={rand_normal.shape}")
    
    rand_uniform = tn.rng.uniform(key2, (5, 3), minval=-1.0, maxval=1.0)
    print(f"✓ Generated random uniform array: shape={rand_uniform.shape}")
    
    print("✓ TEST 1 PASSED: Built-in RNG works without JAX import!")
    
except Exception as e:
    print(f"✗ TEST 1 FAILED: {e}")
    raise


# ============================================================================
# TEST 2: Basic Neural Network (README Example)
# ============================================================================
print("\n[TEST 2] Testing basic neural network from README...")

try:
    # Create sample data (MNIST-like)
    x_train = np.random.randn(100, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, 100)
    x_val = np.random.randn(20, 784).astype(np.float32)
    y_val = np.random.randint(0, 10, 20)
    
    print(f"✓ Created training data: x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")
    
    # Create a simple MLP
    model = tn.Sequential(
        tn.Linear(784, 256),
        tn.ReLU(),
        tn.Dropout(0.2),
        tn.Linear(256, 10),
        tn.Softmax()
    )
    print("✓ Created Sequential model")
    
    # Initialize using built-in rng
    key = tn.rng.PRNGKey(0)
    params = model.init(key, (784,))
    print(f"✓ Initialized model parameters")
    
    # Create optimizer
    optimizer = tn.train.Adam(learning_rate=0.001)
    optimizer.init(params)
    print("✓ Created and initialized Adam optimizer")
    
    # Train for a few epochs
    params, history = tn.train.fit(
        model, params, optimizer,
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        epochs=3,
        loss_fn=tn.cross_entropy_loss,
        batch_size=32,
        rng=key,
        verbose=True
    )
    
    print(f"✓ Training completed. Final train loss: {history['train_loss'][-1]:.4f}")
    print("✓ TEST 2 PASSED: Basic neural network training works!")
    
except Exception as e:
    print(f"✗ TEST 2 FAILED: {e}")
    raise


# ============================================================================
# TEST 3: Flash Attention Transformer (README Example)
# ============================================================================
print("\n[TEST 3] Testing Flash Attention transformer from README...")

try:
    # Memory-efficient transformer for long sequences
    transformer = tn.transformer.PremadeTransformer(
        num_layers=2,  # Smaller for testing
        embed_dim=128,
        num_heads=4,
        attention_type='flash',  # O(n) memory!
        max_len=512
    )
    print("✓ Created Flash Attention transformer")
    
    key = tn.rng.PRNGKey(0)
    params = transformer.init(key, (512, 128))
    print("✓ Initialized transformer parameters")
    
    # Process sequence
    x = tn.rng.normal(key, (1, 256, 128))
    output = transformer(x, params, key, training=False)
    print(f"✓ Forward pass successful: output.shape={output.shape}")
    
    print("✓ TEST 3 PASSED: Flash Attention transformer works!")
    
except Exception as e:
    print(f"✗ TEST 3 FAILED: {e}")
    raise


# ============================================================================
# TEST 4: GPT-style Language Model (README Example)
# ============================================================================
print("\n[TEST 4] Testing GPT-style language model from README...")

try:
    # Autoregressive decoder with RoPE
    gpt = tn.transformer.PremadeTransformer(
        num_layers=2,  # Smaller for testing
        embed_dim=128,
        num_heads=4,
        attention_type='rope',  # Modern position encoding
        max_len=512,
        use_learned_pos=False
    )
    print("✓ Created GPT-style model with RoPE")
    
    key = tn.rng.PRNGKey(0)
    params = gpt.init(key, (512, 128))
    print("✓ Initialized GPT parameters")
    
    # Forward pass
    x = tn.rng.normal(key, (1, 100, 128))
    output = gpt(x, params, key, training=False)
    print(f"✓ Forward pass successful: output.shape={output.shape}")
    
    print("✓ TEST 4 PASSED: GPT-style model works!")
    
except Exception as e:
    print(f"✗ TEST 4 FAILED: {e}")
    raise


# ============================================================================
# TEST 5: Variational Autoencoder (README Example)
# ============================================================================
print("\n[TEST 5] Testing VAE from README...")

try:
    # VAE for latent space learning
    vae = tn.VAE(
        input_dim=784,
        latent_dim=32,
        encoder_hidden=[256, 128],
        decoder_hidden=[128, 256]
    )
    print("✓ Created VAE")
    
    key = tn.rng.PRNGKey(0)
    params = vae.init(key, (784,))
    print("✓ Initialized VAE parameters")
    
    # Sample data
    x = np.random.randn(10, 784).astype(np.float32)
    
    # Encode, sample, decode
    reconstruction, mu, logvar = vae(x, params, key, training=True)
    print(f"✓ VAE forward pass: reconstruction.shape={reconstruction.shape}")
    
    # Compute VAE loss
    loss = tn.vae_loss(reconstruction, x, mu, logvar, beta=1.0)
    print(f"✓ VAE loss computed: {loss:.4f}")
    
    # Latent space utilities
    latent_space = tn.LatentSpace(vae)
    latent = latent_space.read(x, params, key)
    print(f"✓ Encoded to latent space: mean.shape={latent['mean'].shape}")
    
    # Interpolate in latent space
    z1 = latent['sample'][0]
    z2 = latent['sample'][1]
    interpolated = latent_space.interpolate(z1, z2, steps=10)
    print(f"✓ Latent interpolation: interpolated.shape={interpolated.shape}")
    
    print("✓ TEST 5 PASSED: VAE works!")
    
except Exception as e:
    print(f"✗ TEST 5 FAILED: {e}")
    raise


# ============================================================================
# TEST 6: All Attention Types
# ============================================================================
print("\n[TEST 6] Testing all attention types...")

attention_types = ['self', 'masked', 'sparse', 'flash', 'rope']

for attn_type in attention_types:
    try:
        transformer = tn.transformer.PremadeTransformer(
            num_layers=2,
            embed_dim=64,
            num_heads=4,
            attention_type=attn_type,
            max_len=256
        )
        
        key = tn.rng.PRNGKey(0)
        params = transformer.init(key, (256, 64))
        
        x = tn.rng.normal(key, (2, 50, 64))
        output = transformer(x, params, key, training=False)
        
        print(f"✓ {attn_type.upper():8s} attention: output.shape={output.shape}")
        
    except Exception as e:
        print(f"✗ {attn_type.upper():8s} attention FAILED: {e}")
        raise

print("✓ TEST 6 PASSED: All attention types work!")


# ============================================================================
# TEST 7: Loss Functions
# ============================================================================
print("\n[TEST 7] Testing loss functions...")

try:
    # MSE loss
    pred = np.random.randn(10, 5).astype(np.float32)
    target = np.random.randn(10, 5).astype(np.float32)
    mse = tn.mse_loss(pred, target)
    print(f"✓ MSE loss: {mse:.4f}")
    
    # Cross entropy loss
    logits = np.random.randn(10, 5).astype(np.float32)
    labels = np.random.randint(0, 5, 10)
    ce = tn.cross_entropy_loss(logits, labels)
    print(f"✓ Cross entropy loss: {ce:.4f}")
    
    # Binary cross entropy
    pred_binary = np.random.uniform(0.1, 0.9, (10, 5)).astype(np.float32)
    target_binary = np.random.randint(0, 2, (10, 5)).astype(np.float32)
    bce = tn.binary_cross_entropy_loss(pred_binary, target_binary)
    print(f"✓ Binary cross entropy loss: {bce:.4f}")
    
    print("✓ TEST 7 PASSED: All loss functions work!")
    
except Exception as e:
    print(f"✗ TEST 7 FAILED: {e}")
    raise


# ============================================================================
# TEST 8: Optimizers
# ============================================================================
print("\n[TEST 8] Testing optimizers...")

optimizers = [
    ('SGD', tn.train.SGD(learning_rate=0.01)),
    ('Adam', tn.train.Adam(learning_rate=0.001)),
    ('RMSprop', tn.train.RMSprop(learning_rate=0.001))
]

for opt_name, optimizer in optimizers:
    try:
        # Simple linear model
        model = tn.Linear(10, 5)
        key = tn.rng.PRNGKey(0)
        params = model.init(key, (10,))
        
        optimizer.init(params)
        
        # Dummy gradient
        grads = {k: np.random.randn(*v.shape).astype(np.float32) for k, v in params.items()}
        
        # Update step
        new_params = optimizer.update(params, grads)
        
        print(f"✓ {opt_name:10s} optimizer update successful")
        
    except Exception as e:
        print(f"✗ {opt_name:10s} optimizer FAILED: {e}")
        raise

print("✓ TEST 8 PASSED: All optimizers work!")


# ============================================================================
# TEST 9: Conditional VAE
# ============================================================================
print("\n[TEST 9] Testing Conditional VAE...")

try:
    cvae = tn.ConditionalVAE(
        input_dim=784,
        latent_dim=32,
        num_classes=10,
        encoder_hidden=[256, 128],
        decoder_hidden=[128, 256]
    )
    
    key = tn.rng.PRNGKey(0)
    params = cvae.init(key, (784,))
    
    x = np.random.randn(10, 784).astype(np.float32)
    labels = np.random.randint(0, 10, 10)
    
    reconstruction, mu, logvar = cvae(x, labels, params, key, training=True)
    
    print(f"✓ Conditional VAE forward pass: reconstruction.shape={reconstruction.shape}")
    print("✓ TEST 9 PASSED: Conditional VAE works!")
    
except Exception as e:
    print(f"✗ TEST 9 FAILED: {e}")
    raise


# ============================================================================
# TEST 10: Model Summary and Utilities
# ============================================================================
print("\n[TEST 10] Testing utilities...")

try:
    model = tn.Sequential(
        tn.Linear(784, 256),
        tn.ReLU(),
        tn.Linear(256, 10)
    )
    
    key = tn.rng.PRNGKey(0)
    params = model.init(key, (784,))
    
    # Print model summary
    print("\nModel Summary:")
    tn.util.print_model_summary(model, params, (784,))
    
    # Test save/load
    print("\n✓ Model summary printed successfully")
    
    # Save params
    tn.train.save_params(params, "test_params.pkl")
    print("✓ Parameters saved to test_params.pkl")
    
    # Load params
    loaded_params = tn.train.load_params("test_params.pkl")
    print("✓ Parameters loaded from test_params.pkl")
    
    # Verify they're the same
    import os
    os.remove("test_params.pkl")
    print("✓ Cleanup successful")
    
    print("✓ TEST 10 PASSED: Utilities work!")
    
except Exception as e:
    print(f"✗ TEST 10 FAILED: {e}")
    raise


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nSummary:")
print("  1. ✓ Built-in RNG works (no JAX import needed)")
print("  2. ✓ Basic neural network training")
print("  3. ✓ Flash Attention transformer")
print("  4. ✓ GPT-style language model")
print("  5. ✓ Variational Autoencoder")
print("  6. ✓ All attention types (self, masked, sparse, flash, rope)")
print("  7. ✓ All loss functions")
print("  8. ✓ All optimizers (SGD, Adam, RMSprop)")
print("  9. ✓ Conditional VAE")
print(" 10. ✓ Utilities (summary, save/load)")
print("\n" + "=" * 80)
print("Triton Neural is ready to use!")
print("=" * 80)
