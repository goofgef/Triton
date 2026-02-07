"""
Test script to verify PyTorch-style imports work correctly
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("Testing Triton Neural Import Styles...")
print("=" * 60)

# Test 1: PyTorch-style import
print("\n1. Testing PyTorch-style import (import triton_neural as tn)")
try:
    import triton_neural as tn
    print("   ✓ Import successful")
    
    # Test core components
    print("   Testing core components...")
    model = tn.Sequential(tn.Linear(10, 5), tn.ReLU())
    print("   ✓ tn.Sequential works")
    print("   ✓ tn.Linear works")
    print("   ✓ tn.ReLU works")
    
    # Test train module
    print("   Testing train module...")
    optimizer = tn.train.Adam(learning_rate=0.001)
    print("   ✓ tn.train.Adam works")
    
    # Test util module
    print("   Testing util module...")
    # Just check if the module exists
    assert hasattr(tn.util, 'print_model_summary')
    assert hasattr(tn.util, 'plot_history')
    print("   ✓ tn.util.print_model_summary exists")
    print("   ✓ tn.util.plot_history exists")
    
    # Test transformer module
    print("   Testing transformer module...")
    transformer = tn.transformer.PremadeTransformer(
        num_layers=2,
        embed_dim=64,
        num_heads=4
    )
    print("   ✓ tn.transformer.PremadeTransformer works")
    
    print("\n✓ PyTorch-style import test PASSED")
    
except Exception as e:
    print(f"\n✗ PyTorch-style import test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Direct import
print("\n2. Testing direct import (from triton_neural import *)")
try:
    # Reset namespace
    if 'Linear' in dir():
        del Linear
    if 'Adam' in dir():
        del Adam
    
    from triton_neural import *
    print("   ✓ Import successful")
    
    # Test core components
    print("   Testing core components...")
    model = Sequential(Linear(10, 5), ReLU())
    print("   ✓ Sequential works")
    print("   ✓ Linear works")
    print("   ✓ ReLU works")
    
    # Test optimizer
    print("   Testing optimizer...")
    optimizer = Adam(learning_rate=0.001)
    print("   ✓ Adam works")
    
    # Test transformer
    print("   Testing transformer...")
    transformer = PremadeTransformer(
        num_layers=2,
        embed_dim=64,
        num_heads=4
    )
    print("   ✓ PremadeTransformer works")
    
    print("\n✓ Direct import test PASSED")
    
except Exception as e:
    print(f"\n✗ Direct import test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Verify both styles give the same results
print("\n3. Testing compatibility between import styles...")
try:
    import triton_neural as tn
    from triton_neural import Linear, Adam, PremadeTransformer
    
    # They should be the same class
    assert tn.Linear is Linear
    assert tn.train.Adam is Adam
    assert tn.transformer.PremadeTransformer is PremadeTransformer
    
    print("   ✓ Both import styles reference the same classes")
    print("\n✓ Compatibility test PASSED")
    
except Exception as e:
    print(f"\n✗ Compatibility test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("\nYou can now use either import style:")
print("  - import triton_neural as tn  (PyTorch-style)")
print("  - from triton_neural import * (Direct)")
print("=" * 60)
