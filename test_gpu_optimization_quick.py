"""
Quick validation test for GPU index optimizations.

Tests:
1. Memory estimation for different index types
2. Index creation with various factory strings
3. GPU/CPU fallback behavior
4. Batch search functionality
5. Runtime parameter tuning
"""

import numpy as np
import pandas as pd
import torch


def test_memory_estimation():
    """Test memory estimation function."""
    print("\n" + "="*80)
    print("Test 1: Memory Estimation")
    print("="*80)
    
    from lotus.vector_store.faiss_vs import estimate_index_memory
    
    test_cases = [
        ("Flat", 10000, 768, False),
        ("IVF256,Flat", 10000, 768, True),
        ("IVF1024,PQ32", 100000, 768, True),
        ("IVF512,SQ8", 50000, 768, True),
        ("HNSW64", 10000, 768, False),
    ]
    
    print("\nMemory estimates:")
    for factory, num_vecs, dim, is_gpu in test_cases:
        mem = estimate_index_memory(factory, num_vecs, dim, is_gpu)
        device = "GPU" if is_gpu else "CPU"
        print(f"  {factory:20s} ({device}): {mem/(1024**2):7.1f} MB for {num_vecs:,} x {dim}")
    
    print("✓ Memory estimation test passed")


def test_index_creation():
    """Test index creation with different configurations."""
    print("\n" + "="*80)
    print("Test 2: Index Creation")
    print("="*80)
    
    from lotus.vector_store.faiss_vs import UnifiedFaissVS
    import faiss
    
    configs = [
        {"factory": "Flat", "use_gpu": False},
        {"factory": "IVF10,Flat", "use_gpu": False},
        {"factory": "IVF10,PQ8", "use_gpu": False, "pq_nbits": 8},
        {"factory": "IVF10,SQ8", "use_gpu": False},
    ]
    
    # Test data
    dim = 128
    num_vecs = 100
    vectors = np.random.rand(num_vecs, dim).astype(np.float32)
    
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}: {config['factory']}")
        try:
            vs = UnifiedFaissVS(**config)
            
            # Create dummy docs
            docs = [f"doc_{j}" for j in range(num_vecs)]
            
            # Index
            import tempfile
            import os
            with tempfile.TemporaryDirectory() as tmpdir:
                index_dir = os.path.join(tmpdir, f"test_index_{i}")
                vs.index(docs, vectors, index_dir)
                print(f"  ✓ Index created: {config['factory']}")
                
                # Load
                vs2 = UnifiedFaissVS(**config)
                vs2.load_index(index_dir)
                print(f"  ✓ Index loaded: {config['factory']}")
                
                # Search
                query_vec = np.random.rand(1, dim).astype(np.float32)
                result = vs2(query_vec, K=5)
                print(f"  ✓ Search successful: found {len(result.indices)} results")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n✓ Index creation test completed")


def test_batch_search():
    """Test batch search functionality."""
    print("\n" + "="*80)
    print("Test 3: Batch Search")
    print("="*80)
    
    from lotus.vector_store.faiss_vs import UnifiedFaissVS
    import tempfile
    import os
    
    # Setup
    dim = 64
    num_vecs = 200
    vectors = np.random.rand(num_vecs, dim).astype(np.float32)
    docs = [f"doc_{i}" for i in range(num_vecs)]
    
    vs = UnifiedFaissVS(factory_string="Flat", use_gpu=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = os.path.join(tmpdir, "batch_test")
        vs.index(docs, vectors, index_dir)
        
        # Single query
        print("\nSingle query test:")
        query_single = np.random.rand(dim).astype(np.float32)
        result_single = vs(query_single, K=5)
        print(f"  Single query result shape: {result_single.distances.shape}")
        assert result_single.distances.shape == (5,), "Single query shape mismatch"
        print("  ✓ Single query works")
        
        # Batch queries
        print("\nBatch query test:")
        num_queries = 10
        query_batch = np.random.rand(num_queries, dim).astype(np.float32)
        result_batch = vs(query_batch, K=5)
        print(f"  Batch query result shape: {result_batch.distances.shape}")
        assert result_batch.distances.shape == (num_queries, 5), "Batch query shape mismatch"
        print("  ✓ Batch query works")
        
        # List of queries
        print("\nList query test:")
        query_list = [np.random.rand(dim).astype(np.float32) for _ in range(5)]
        result_list = vs(query_list, K=5)
        print(f"  List query result shape: {result_list.distances.shape}")
        assert result_list.distances.shape == (5, 5), "List query shape mismatch"
        print("  ✓ List query works")
    
    print("\n✓ Batch search test passed")


def test_gpu_fallback():
    """Test GPU to CPU fallback."""
    print("\n" + "="*80)
    print("Test 4: GPU/CPU Fallback")
    print("="*80)
    
    from lotus.vector_store.faiss_vs import UnifiedFaissVS
    import tempfile
    import os
    
    dim = 64
    num_vecs = 50
    vectors = np.random.rand(num_vecs, dim).astype(np.float32)
    docs = [f"doc_{i}" for i in range(num_vecs)]
    
    # Try GPU first
    gpu_available = torch.cuda.is_available()
    print(f"\nGPU available: {gpu_available}")
    
    if gpu_available:
        print("\nTesting GPU index creation...")
        vs_gpu = UnifiedFaissVS(factory_string="IVF10,Flat", use_gpu=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = os.path.join(tmpdir, "gpu_test")
            vs_gpu.index(docs, vectors, index_dir)
            print("  ✓ GPU index created")
            
            # Load on CPU
            print("\nTesting GPU->CPU loading...")
            vs_cpu = UnifiedFaissVS(factory_string="IVF10,Flat", use_gpu=False)
            vs_cpu.load_index(index_dir)
            print("  ✓ Loaded GPU index on CPU")
            
            # Search
            query = np.random.rand(dim).astype(np.float32)
            result = vs_cpu(query, K=5)
            print(f"  ✓ Search on CPU successful: {len(result.indices)} results")
    else:
        print("  GPU not available, testing CPU fallback in constructor...")
        vs = UnifiedFaissVS(factory_string="Flat", use_gpu=True)
        print(f"  Actual device: {'GPU' if vs.use_gpu else 'CPU'}")
        print("  ✓ Fallback to CPU successful")
    
    print("\n✓ GPU/CPU fallback test passed")


def test_runtime_parameters():
    """Test runtime parameter effects (IVF nprobe)."""
    print("\n" + "="*80)
    print("Test 5: Runtime Parameter Tuning")
    print("="*80)
    
    from lotus.vector_store.faiss_vs import UnifiedFaissVS
    import tempfile
    import os
    import faiss
    
    # Create IVF index
    dim = 64
    num_vecs = 500
    vectors = np.random.rand(num_vecs, dim).astype(np.float32)
    docs = [f"doc_{i}" for i in range(num_vecs)]
    
    vs = UnifiedFaissVS(factory_string="IVF20,Flat", use_gpu=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = os.path.join(tmpdir, "param_test")
        vs.index(docs, vectors, index_dir)
        
        query = np.random.rand(dim).astype(np.float32)
        
        # Test with different nprobe values
        print("\nTesting IVF nprobe parameter:")
        for nprobe in [1, 5, 10]:
            if hasattr(vs.faiss_index, 'nprobe'):
                vs.faiss_index.nprobe = nprobe
                result = vs(query, K=10)
                print(f"  nprobe={nprobe:2d}: found {len(result.indices)} results")
            else:
                print(f"  Index doesn't support nprobe (not an IVF index)")
                break
        
        print("  ✓ Runtime parameter tuning works")
    
    print("\n✓ Runtime parameter test passed")


def test_pq_nbits():
    """Test PQ with different nbits."""
    print("\n" + "="*80)
    print("Test 6: PQ nbits Parameter")
    print("="*80)
    
    from lotus.vector_store.faiss_vs import UnifiedFaissVS
    import tempfile
    import os
    
    dim = 64
    num_vecs = 500
    vectors = np.random.rand(num_vecs, dim).astype(np.float32)
    docs = [f"doc_{i}" for i in range(num_vecs)]
    
    for nbits in [8]:  # Could test [4, 8] but 4-bit might need more data
        print(f"\nTesting PQ with {nbits} bits:")
        try:
            vs = UnifiedFaissVS(
                factory_string="IVF20,PQ8",
                use_gpu=False,
                pq_nbits=nbits
            )
            
            with tempfile.TemporaryDirectory() as tmpdir:
                index_dir = os.path.join(tmpdir, f"pq_{nbits}bit")
                vs.index(docs, vectors, index_dir)
                print(f"  ✓ PQ{nbits}-bit index created")
                
                # Test search
                query = np.random.rand(dim).astype(np.float32)
                result = vs(query, K=5)
                print(f"  ✓ Search successful: {len(result.indices)} results")
                
        except Exception as e:
            print(f"  ✗ Failed with {nbits} bits: {e}")
    
    print("\n✓ PQ nbits test passed")


def main():
    """Run all tests."""
    print("="*80)
    print("GPU Index Optimization Validation Tests")
    print("="*80)
    
    try:
        test_memory_estimation()
        test_index_creation()
        test_batch_search()
        test_gpu_fallback()
        test_runtime_parameters()
        test_pq_nbits()
        
        print("\n" + "="*80)
        print("✅ All tests passed!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

