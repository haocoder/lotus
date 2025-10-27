"""
Complete example demonstrating GPU-accelerated vector indexing and search.

This example showcases:
1. CPU/GPU dual support for index creation
2. Various index types (Flat, IVF-Flat, IVF-PQ, IVF-SQ, HNSW)
3. cuVS acceleration with RMM memory management
4. Memory estimation and optimization
5. Batch search operations
6. Runtime parameter tuning for recall/speed tradeoff
"""

import pandas as pd
import lotus
from lotus.models import SentenceTransformersRM
from lotus.vector_store import UnifiedFaissVS


def example_flat_index_gpu():
    """Example 1: Flat index on GPU (exact search, best for < 10K vectors)."""
    print("\n" + "="*80)
    print("Example 1: Flat Index on GPU (Exact Search)")
    print("="*80)
    
    # Configure with GPU-accelerated components
    rm = SentenceTransformersRM(
        model="intfloat/e5-base-v2",
        device="cuda"  # Use GPU for embedding generation
    )
    vs = UnifiedFaissVS(
        factory_string="Flat",
        use_gpu=True
    )
    lotus.settings.configure(rm=rm, vs=vs)
    
    # Create sample dataset
    df = pd.DataFrame({
        "text": [
            "Machine learning algorithms for classification",
            "Deep neural networks and transformers",
            "Natural language processing techniques",
            "Computer vision and image recognition",
            "Data science and statistical analysis",
        ] * 100  # 500 documents
    })
    
    print(f"Dataset: {len(df)} documents")
    
    # Create index with GPU acceleration
    print("\nCreating Flat index on GPU...")
    df.sem_index(
        "text",
        "flat_gpu_index",
        use_gpu=True,
        factory_string="Flat"
    )
    
    # Perform search
    print("\nSearching with GPU acceleration...")
    df.load_sem_index("text", "flat_gpu_index")
    results = df.sem_search(
        "text",
        "artificial intelligence and machine learning",
        K=5,
        use_gpu=True
    )
    
    print(f"\nTop 5 results:")
    print(results["text"].to_list())


def example_ivf_flat_gpu():
    """Example 2: IVF-Flat index on GPU (good balance for 10K-1M vectors)."""
    print("\n" + "="*80)
    print("Example 2: IVF-Flat Index on GPU (Balanced Performance)")
    print("="*80)
    
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2", device="cuda")
    vs = UnifiedFaissVS(factory_string="IVF256,Flat", use_gpu=True)
    lotus.settings.configure(rm=rm, vs=vs)
    
    # Larger dataset
    df = pd.DataFrame({
        "text": [
            f"Document about topic {i % 50}: content for {i}"
            for i in range(5000)
        ]
    })
    
    print(f"Dataset: {len(df)} documents")
    
    # Create IVF index (with automatic training)
    print("\nCreating IVF-Flat index on GPU (with training)...")
    df.sem_index(
        "text",
        "ivf_flat_gpu_index",
        use_gpu=True,
        factory_string="IVF256,Flat"
    )
    
    # Search with runtime parameter tuning
    print("\nSearching with nprobe tuning...")
    df.load_sem_index("text", "ivf_flat_gpu_index")
    
    # Lower nprobe = faster but lower recall
    results_fast = df.sem_search(
        "text",
        "topic 25",
        K=10,
        use_gpu=True,
        ivf_nprobe=1  # Fast: only search 1 cluster
    )
    print(f"Fast search (nprobe=1): {len(results_fast)} results")
    
    # Higher nprobe = slower but better recall
    results_accurate = df.sem_search(
        "text",
        "topic 25",
        K=10,
        use_gpu=True,
        ivf_nprobe=32  # Accurate: search 32 clusters
    )
    print(f"Accurate search (nprobe=32): {len(results_accurate)} results")


def example_ivf_pq_gpu():
    """Example 3: IVF-PQ index on GPU (memory-efficient for > 1M vectors)."""
    print("\n" + "="*80)
    print("Example 3: IVF-PQ Index on GPU (High Compression)")
    print("="*80)
    
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2", device="cuda")
    
    # IVF-PQ: High compression ratio
    # nlist=1024: number of clusters (higher = better accuracy, more training time)
    # PQ32: 32 subquantizers (higher = better accuracy, more memory)
    # pq_nbits=8: 8 bits per code (can be 4, 6, 8, 10, 12, 16)
    vs = UnifiedFaissVS(
        factory_string="IVF1024,PQ32",
        use_gpu=True,
        pq_nbits=8  # 8-bit product quantization
    )
    lotus.settings.configure(rm=rm, vs=vs)
    
    # Simulate large dataset
    df = pd.DataFrame({
        "text": [
            f"Research paper on {i % 100} with methodology {i}"
            for i in range(10000)
        ]
    })
    
    print(f"Dataset: {len(df)} documents")
    print("Memory efficiency: ~32 bytes per vector (vs ~512 bytes for Flat)")
    
    print("\nCreating IVF-PQ index on GPU...")
    df.sem_index(
        "text",
        "ivf_pq_gpu_index",
        use_gpu=True,
        factory_string="IVF1024,PQ32",
        pq_nbits=8
    )
    
    print("\nSearching compressed index...")
    df.load_sem_index("text", "ivf_pq_gpu_index")
    results = df.sem_search(
        "text",
        "methodology research",
        K=10,
        use_gpu=True,
        ivf_nprobe=16
    )
    print(f"Found {len(results)} results from compressed index")


def example_ivf_sq_gpu():
    """Example 4: IVF-SQ8 index on GPU (4x memory reduction)."""
    print("\n" + "="*80)
    print("Example 4: IVF-SQ8 Index on GPU (Scalar Quantization)")
    print("="*80)
    
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2", device="cuda")
    
    # IVF-SQ8: Scalar quantization (4x compression, better than PQ for some use cases)
    vs = UnifiedFaissVS(factory_string="IVF512,SQ8", use_gpu=True)
    lotus.settings.configure(rm=rm, vs=vs)
    
    df = pd.DataFrame({
        "text": [f"Article number {i} about various topics" for i in range(3000)]
    })
    
    print(f"Dataset: {len(df)} documents")
    print("Memory: ~128 bytes per vector (4x compression)")
    
    print("\nCreating IVF-SQ8 index on GPU...")
    df.sem_index(
        "text",
        "ivf_sq_gpu_index",
        use_gpu=True,
        factory_string="IVF512,SQ8"
    )
    
    df.load_sem_index("text", "ivf_sq_gpu_index")
    results = df.sem_search("text", "articles about topics", K=10, use_gpu=True)
    print(f"Found {len(results)} results")


def example_batch_search_gpu():
    """Example 5: Batch search on GPU (10-100x faster than sequential)."""
    print("\n" + "="*80)
    print("Example 5: Batch Search on GPU (High Throughput)")
    print("="*80)
    
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2", device="cuda")
    vs = UnifiedFaissVS(factory_string="IVF256,Flat", use_gpu=True)
    lotus.settings.configure(rm=rm, vs=vs)
    
    df = pd.DataFrame({
        "text": [f"Document {i} with content" for i in range(2000)]
    })
    
    print(f"Dataset: {len(df)} documents")
    
    df.sem_index("text", "batch_gpu_index", use_gpu=True, factory_string="IVF256,Flat")
    df.load_sem_index("text", "batch_gpu_index")
    
    # Batch search: multiple queries at once
    queries = [
        "document content",
        "information retrieval",
        "semantic search",
        "vector database",
        "machine learning"
    ]
    
    print(f"\nPerforming batch search with {len(queries)} queries...")
    results = df.sem_search(
        "text",
        queries,  # Pass list of queries for batch processing
        K=5,
        use_gpu=True
    )
    
    print(f"Results: {len(results)} DataFrames (one per query)")
    for i, result_df in enumerate(results):
        print(f"  Query {i+1}: {len(result_df)} results")


def example_hnsw_cpu():
    """Example 6: HNSW index on CPU (best for low-latency queries)."""
    print("\n" + "="*80)
    print("Example 6: HNSW Index on CPU (Low Latency)")
    print("="*80)
    
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    
    # HNSW: Graph-based index (best on CPU, GPU support limited)
    # M=64: number of connections per node (higher = better accuracy, more memory)
    vs = UnifiedFaissVS(
        factory_string="HNSW64",
        use_gpu=False  # HNSW works best on CPU
    )
    lotus.settings.configure(rm=rm, vs=vs)
    
    df = pd.DataFrame({
        "text": [f"Entry {i} for HNSW test" for i in range(1000)]
    })
    
    print(f"Dataset: {len(df)} documents")
    print("Note: HNSW is optimized for CPU, not GPU")
    
    print("\nCreating HNSW index on CPU...")
    df.sem_index(
        "text",
        "hnsw_cpu_index",
        use_gpu=False,
        factory_string="HNSW64"
    )
    
    print("\nSearching with efSearch tuning...")
    df.load_sem_index("text", "hnsw_cpu_index")
    
    # Lower efSearch = faster but lower recall
    results_fast = df.sem_search(
        "text",
        "HNSW test",
        K=10,
        use_gpu=False,
        hnsw_ef_search=16  # Fast
    )
    print(f"Fast search (efSearch=16): {len(results_fast)} results")
    
    # Higher efSearch = slower but better recall
    results_accurate = df.sem_search(
        "text",
        "HNSW test",
        K=10,
        use_gpu=False,
        hnsw_ef_search=128  # Accurate
    )
    print(f"Accurate search (efSearch=128): {len(results_accurate)} results")


def example_memory_estimation():
    """Example 7: Memory estimation for different index types."""
    print("\n" + "="*80)
    print("Example 7: Index Memory Estimation")
    print("="*80)
    
    from lotus.vector_store.faiss_vs import estimate_index_memory
    
    num_vecs = 100000  # 100K vectors
    dim = 768          # Typical embedding dimension
    
    print(f"\nEstimating memory for {num_vecs:,} vectors of {dim} dimensions:")
    print("-" * 80)
    
    # Flat index
    mem_flat = estimate_index_memory("Flat", num_vecs, dim, is_gpu=True)
    print(f"Flat (GPU):        {mem_flat / (1024**2):.1f} MB")
    
    # IVF-Flat
    mem_ivf_flat = estimate_index_memory("IVF1024,Flat", num_vecs, dim, is_gpu=True)
    print(f"IVF-Flat (GPU):    {mem_ivf_flat / (1024**2):.1f} MB")
    
    # IVF-SQ8 (4x compression)
    mem_ivf_sq = estimate_index_memory("IVF1024,SQ8", num_vecs, dim, is_gpu=True)
    print(f"IVF-SQ8 (GPU):     {mem_ivf_sq / (1024**2):.1f} MB (4x compression)")
    
    # IVF-PQ (high compression)
    mem_ivf_pq = estimate_index_memory("IVF1024,PQ32", num_vecs, dim, is_gpu=True)
    print(f"IVF-PQ32 (GPU):    {mem_ivf_pq / (1024**2):.1f} MB (24x compression)")
    
    # HNSW (CPU)
    mem_hnsw = estimate_index_memory("HNSW64", num_vecs, dim, is_gpu=False)
    print(f"HNSW64 (CPU):      {mem_hnsw / (1024**2):.1f} MB (1.5x overhead)")
    
    print("\nRecommendations:")
    print("- Small datasets (< 10K): Use Flat for exact search")
    print("- Medium datasets (10K-1M): Use IVF-Flat for balance")
    print("- Large datasets (> 1M): Use IVF-PQ for memory efficiency")
    print("- Low latency needs: Use HNSW on CPU")
    print("- Memory constrained: Use IVF-SQ8 or IVF-PQ")


def main():
    """Run all examples."""
    print("GPU-Accelerated Vector Indexing Examples")
    print("=" * 80)
    print("This demo showcases various FAISS index types with GPU acceleration.")
    print("=" * 80)
    
    try:
        # Run examples
        example_flat_index_gpu()
        example_ivf_flat_gpu()
        example_ivf_pq_gpu()
        example_ivf_sq_gpu()
        example_batch_search_gpu()
        example_hnsw_cpu()
        example_memory_estimation()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Some examples require GPU. If GPU is not available,")
        print("the system will automatically fall back to CPU.")


if __name__ == "__main__":
    main()

