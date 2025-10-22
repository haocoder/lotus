"""
GPU acceleration example for Lotus semantic operations.

This example demonstrates how to use GPU acceleration with various Lotus operators
including sem_index, sem_search, sem_cluster_by, and sem_sim_join.
"""

import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissGPUVS
from lotus.config import configure_gpu, get_gpu_monitor
from lotus.utils import gpu_cluster

# Configure Lotus with GPU support
def setup_lotus_with_gpu():
    """Setup Lotus with GPU-accelerated components."""
    
    # Configure GPU settings
    configure_gpu(
        prefer_gpu=True,
        fallback_to_cpu=True,
        gpu_device_ids=[0],  # Use first GPU
        gpu_memory_fraction=0.8,
        enable_monitoring=True,
        metrics_file="gpu_performance_metrics.json"
    )
    
    # Setup models
    lm = LM(model="gpt-4o-mini")
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    
    # Use GPU-accelerated vector store
    vs = FaissGPUVS(
        factory_string="IVF1024,Flat",  # Good for GPU
        metric="METRIC_INNER_PRODUCT"
    )
    
    # Configure Lotus
    lotus.settings.configure(lm=lm, rm=rm, vs=vs)
    
    print("✅ Lotus configured with GPU acceleration")
    return lm, rm, vs


def create_sample_data():
    """Create sample data for demonstration."""
    df = pd.DataFrame({
        'title': [
            'Machine learning tutorial for beginners',
            'Advanced deep learning techniques',
            'Data science with Python',
            'Natural language processing guide',
            'Computer vision applications',
            'Reinforcement learning basics',
            'Statistical analysis methods',
            'Big data processing tools',
            'AI ethics and fairness',
            'Quantum computing introduction',
            'Cooking healthy Mediterranean food',
            'Traditional Italian recipes',
            'Baking bread at home',
            'Vegetarian meal planning',
            'Wine and food pairing',
            'Japanese cuisine basics',
            'Outdoor hiking adventures',
            'Mountain climbing safety',
            'Camping equipment guide',
            'Photography for travelers'
        ],
        'category': [
            'AI/ML', 'AI/ML', 'AI/ML', 'AI/ML', 'AI/ML', 'AI/ML',
            'Data Science', 'Data Science', 'AI/ML', 'Technology',
            'Cooking', 'Cooking', 'Cooking', 'Cooking', 'Cooking', 'Cooking',
            'Outdoor', 'Outdoor', 'Outdoor', 'Outdoor'
        ]
    })
    
    return df


def demo_gpu_indexing(df):
    """Demonstrate GPU-accelerated indexing."""
    print("\n🚀 GPU Indexing Demo")
    print("=" * 50)
    
    # Create index with GPU acceleration
    df_indexed = df.sem_index('title', 'gpu_title_index', use_gpu=True)
    
    print(f"✅ Indexed {len(df)} documents with GPU acceleration")
    return df_indexed


def demo_gpu_search(df):
    """Demonstrate GPU-accelerated semantic search."""
    print("\n🔍 GPU Semantic Search Demo")
    print("=" * 50)
    
    # Search with GPU acceleration
    search_results = df.sem_search(
        'title', 
        'artificial intelligence and machine learning',
        K=5,
        use_gpu=True,
        return_scores=True
    )
    
    print("Search Results:")
    for idx, row in search_results.iterrows():
        print(f"  📄 {row['title']}")
        if 'vec_scores_sim_score' in row:
            print(f"     Score: {row['vec_scores_sim_score']:.3f}")
    
    return search_results


def demo_gpu_clustering(df):
    """Demonstrate GPU-accelerated clustering."""
    print("\n🎯 GPU Clustering Demo")
    print("=" * 50)
    
    # Cluster with GPU acceleration
    clustered_df = df.sem_cluster_by(
        'title', 
        ncentroids=3,
        prefer_gpu=True,
        verbose=True,
        niter=25
    )
    
    print("Clustering Results:")
    for cluster_id in sorted(clustered_df['cluster_id'].unique()):
        cluster_docs = clustered_df[clustered_df['cluster_id'] == cluster_id]
        print(f"\n📊 Cluster {cluster_id} ({len(cluster_docs)} documents):")
        for _, row in cluster_docs.head(3).iterrows():
            print(f"  • {row['title']}")
    
    return clustered_df


def demo_gpu_similarity_join(df):
    """Demonstrate GPU-accelerated similarity join."""
    print("\n🔗 GPU Similarity Join Demo")
    print("=" * 50)
    
    # Create category dataframe
    categories_df = pd.DataFrame({
        'category_desc': [
            'Machine Learning and Artificial Intelligence',
            'Data Analysis and Statistics', 
            'Culinary Arts and Cooking',
            'Outdoor Activities and Adventure'
        ]
    })
    
    # Index categories
    categories_df = categories_df.sem_index('category_desc', 'category_index', use_gpu=True)
    
    # Perform similarity join with GPU
    joined_df = df.sem_sim_join(
        categories_df,
        left_on='title',
        right_on='category_desc',
        K=1,
        use_gpu=True,
        score_suffix='_match'
    )
    
    print("Similarity Join Results (top 5):")
    for _, row in joined_df.head(5).iterrows():
        print(f"  📄 {row['title']}")
        print(f"     → Matched with: {row['category_desc']}")
        print(f"     → Score: {row['_scores_match']:.3f}")
        print()
    
    return joined_df


def demo_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("\n📊 GPU Performance Monitoring")
    print("=" * 50)
    
    monitor = get_gpu_monitor()
    if monitor:
        # Get performance summary
        summary = monitor.get_metrics_summary()
        
        if summary:
            print(f"Total Operations: {summary.get('total_operations', 0)}")
            print(f"Monitoring Duration: {summary.get('monitoring_duration', 0):.2f}s")
            
            print("\nOperation Performance:")
            for op_name, stats in summary.get('operations', {}).items():
                print(f"  🔧 {op_name}:")
                print(f"     Count: {stats['count']}")
                print(f"     Avg Duration: {stats['avg_duration']:.3f}s")
                print(f"     Success Rate: {stats['success_rate']:.1%}")
                if stats.get('avg_throughput', 0) > 0:
                    print(f"     Avg Throughput: {stats['avg_throughput']:.1f} items/s")
        
        # Save detailed metrics
        monitor.save_metrics("detailed_gpu_metrics.json")
        print("\n💾 Detailed metrics saved to 'detailed_gpu_metrics.json'")
    else:
        print("Performance monitoring not available")


def main():
    """Main function to run all GPU acceleration demos."""
    print("🚀 Lotus GPU Acceleration Demo")
    print("=" * 60)
    
    # Setup
    try:
        setup_lotus_with_gpu()
    except Exception as e:
        print(f"❌ GPU setup failed: {e}")
        print("This demo requires GPU support. Please ensure:")
        print("1. CUDA is installed")
        print("2. faiss-gpu is installed: pip install faiss-gpu")
        print("3. GPU drivers are properly configured")
        return
    
    # Create sample data
    df = create_sample_data()
    print(f"📊 Created sample dataset with {len(df)} documents")
    
    try:
        # Run demos
        df_indexed = demo_gpu_indexing(df)
        demo_gpu_search(df_indexed)
        demo_gpu_clustering(df_indexed)
        demo_gpu_similarity_join(df_indexed)
        
        # Show performance monitoring
        demo_performance_monitoring()
        
        print("\n🎉 GPU acceleration demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Falling back to CPU operations...")
        
        # Fallback demos without GPU
        try:
            df_indexed = df.sem_index('title', 'cpu_title_index')
            cpu_results = df_indexed.sem_search('title', 'machine learning', K=3)
            print(f"✅ CPU fallback successful - found {len(cpu_results)} results")
        except Exception as cpu_error:
            print(f"❌ CPU fallback also failed: {cpu_error}")


if __name__ == "__main__":
    main()
