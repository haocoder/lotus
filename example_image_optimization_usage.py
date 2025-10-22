#!/usr/bin/env python3
"""
图片优化在sem_filter中的使用示例

展示如何在实际场景中使用新的图片优化功能
"""

import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import tempfile
from PIL import Image
import numpy as np

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_filter import sem_filter
from lotus.utils.image_optimizer import ImageOptimizer


def create_sample_dataframe_with_images() -> pd.DataFrame:
    """
    创建包含不同类型图片的示例DataFrame
    """
    print("Creating sample DataFrame with mixed image types...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 创建一些测试图片
    images = []
    for i in range(5):
        # 创建不同尺寸的图片
        size = 200 + i * 100
        color = ['red', 'blue', 'green', 'yellow', 'purple'][i]
        img = Image.new('RGB', (size, size), color=color)
        
        # 保存为文件
        img_path = f"{temp_dir}/image_{i}.jpg"
        img.save(img_path, 'JPEG', quality=90)
        images.append(img_path)
    
    # 创建DataFrame
    data = {
        'text': [
            'This is a red image showing a sunset',
            'This is a blue image of the ocean', 
            'This is a green image of a forest',
            'This is a yellow image of a sunflower',
            'This is a purple image of a flower'
        ],
        'image_url': [
            'https://example.com/sunset.jpg',
            'https://example.com/ocean.jpg', 
            'https://example.com/forest.jpg',
            'https://example.com/sunflower.jpg',
            'https://example.com/flower.jpg'
        ],
        'image_file': images,
        'category': ['nature', 'nature', 'nature', 'nature', 'nature']
    }
    
    df = pd.DataFrame(data)
    print(f"Created DataFrame with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Image files saved to: {temp_dir}")
    
    return df, temp_dir


def demonstrate_image_optimization_before_filter():
    """
    演示在sem_filter之前进行图片优化
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Image Optimization Before sem_filter")
    print("="*60)
    
    # 创建示例数据
    df, temp_dir = create_sample_dataframe_with_images()
    
    # 创建图片优化器
    optimizer = ImageOptimizer(
        max_size=(512, 512),
        quality=80,
        format="JPEG",
        enable_cache=True
    )
    
    print("\n1. Analyzing image types in DataFrame...")
    
    # 分析图片类型
    for idx, row in df.iterrows():
        print(f"Row {idx}:")
        print(f"  Text: {row['text'][:50]}...")
        print(f"  URL image: {row['image_url']}")
        print(f"  File image: {row['image_file']}")
        
        # 检查URL图片
        is_url = optimizer.is_url_image(row['image_url'])
        print(f"  URL detected: {is_url}")
        
        # 检查文件图片
        is_file = optimizer.is_file_path(row['image_file'])
        print(f"  File detected: {is_file}")
        print()
    
    print("\n2. Optimizing images...")
    
    # 优化图片
    optimized_images = []
    for idx, row in df.iterrows():
        print(f"Optimizing row {idx}...")
        
        # URL图片直接使用
        if optimizer.is_url_image(row['image_url']):
            optimized_url = row['image_url']
            print(f"  URL image kept as-is: {optimized_url}")
        else:
            optimized_url = optimizer.optimize_image(row['image_url'])
            print(f"  URL image optimized: {len(optimized_url)} chars")
        
        # 文件图片进行优化
        if optimizer.is_file_path(row['image_file']):
            optimized_file = optimizer.optimize_image(row['image_file'])
            print(f"  File image optimized: {len(optimized_file)} chars")
        else:
            optimized_file = row['image_file']
            print(f"  File image kept as-is: {optimized_file}")
        
        optimized_images.append({
            'text': row['text'],
            'image_url': optimized_url,
            'image_file': optimized_file,
            'category': row['category']
        })
    
    print("\n3. Cache statistics...")
    cache_stats = optimizer.get_cache_stats()
    print(f"Cache enabled: {cache_stats['enabled']}")
    if cache_stats['enabled']:
        print(f"Cache size: {cache_stats['cache_size']}")
        print(f"Total cached bytes: {cache_stats['total_bytes']:,}")
    
    return optimized_images, temp_dir


def demonstrate_sem_filter_with_optimized_images():
    """
    演示使用优化后的图片进行sem_filter
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: sem_filter with Optimized Images")
    print("="*60)
    
    # 配置模型（使用模拟模型避免实际API调用）
    print("Note: This demonstration shows the integration pattern.")
    print("In practice, you would use a real LM model.")
    
    # 创建优化后的数据
    optimized_images, temp_dir = demonstrate_image_optimization_before_filter()
    
    print("\n4. Preparing data for sem_filter...")
    
    # 转换为sem_filter需要的格式
    docs = []
    for item in optimized_images:
        doc = {
            'text': item['text'],
            'image': {
                'url_image': item['image_url'],
                'file_image': item['image_file']
            }
        }
        docs.append(doc)
    
    print(f"Prepared {len(docs)} documents for sem_filter")
    
    # 显示文档结构
    for i, doc in enumerate(docs):
        print(f"Document {i}:")
        print(f"  Text: {doc['text'][:50]}...")
        print(f"  URL image: {doc['image']['url_image'][:50]}...")
        print(f"  File image: {doc['image']['file_image'][:50]}...")
        print()
    
    print("5. sem_filter integration pattern:")
    print("""
    # 实际使用时的代码模式：
    
    # 1. 创建优化器
    optimizer = ImageOptimizer(max_size=(512, 512), quality=80)
    
    # 2. 预处理图片
    optimized_docs = []
    for doc in docs:
        if 'image' in doc:
            # 优化图片
            optimized_image = optimizer.optimize_image(doc['image'])
            doc['image'] = optimized_image
        optimized_docs.append(doc)
    
    # 3. 使用sem_filter
    result = sem_filter(
        docs=optimized_docs,
        model=model,
        user_instruction="Does this image contain natural scenes?",
        use_batch_processing=True,
        batch_size=5
    )
    """)
    
    return temp_dir


def demonstrate_performance_benefits():
    """
    演示性能优势
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Performance Benefits")
    print("="*60)
    
    # 创建大图片进行测试
    print("Creating large test images...")
    
    # 创建2048x2048的大图片
    large_image = Image.new('RGB', (2048, 2048), color='red')
    
    # 测试原始方法
    print("\nTesting original method...")
    import time
    from lotus.utils import fetch_image
    
    start_time = time.time()
    original_result = fetch_image(large_image, "base64")
    original_time = time.time() - start_time
    original_size = len(original_result)
    
    # 测试优化方法
    print("Testing optimized method...")
    optimizer = ImageOptimizer(max_size=(512, 512), quality=80)
    
    start_time = time.time()
    optimized_result = optimizer.optimize_image(large_image)
    optimized_time = time.time() - start_time
    optimized_size = len(optimized_result)
    
    # 显示结果
    print(f"\nPerformance Comparison:")
    print(f"Original method:")
    print(f"  Time: {original_time:.3f}s")
    print(f"  Size: {original_size:,} bytes")
    print(f"Optimized method:")
    print(f"  Time: {optimized_time:.3f}s")
    print(f"  Size: {optimized_size:,} bytes")
    print(f"\nImprovements:")
    print(f"  Speed: {original_time/optimized_time:.2f}x faster")
    print(f"  Size reduction: {(original_size-optimized_size)/original_size:.1%}")
    print(f"  Memory savings: {(original_size-optimized_size)/1024/1024:.1f} MB")


def main():
    """
    运行所有演示
    """
    print("🚀 Image Optimization Usage Examples")
    print("="*80)
    
    try:
        # 演示图片优化
        temp_dir1 = demonstrate_image_optimization_before_filter()
        
        # 演示sem_filter集成
        temp_dir2 = demonstrate_sem_filter_with_optimized_images()
        
        # 演示性能优势
        demonstrate_performance_benefits()
        
        print("\n" + "="*80)
        print("✅ All demonstrations completed successfully!")
        print("\nKey Benefits:")
        print("1. URL images are passed directly to the model (no processing)")
        print("2. Local images are intelligently compressed")
        print("3. Significant memory and bandwidth savings")
        print("4. Maintains full compatibility with existing sem_filter workflow")
        print("5. Caching reduces repeated processing overhead")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 清理临时文件
        import shutil
        try:
            if 'temp_dir1' in locals():
                shutil.rmtree(temp_dir1)
            if 'temp_dir2' in locals():
                shutil.rmtree(temp_dir2)
        except:
            pass
    
    return 0


if __name__ == "__main__":
    exit(main())
