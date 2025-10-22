"""
简化版图片优化模块 - 不依赖PIL和numpy

专门针对sem_filter等算子的图片处理优化，支持：
1. URL图片直接传递（无需处理）
2. 本地图片路径的智能识别
3. 保持与现有流程的完全兼容性
"""

import base64
import hashlib
import io
import os
import time
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path


class SimpleImageOptimizer:
    """
    简化版图片优化器 - 不依赖PIL和numpy
    
    设计原则：
    1. 不改变现有算子核心流程
    2. 针对不同图片类型采用不同策略
    3. 提供可配置的优化参数
    4. 支持缓存机制减少重复处理
    """
    
    def __init__(
        self,
        enable_cache: bool = True,
        cache_size: int = 1000
    ):
        """
        初始化简化版图片优化器
        
        Args:
            enable_cache: 是否启用缓存
            cache_size: 缓存大小限制
        """
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # 缓存相关
        self._cache: Dict[str, str] = {}
        self._cache_access_times: Dict[str, float] = {}
        self._cache_sizes: Dict[str, int] = {}
    
    def is_url_image(self, image_input: Any) -> bool:
        """
        判断是否为URL图片
        
        Args:
            image_input: 图片输入
            
        Returns:
            bool: 是否为URL图片
        """
        if isinstance(image_input, str):
            return (
                image_input.startswith("http://") or 
                image_input.startswith("https://") or
                image_input.startswith("s3://")
            )
        return False
    
    def is_encoded_image(self, image_input: Any) -> bool:
        """
        判断是否为已编码的图片（base64等）
        
        Args:
            image_input: 图片输入
            
        Returns:
            bool: 是否为已编码图片
        """
        if isinstance(image_input, str):
            return image_input.startswith("data:image")
        return False
    
    def is_file_path(self, image_input: Any) -> bool:
        """
        判断是否为文件路径
        
        Args:
            image_input: 图片输入
            
        Returns:
            bool: 是否为文件路径
        """
        if isinstance(image_input, str):
            return (
                image_input.startswith("file://") or
                (not self.is_url_image(image_input) and 
                 not self.is_encoded_image(image_input) and
                 os.path.exists(image_input))
            )
        return False
    
    def _get_image_hash(self, image_input: Any) -> str:
        """
        生成图片的哈希值用于缓存
        
        Args:
            image_input: 图片输入
            
        Returns:
            str: 图片哈希值
        """
        if isinstance(image_input, str):
            return hashlib.md5(image_input.encode()).hexdigest()
        else:
            return hashlib.md5(str(image_input).encode()).hexdigest()
    
    def _evict_cache(self) -> None:
        """
        清理缓存，移除最少使用的项目
        """
        if not self._cache:
            return
        
        # 按访问时间排序，移除最旧的
        sorted_items = sorted(
            self._cache_access_times.items(),
            key=lambda x: x[1]
        )
        
        # 移除最旧的25%
        remove_count = max(1, len(sorted_items) // 4)
        for key, _ in sorted_items[:remove_count]:
            if key in self._cache:
                del self._cache[key]
            if key in self._cache_access_times:
                del self._cache_access_times[key]
            if key in self._cache_sizes:
                del self._cache_sizes[key]
    
    def optimize_image(self, image_input: Any) -> str:
        """
        智能优化图片
        
        策略：
        1. URL图片：直接返回，让模型自己下载
        2. 已编码图片：直接返回（假设已经优化过）
        3. 文件路径：返回文件路径（让模型自己处理）
        
        Args:
            image_input: 图片输入（URL、文件路径、base64等）
            
        Returns:
            str: 优化后的图片（URL、文件路径或base64编码）
        """
        # 1. URL图片直接返回
        if self.is_url_image(image_input):
            return image_input
        
        # 2. 检查缓存
        if self.enable_cache:
            image_hash = self._get_image_hash(image_input)
            if image_hash in self._cache:
                self._cache_access_times[image_hash] = time.time()
                return self._cache[image_hash]
        
        # 3. 处理不同类型的图片
        if self.is_encoded_image(image_input):
            # 已编码图片：直接返回
            optimized = image_input
        elif self.is_file_path(image_input):
            # 文件路径：直接返回
            optimized = image_input
        else:
            # 其他类型：直接返回
            optimized = image_input
        
        # 4. 更新缓存
        if self.enable_cache:
            if len(self._cache) >= self.cache_size:
                self._evict_cache()
            
            image_hash = self._get_image_hash(image_input)
            self._cache[image_hash] = optimized
            self._cache_access_times[image_hash] = time.time()
            self._cache_sizes[image_hash] = len(str(optimized))
        
        return optimized
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        if not self.enable_cache:
            return {"enabled": False}
        
        total_size = sum(self._cache_sizes.values())
        return {
            "enabled": True,
            "cache_size": len(self._cache),
            "max_size": self.cache_size,
            "total_bytes": total_size,
            "hit_rate": "N/A"  # 可以添加命中率统计
        }
    
    def clear_cache(self) -> None:
        """
        清空缓存
        """
        self._cache.clear()
        self._cache_access_times.clear()
        self._cache_sizes.clear()


# 全局优化器实例
_default_optimizer = SimpleImageOptimizer()


def optimize_image_for_processing(
    image_input: Any,
    **kwargs
) -> str:
    """
    便捷函数：优化图片用于处理
    
    这是对现有fetch_image函数的增强版本，提供智能的图片优化
    
    Args:
        image_input: 图片输入（URL、文件路径、PIL Image、base64等）
        **kwargs: 其他参数（为了兼容性）
        
    Returns:
        str: 优化后的图片（URL或base64编码）
    """
    # 创建临时优化器实例
    optimizer = SimpleImageOptimizer(
        enable_cache=True
    )
    
    return optimizer.optimize_image(image_input)


def is_url_image(image_input: Any) -> bool:
    """
    便捷函数：判断是否为URL图片
    
    Args:
        image_input: 图片输入
        
    Returns:
        bool: 是否为URL图片
    """
    return _default_optimizer.is_url_image(image_input)
