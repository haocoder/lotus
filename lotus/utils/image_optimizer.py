"""
图片优化模块 - 针对sem_filter等算子的图片处理优化

该模块提供智能的图片压缩和优化功能，支持：
1. URL图片直接传递（无需处理）
2. 本地图片/文件路径的智能压缩
3. 已编码图片的优化处理
4. 保持与现有流程的完全兼容性
"""

import base64
import hashlib
import io
import os
import time
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path

from PIL import Image


import numpy as np



class ImageOptimizer:
    """
    图片优化器 - 提供智能的图片压缩和优化功能
    
    设计原则：
    1. 不改变现有算子核心流程
    2. 针对不同图片类型采用不同策略
    3. 提供可配置的压缩参数
    4. 支持缓存机制减少重复处理
    
    压缩策略：
    - _compress_image_simple: 简单快速压缩，适合对质量要求不高的场景
    - _compress_image_advanced: 智能渐进式压缩，适合对压缩率要求高的场景
    """
    
    def __init__(
        self,
        max_size: Tuple[int, int] = (1024, 1024),
        quality: int = 85,
        format: str = "JPEG",
        enable_cache: bool = True,
        cache_size: int = 1000,
        use_advanced_compression: bool = True
    ):
        """
        初始化图片优化器
        
        Args:
            max_size: 最大图片尺寸 (width, height)
            quality: JPEG质量 (1-100)
            format: 输出格式 ("JPEG", "PNG", "WEBP")
            enable_cache: 是否启用缓存
            cache_size: 缓存大小限制
            use_advanced_compression: 是否使用高级压缩策略
        """
        self.max_size = max_size
        self.quality = quality
        self.format = format.upper()
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.use_advanced_compression = use_advanced_compression
        
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
        elif isinstance(image_input, (Image.Image, np.ndarray)):
            # 对于PIL Image或numpy数组，使用其数据生成哈希
            if isinstance(image_input, Image.Image):
                data = image_input.tobytes()
            else:
                data = image_input.tobytes()
            return hashlib.md5(data).hexdigest()
        else:
            return hashlib.md5(str(image_input).encode()).hexdigest()
    
    def _compress_image_simple(self, image: Any) -> str:
        """
        简单压缩图片并返回base64编码
        
        使用单次压缩策略，适合快速处理
        
        Args:
            image: PIL Image对象
            
        Returns:
            str: 压缩后的base64编码图片
        """
        if Image is None:
            raise ValueError("PIL Image module not available")
        
        # 1. 尺寸优化
        if image.size[0] > self.max_size[0] or image.size[1] > self.max_size[1]:
            # 保持宽高比缩放
            image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
        
        # 2. 格式优化
        buffered = io.BytesIO()
        
        # 根据原图格式选择最佳输出格式
        if self.format == "JPEG":
            # JPEG不支持透明度，需要转换为RGB
            if image.mode in ("RGBA", "LA", "P"):
                # 创建白色背景
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "P":
                    image = image.convert("RGBA")
                background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            
            image.save(buffered, format="JPEG", quality=self.quality, optimize=True)
            mime_type = "jpeg"
        elif self.format == "PNG":
            image.save(buffered, format="PNG", optimize=True)
            mime_type = "png"
        elif self.format == "WEBP":
            image.save(buffered, format="WEBP", quality=self.quality, optimize=True)
            mime_type = "webp"
        else:
            # 默认使用JPEG
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=self.quality, optimize=True)
            mime_type = "jpeg"
        
        # 3. 返回base64编码
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{mime_type};base64,{encoded}"
    
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
        2. 已编码图片：检查是否需要重新压缩
        3. 文件路径/PIL Image：进行压缩处理
        
        Args:
            image_input: 图片输入（URL、文件路径、PIL Image、base64等）
            
        Returns:
            str: 优化后的图片（URL或base64编码）
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
            # 已编码图片：检查是否需要重新压缩
            optimized = self._optimize_encoded_image(image_input)
        elif self.is_file_path(image_input):
            # 文件路径：加载并压缩
            optimized = self._optimize_file_image(image_input)
        elif isinstance(image_input, (Image.Image, np.ndarray)):
            # PIL Image或numpy数组：直接压缩
            optimized = self._optimize_pil_image(image_input)
        else:
            # 其他类型：尝试转换为PIL Image
            optimized = self._optimize_unknown_image(image_input)
        
        # 4. 更新缓存
        if self.enable_cache:
            if len(self._cache) >= self.cache_size:
                self._evict_cache()
            
            image_hash = self._get_image_hash(image_input)
            self._cache[image_hash] = optimized
            self._cache_access_times[image_hash] = time.time()
            self._cache_sizes[image_hash] = len(optimized)
        
        return optimized
    
    def _optimize_encoded_image(self, encoded_image: str) -> str:
        """
        优化已编码的图片
        
        Args:
            encoded_image: base64编码的图片
            
        Returns:
            str: 优化后的base64编码图片
        """
        try:
            # 解析base64数据
            if "base64," in encoded_image:
                header, data = encoded_image.split("base64,", 1)
                image_data = base64.b64decode(data)
            else:
                image_data = base64.b64decode(encoded_image)
            
            # 加载图片
            image = Image.open(io.BytesIO(image_data))
            
            # 检查是否需要压缩
            current_size = len(image_data)
            if current_size < 100 * 1024:  # 小于100KB，可能不需要压缩
                return encoded_image
            
            # 压缩图片
            return self._choose_compression_method(image)
            
        except Exception as e:
            # 如果处理失败，返回原图
            print(f"Warning: Failed to optimize encoded image: {e}")
            return encoded_image
    
    def _compress_image_advanced(self, image: Image.Image) -> str:
        """
        智能压缩图片，减少base64编码大小
        
        使用渐进式多级压缩策略，自动选择最优压缩结果
        
        压缩策略：
        1. 尺寸压缩：按比例缩放到最大尺寸
        2. 质量压缩：调整JPEG质量参数
        3. 格式优化：选择最适合的格式
        4. 渐进式压缩：多级压缩直到达到目标大小
        
        Args:
            image: PIL Image对象
            
        Returns:
            str: 压缩后的base64编码图片
        """
        if Image is None:
            raise ValueError("PIL Image module not available")
        
        # 转换为RGB模式（JPEG需要）
        if image.mode in ('RGBA', 'LA', 'P'):
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 渐进式压缩策略
        compression_levels = [
            # (max_size, quality, format)
            (self.max_size, self.quality, self.format),  # 标准压缩
            ((self.max_size[0]//2, self.max_size[1]//2), max(60, self.quality-10), "JPEG"),  # 中等压缩
            ((self.max_size[0]//3, self.max_size[1]//3), max(40, self.quality-20), "JPEG"),  # 高压缩
            ((self.max_size[0]//4, self.max_size[1]//4), max(20, self.quality-30), "JPEG"),  # 极高压缩
        ]
        
        best_result = None
        best_size = float('inf')
        
        for max_size, quality, format in compression_levels:
            try:
                # 1. 尺寸压缩
                compressed_image = self._resize_image(image, max_size)
                
                # 2. 质量压缩
                result = self._encode_with_quality(compressed_image, quality, format)
                
                # 3. 检查压缩效果
                result_size = len(result)
                if result_size < best_size:
                    best_result = result
                    best_size = result_size
                
                # 如果已经足够小，停止压缩
                if result_size < 200 * 1024:  # 小于200KB
                    break
                    
            except Exception as e:
                print(f"Warning: Compression level failed: {e}")
                continue
        
        return best_result or self._encode_with_quality(image, self.quality, self.format)
    
    def _resize_image(self, image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
        """
        智能调整图片尺寸
        
        Args:
            image: 原始图片
            max_size: 最大尺寸 (width, height)
            
        Returns:
            Image.Image: 调整后的图片
        """
        if Image is None:
            raise ValueError("PIL Image module not available")
        
        # 计算缩放比例
        width, height = image.size
        max_width, max_height = max_size
        
        # 计算保持宽高比的最佳尺寸
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # 不放大图片
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 使用高质量重采样
            resample = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            return image.resize((new_width, new_height), resample=resample)
        
        return image
    
    def _choose_compression_method(self, image: Any) -> str:
        """
        根据配置选择压缩方法
        
        Args:
            image: 图片对象
            
        Returns:
            str: 压缩后的base64编码图片
        """
        if self.use_advanced_compression:
            return self._compress_image_advanced(image)
        else:
            return self._compress_image_simple(image)
    
    def _encode_with_quality(self, image: Image.Image, quality: int, format: str) -> str:
        """
        使用指定质量和格式编码图片
        
        Args:
            image: PIL Image对象
            quality: 压缩质量 (1-100)
            format: 输出格式
            
        Returns:
            str: base64编码的图片
        """
        if Image is None:
            raise ValueError("PIL Image module not available")
        
        # 准备输出缓冲区
        output = io.BytesIO()
        
        # 根据格式选择编码参数
        if format.upper() == "JPEG":
            # JPEG优化参数
            image.save(
                output, 
                format="JPEG", 
                quality=quality,
                optimize=True,  # 启用优化
                progressive=True,  # 渐进式JPEG
                subsampling=0 if quality > 80 else 1  # 高质量时使用4:4:4采样
            )
        elif format.upper() == "PNG":
            # PNG优化参数
            image.save(
                output,
                format="PNG",
                optimize=True,  # 启用优化
                compress_level=6  # 压缩级别 (0-9)
            )
        elif format.upper() == "WEBP":
            # WebP优化参数
            image.save(
                output,
                format="WEBP",
                quality=quality,
                method=6,  # 压缩方法 (0-6)
                lossless=False
            )
        else:
            # 默认JPEG
            image.save(output, format="JPEG", quality=quality, optimize=True)
        
        # 转换为base64
        image_data = output.getvalue()
        base64_data = base64.b64encode(image_data).decode('utf-8')
        
        # 返回完整的data URL
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{base64_data}"
    
    def _optimize_file_image(self, file_path: str) -> str:
        """
        优化文件路径图片
        
        Args:
            file_path: 图片文件路径
            
        Returns:
            str: 优化后的base64编码图片
        """
        try:
            # 处理file://前缀
            if file_path.startswith("file://"):
                file_path = file_path[7:]
            
            # 加载图片
            image = Image.open(file_path)
            return self._choose_compression_method(image)
            
        except Exception as e:
            raise ValueError(f"Failed to load image from {file_path}: {e}")
    
    def _optimize_pil_image(self, image: Union[Image.Image, Any]) -> str:
        """
        优化PIL Image或numpy数组
        
        Args:
            image: PIL Image或numpy数组
            
        Returns:
            str: 优化后的base64编码图片
        """
        try:
            if np is not None and isinstance(image, np.ndarray):
                # 转换numpy数组为PIL Image
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                image = Image.fromarray(image)
            elif Image is None:
                raise ValueError("PIL Image module not available")
            
            return self._choose_compression_method(image)
            
        except Exception as e:
            raise ValueError(f"Failed to optimize PIL image: {e}")
    
    def _optimize_unknown_image(self, image_input: Any) -> str:
        """
        优化未知类型的图片输入
        
        Args:
            image_input: 未知类型的图片输入
            
        Returns:
            str: 优化后的base64编码图片
        """
        try:
            # 尝试转换为PIL Image
            if hasattr(image_input, 'convert'):
                # 可能是PIL Image的子类
                image = image_input.convert("RGB")
            else:
                # 尝试其他转换方法
                image = Image.open(image_input)
            
            return self._choose_compression_method(image)
            
        except Exception as e:
            raise ValueError(f"Failed to optimize unknown image type {type(image_input)}: {e}")
    
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
_default_optimizer = ImageOptimizer()


def optimize_image_for_processing(
    image_input: Any,
    max_size: Tuple[int, int] = (1024, 1024),
    quality: int = 85,
    format: str = "JPEG"
) -> str:
    """
    便捷函数：优化图片用于处理
    
    这是对现有fetch_image函数的增强版本，提供智能的图片优化
    
    Args:
        image_input: 图片输入（URL、文件路径、PIL Image、base64等）
        max_size: 最大图片尺寸
        quality: 压缩质量
        format: 输出格式
        
    Returns:
        str: 优化后的图片（URL或base64编码）
    """
    # 创建临时优化器实例
    optimizer = ImageOptimizer(
        max_size=max_size,
        quality=quality,
        format=format,
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
