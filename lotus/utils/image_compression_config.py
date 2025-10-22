"""
图片压缩配置管理器

为sem_filter等算子提供统一的图片压缩配置管理
"""

from typing import Any, Dict, Optional, Tuple
from lotus.utils.image_optimizer import ImageOptimizer


class ImageCompressionConfig:
    """
    图片压缩配置管理器
    
    提供统一的图片压缩配置管理，支持：
    1. 全局配置设置
    2. 算子级别的配置覆盖
    3. 智能压缩策略选择
    """
    
    def __init__(
        self,
        enable_compression: bool = True,
        strategy: str = "advanced",  # "simple" or "advanced"
        max_size: Tuple[int, int] = (1024, 1024),
        quality: int = 85,
        format: str = "JPEG",
        enable_cache: bool = True,
        cache_size: int = 1000
    ):
        """
        初始化图片压缩配置
        
        Args:
            enable_compression: 是否启用图片压缩
            strategy: 压缩策略 ("simple" or "advanced")
            max_size: 最大图片尺寸
            quality: 压缩质量 (1-100)
            format: 输出格式 ("JPEG", "PNG", "WEBP")
            enable_cache: 是否启用缓存
            cache_size: 缓存大小限制
        """
        self.enable_compression = enable_compression
        self.strategy = strategy
        self.max_size = max_size
        self.quality = quality
        self.format = format.upper()
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # 创建优化器实例
        self._optimizer: Optional[ImageOptimizer] = None
        self._create_optimizer()
    
    def _create_optimizer(self) -> None:
        """创建图片优化器实例"""
        if self.enable_compression:
            self._optimizer = ImageOptimizer(
                max_size=self.max_size,
                quality=self.quality,
                format=self.format,
                enable_cache=self.enable_cache,
                cache_size=self.cache_size,
                use_advanced_compression=(self.strategy == "advanced")
            )
        else:
            self._optimizer = None
    
    def update_config(
        self,
        enable_compression: Optional[bool] = None,
        strategy: Optional[str] = None,
        max_size: Optional[Tuple[int, int]] = None,
        quality: Optional[int] = None,
        format: Optional[str] = None,
        enable_cache: Optional[bool] = None,
        cache_size: Optional[int] = None
    ) -> None:
        """
        更新配置参数
        
        Args:
            enable_compression: 是否启用图片压缩
            strategy: 压缩策略
            max_size: 最大图片尺寸
            quality: 压缩质量
            format: 输出格式
            enable_cache: 是否启用缓存
            cache_size: 缓存大小限制
        """
        if enable_compression is not None:
            self.enable_compression = enable_compression
        if strategy is not None:
            self.strategy = strategy
        if max_size is not None:
            self.max_size = max_size
        if quality is not None:
            self.quality = quality
        if format is not None:
            self.format = format.upper()
        if enable_cache is not None:
            self.enable_cache = enable_cache
        if cache_size is not None:
            self.cache_size = cache_size
        
        # 重新创建优化器
        self._create_optimizer()
    
    def optimize_image(self, image_input: Any) -> str:
        """
        优化图片
        
        Args:
            image_input: 图片输入
            
        Returns:
            str: 优化后的图片（URL或base64编码）
        """
        if not self.enable_compression or self._optimizer is None:
            return str(image_input)
        
        return self._optimizer.optimize_image(image_input)
    
    def get_optimizer(self) -> Optional[ImageOptimizer]:
        """
        获取优化器实例
        
        Returns:
            ImageOptimizer or None: 优化器实例
        """
        return self._optimizer
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取当前配置
        
        Returns:
            Dict: 配置字典
        """
        return {
            "enable_compression": self.enable_compression,
            "strategy": self.strategy,
            "max_size": self.max_size,
            "quality": self.quality,
            "format": self.format,
            "enable_cache": self.enable_cache,
            "cache_size": self.cache_size
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        if self._optimizer is not None:
            self._optimizer.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        if self._optimizer is not None:
            return self._optimizer.get_cache_stats()
        return {"enabled": False}


# 全局配置实例
_global_config = ImageCompressionConfig()


def get_global_config() -> ImageCompressionConfig:
    """
    获取全局图片压缩配置
    
    Returns:
        ImageCompressionConfig: 全局配置实例
    """
    return _global_config


def set_global_config(
    enable_compression: Optional[bool] = None,
    strategy: Optional[str] = None,
    max_size: Optional[Tuple[int, int]] = None,
    quality: Optional[int] = None,
    format: Optional[str] = None,
    enable_cache: Optional[bool] = None,
    cache_size: Optional[int] = None
) -> None:
    """
    设置全局图片压缩配置
    
    Args:
        enable_compression: 是否启用图片压缩
        strategy: 压缩策略
        max_size: 最大图片尺寸
        quality: 压缩质量
        format: 输出格式
        enable_cache: 是否启用缓存
        cache_size: 缓存大小限制
    """
    _global_config.update_config(
        enable_compression=enable_compression,
        strategy=strategy,
        max_size=max_size,
        quality=quality,
        format=format,
        enable_cache=enable_cache,
        cache_size=cache_size
    )


def create_config_from_sem_filter_params(
    enable_image_compression: bool = True,
    image_compression_strategy: str = "advanced",
    image_max_size: Tuple[int, int] = (1024, 1024),
    image_quality: int = 85,
    image_format: str = "JPEG"
) -> ImageCompressionConfig:
    """
    从sem_filter参数创建图片压缩配置
    
    Args:
        enable_image_compression: 是否启用图片压缩
        image_compression_strategy: 压缩策略
        image_max_size: 最大图片尺寸
        image_quality: 压缩质量
        image_format: 输出格式
        
    Returns:
        ImageCompressionConfig: 配置实例
    """
    return ImageCompressionConfig(
        enable_compression=enable_image_compression,
        strategy=image_compression_strategy,
        max_size=image_max_size,
        quality=image_quality,
        format=image_format,
        enable_cache=True,
        cache_size=1000
    )
