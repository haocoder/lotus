#!/usr/bin/env python3
"""
独立的图片压缩测试脚本

测试图片压缩效果和base64编码大小优化，不依赖numpy
"""

import base64
import io
import time
from typing import Any, Dict, Optional, Tuple, Union

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

class StandaloneImageOptimizer:
    """
    独立的图片优化器 - 专门用于测试压缩效果
    """
    
    def __init__(
        self,
        max_size: Tuple[int, int] = (1024, 1024),
        quality: int = 85,
        format: str = "JPEG",
        enable_cache: bool = True,
        cache_size: int = 1000
    ):
        self.max_size = max_size
        self.quality = quality
        self.format = format.upper()
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # 缓存相关
        self._cache: Dict[str, str] = {}
        self._cache_access_times: Dict[str, float] = {}
        self._cache_sizes: Dict[str, int] = {}
    
    def is_url_image(self, image_input: Any) -> bool:
        """判断是否为URL图片"""
        if isinstance(image_input, str):
            return (
                image_input.startswith("http://") or 
                image_input.startswith("https://") or
                image_input.startswith("s3://")
            )
        return False
    
    def is_encoded_image(self, image_input: Any) -> bool:
        """判断是否为已编码的图片"""
        if isinstance(image_input, str):
            return image_input.startswith("data:image")
        return False
    
    def is_file_path(self, image_input: Any) -> bool:
        """判断是否为文件路径"""
        if isinstance(image_input, str):
            return (
                not self.is_url_image(image_input) and 
                not self.is_encoded_image(image_input) and
                ("/" in image_input or "\\" in image_input or image_input.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')))
            )
        return False
    
    def optimize_image(self, image_input: Any) -> str:
        """智能优化图片"""
        # URL图片直接返回
        if self.is_url_image(image_input):
            return image_input
        
        # 检查缓存
        if self.enable_cache:
            image_hash = self._get_image_hash(image_input)
            if image_hash in self._cache:
                self._cache_access_times[image_hash] = time.time()
                return self._cache[image_hash]
        
        # 处理不同类型的图片
        if self.is_encoded_image(image_input):
            optimized = self._optimize_encoded_image(image_input)
        elif self.is_file_path(image_input):
            optimized = self._optimize_file_image(image_input)
        elif isinstance(image_input, Image.Image):
            optimized = self._optimize_pil_image(image_input)
        else:
            optimized = str(image_input)
        
        # 更新缓存
        if self.enable_cache:
            if len(self._cache) >= self.cache_size:
                self._evict_cache()
            
            image_hash = self._get_image_hash(image_input)
            self._cache[image_hash] = optimized
            self._cache_access_times[image_hash] = time.time()
            self._cache_sizes[image_hash] = len(optimized)
        
        return optimized
    
    def _get_image_hash(self, image_input: Any) -> str:
        """生成图片哈希值"""
        import hashlib
        if isinstance(image_input, str):
            return hashlib.md5(image_input.encode()).hexdigest()
        else:
            return hashlib.md5(str(image_input).encode()).hexdigest()
    
    def _evict_cache(self) -> None:
        """清理缓存"""
        if not self._cache:
            return
        
        sorted_items = sorted(
            self._cache_access_times.items(),
            key=lambda x: x[1]
        )
        
        remove_count = max(1, len(sorted_items) // 4)
        for key, _ in sorted_items[:remove_count]:
            if key in self._cache:
                del self._cache[key]
            if key in self._cache_access_times:
                del self._cache_access_times[key]
            if key in self._cache_sizes:
                del self._cache_sizes[key]
    
    def _optimize_encoded_image(self, encoded_image: str) -> str:
        """优化已编码的图片"""
        if not PIL_AVAILABLE:
            return encoded_image
        
        try:
            if "base64," in encoded_image:
                header, data = encoded_image.split("base64,", 1)
                image_data = base64.b64decode(data)
            else:
                image_data = base64.b64decode(encoded_image)
            
            image = Image.open(io.BytesIO(image_data))
            current_size = len(image_data)
            
            if current_size < 100 * 1024:  # 小于100KB
                return encoded_image
            
            return self._compress_image(image)
            
        except Exception as e:
            print(f"Warning: Failed to optimize encoded image: {e}")
            return encoded_image
    
    def _optimize_file_image(self, file_path: str) -> str:
        """优化文件路径图片"""
        if not PIL_AVAILABLE:
            return file_path
        
        try:
            if file_path.startswith("file://"):
                file_path = file_path[7:]
            
            image = Image.open(file_path)
            return self._compress_image(image)
            
        except Exception as e:
            raise ValueError(f"Failed to load image from {file_path}: {e}")
    
    def _optimize_pil_image(self, image) -> str:
        """优化PIL Image"""
        if not PIL_AVAILABLE:
            return str(image)
        
        return self._compress_image(image)
    
    def _compress_image(self, image) -> str:
        """智能压缩图片"""
        if not PIL_AVAILABLE:
            return "data:image/jpeg;base64,"
        
        # 转换为RGB模式
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 渐进式压缩策略
        compression_levels = [
            (self.max_size, self.quality, self.format),
            ((self.max_size[0]//2, self.max_size[1]//2), max(60, self.quality-10), "JPEG"),
            ((self.max_size[0]//3, self.max_size[1]//3), max(40, self.quality-20), "JPEG"),
            ((self.max_size[0]//4, self.max_size[1]//4), max(20, self.quality-30), "JPEG"),
        ]
        
        best_result = None
        best_size = float('inf')
        
        for max_size, quality, format in compression_levels:
            try:
                compressed_image = self._resize_image(image, max_size)
                result = self._encode_with_quality(compressed_image, quality, format)
                
                result_size = len(result)
                if result_size < best_size:
                    best_result = result
                    best_size = result_size
                
                if result_size < 200 * 1024:  # 小于200KB
                    break
                    
            except Exception as e:
                print(f"Warning: Compression level failed: {e}")
                continue
        
        return best_result or self._encode_with_quality(image, self.quality, self.format)
    
    def _resize_image(self, image, max_size: Tuple[int, int]):
        """智能调整图片尺寸"""
        width, height = image.size
        max_width, max_height = max_size
        
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resample = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            return image.resize((new_width, new_height), resample=resample)
        
        return image
    
    def _encode_with_quality(self, image, quality: int, format: str) -> str:
        """使用指定质量和格式编码图片"""
        output = io.BytesIO()
        
        if format.upper() == "JPEG":
            image.save(
                output, 
                format="JPEG", 
                quality=quality,
                optimize=True,
                progressive=True,
                subsampling=0 if quality > 80 else 1
            )
        elif format.upper() == "PNG":
            image.save(
                output,
                format="PNG",
                optimize=True,
                compress_level=6
            )
        elif format.upper() == "WEBP":
            image.save(
                output,
                format="WEBP",
                quality=quality,
                method=6,
                lossless=False
            )
        else:
            image.save(output, format="JPEG", quality=quality, optimize=True)
        
        image_data = output.getvalue()
        base64_data = base64.b64encode(image_data).decode('utf-8')
        
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{base64_data}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.enable_cache:
            return {"enabled": False}
        
        total_size = sum(self._cache_sizes.values())
        return {
            "enabled": True,
            "cache_size": len(self._cache),
            "max_size": self.cache_size,
            "total_bytes": total_size
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._cache_access_times.clear()
        self._cache_sizes.clear()


def test_compression_effects():
    """测试压缩效果"""
    print("=== 图片压缩效果测试 ===")
    
    if not PIL_AVAILABLE:
        print("PIL不可用，跳过压缩测试")
        return
    
    # 创建测试图片
    test_sizes = [
        (100, 100, "小图片"),
        (500, 500, "中等图片"),
        (1000, 1000, "大图片"),
        (2048, 2048, "超大图片")
    ]
    
    compression_configs = [
        {"max_size": (1024, 1024), "quality": 85, "format": "JPEG", "name": "标准压缩"},
        {"max_size": (512, 512), "quality": 70, "format": "JPEG", "name": "中等压缩"},
        {"max_size": (256, 256), "quality": 50, "format": "JPEG", "name": "高压缩"},
        {"max_size": (128, 128), "quality": 30, "format": "JPEG", "name": "极高压缩"},
    ]
    
    print("--- 压缩效果对比 ---")
    for size, name in test_sizes:
        print(f"\n[{name}] ({size[0]}x{size[1]}):")
        
        # 创建测试图片
        test_image = Image.new('RGB', size, color='red')
        original_size = len(test_image.tobytes())
        
        for config in compression_configs:
            optimizer = StandaloneImageOptimizer(
                max_size=config["max_size"],
                quality=config["quality"],
                format=config["format"]
            )
            
            start_time = time.time()
            compressed = optimizer._compress_image(test_image)
            end_time = time.time()
            
            compressed_size = len(compressed)
            compression_ratio = (1 - compressed_size / original_size) * 100
            processing_time = end_time - start_time
            
            print(f"  {config['name']}: {compressed_size:,} bytes ({compression_ratio:.1f}% 压缩) - {processing_time:.3f}s")


def test_base64_optimization():
    """测试base64编码优化"""
    print("\n=== Base64编码优化测试 ===")
    
    if not PIL_AVAILABLE:
        print("PIL不可用，跳过Base64测试")
        return
    
    # 测试不同格式
    formats = ["JPEG", "PNG", "WEBP"]
    qualities = [30, 50, 70, 85]
    
    print("--- 格式和质量对比 ---")
    for format in formats:
        print(f"\n[{format}格式]:")
        for quality in qualities:
            test_image = Image.new('RGB', (500, 500), color='blue')
            
            optimizer = StandaloneImageOptimizer(
                max_size=(400, 400),
                quality=quality,
                format=format
            )
            
            compressed = optimizer._compress_image(test_image)
            base64_size = len(compressed)
            
            print(f"  质量{quality}: {base64_size:,} bytes")


def test_url_detection():
    """测试URL检测功能"""
    print("\n=== URL检测测试 ===")
    
    optimizer = StandaloneImageOptimizer()
    
    test_cases = [
        ("https://example.com/image.jpg", True, False, False),
        ("http://test.com/pic.png", True, False, False),
        ("s3://bucket/image.webp", True, False, False),
        ("data:image/jpeg;base64,/9j/4AAQ", False, True, False),
        ("/path/to/local/image.jpg", False, False, True),
        ("C:\\Users\\image.png", False, False, True),
        ("image.jpg", False, False, True),
    ]
    
    print("--- URL检测结果 ---")
    for input_str, expected_url, expected_encoded, expected_file in test_cases:
        is_url = optimizer.is_url_image(input_str)
        is_encoded = optimizer.is_encoded_image(input_str)
        is_file = optimizer.is_file_path(input_str)
        
        print(f"输入: {input_str[:30]}...")
        print(f"  URL: {is_url} (期望: {expected_url})")
        print(f"  编码: {is_encoded} (期望: {expected_encoded})")
        print(f"  文件: {is_file} (期望: {expected_file})")
        print()


def test_cache_functionality():
    """测试缓存功能"""
    print("\n=== 缓存功能测试 ===")
    
    optimizer = StandaloneImageOptimizer(enable_cache=True, cache_size=5)
    
    # 模拟缓存操作
    print("--- 缓存操作测试 ---")
    for i in range(8):  # 超过缓存大小
        key = f"image_{i}"
        optimizer._cache[key] = f"compressed_{i}"
        optimizer._cache_access_times[key] = time.time()
        optimizer._cache_sizes[key] = 1000
        time.sleep(0.001)  # 确保时间不同
    
    print(f"缓存大小: {len(optimizer._cache)}")
    print(f"缓存统计: {optimizer.get_cache_stats()}")
    
    # 清理缓存
    optimizer.clear_cache()
    print(f"清理后缓存大小: {len(optimizer._cache)}")


def main():
    """主测试函数"""
    print("开始独立的图片压缩测试")
    print("=" * 50)
    
    # 测试压缩效果
    test_compression_effects()
    
    # 测试base64优化
    test_base64_optimization()
    
    # 测试URL检测
    test_url_detection()
    
    # 测试缓存功能
    test_cache_functionality()
    
    print("\n" + "=" * 50)
    print("所有测试完成")


if __name__ == "__main__":
    main()
