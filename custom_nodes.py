import numpy as np
from PIL import Image, ImageDraw, ImageMath
import torch
from typing import Tuple
import random
import math

class JigsawPuzzleEffect:
    def __init__(self):
        self.type = "JigsawPuzzleEffect"
        self.output_node = True
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "piece_size": ("INT", {
                    "default": 50,
                    "min": 20,
                    "max": 200,
                    "step": 1
                }),
                "num_missing": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "step": 1
                }),
                "stroke_opacity": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "emboss_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "image/effects"

    def draw_puzzle_piece(self, draw, x, y, size, width, height):
        """绘制单个拼图块的形状，包括四边的凹凸"""
        tab = size * 0.25  # 调整凸起/凹陷的大小
        radius = size * 0.1  # 圆角半径
        steps = 60  # 增加曲线的平滑度
        
        def bezier_curve(p0, p1, p2, p3, steps):
            """生成三次贝塞尔曲线的点"""
            points = []
            for t in range(steps + 1):
                t = t / steps
                t2 = t * t
                t3 = t2 * t
                u = 1 - t
                u2 = u * u
                u3 = u2 * u
                x = u3*p0[0] + 3*u2*t*p1[0] + 3*u*t2*p2[0] + t3*p3[0]
                y = u3*p0[1] + 3*u2*t*p1[1] + 3*u*t2*p2[1] + t3*p3[1]
                points.append((x, y))
            return points

        points = []
        
        # 上边
        if y == 0:
            points.extend([(x, y), (x + size, y)])
        else:
            points.extend([(x, y), (x + (size - tab) / 2, y)])
            # 绘制凸起
            center_x = x + size / 2
            center_y = y
            for i in range(steps + 1):
                angle = math.pi - (math.pi * i / steps)
                px = center_x + tab/2 * math.cos(angle)
                py = center_y - tab/2 * math.sin(angle)
                points.append((px, py))
            points.extend([(x + (size + tab) / 2, y), (x + size, y)])

        # 右边
        if x + size >= width:
            points.extend([(x + size, y), (x + size, y + size)])
        else:
            points.extend([(x + size, y), (x + size, y + (size - tab) / 2)])
            # 绘制凸起
            center_x = x + size
            center_y = y + size / 2
            for i in range(steps + 1):
                angle = -math.pi/2 + (math.pi * i / steps)
                px = center_x + tab/2 * math.cos(angle)
                py = center_y + tab/2 * math.sin(angle)
                points.append((px, py))
            points.extend([(x + size, y + (size + tab) / 2), (x + size, y + size)])

        # 下边
        if y + size >= height:
            points.extend([(x + size, y + size), (x, y + size)])
        else:
            points.extend([(x + size, y + size), (x + (size + tab) / 2, y + size)])
            # 绘制凹陷
            center_x = x + size / 2
            center_y = y + size
            for i in range(steps + 1):
                angle = (math.pi * i / steps)
                px = center_x + tab/2 * math.cos(angle)
                py = center_y + tab/2 * math.sin(angle)
                points.append((px, py))
            points.extend([(x + (size - tab) / 2, y + size), (x, y + size)])

        # 左边
        if x == 0:
            points.extend([(x, y + size), (x, y)])
        else:
            points.extend([(x, y + size), (x, y + (size + tab) / 2)])
            # 绘制凹陷
            center_x = x
            center_y = y + size / 2
            for i in range(steps + 1):
                angle = math.pi/2 + (math.pi * i / steps)
                px = center_x + tab/2 * math.cos(angle)
                py = center_y + tab/2 * math.sin(angle)
                points.append((px, py))
            points.extend([(x, y + (size - tab) / 2), (x, y)])

        # 绘制拼图块的基本形状
        draw.polygon(points, fill=255)
        
        # 创建渐变遮罩
        gradient_mask = Image.new('L', (width, height), 0)
        gradient_draw = ImageDraw.Draw(gradient_mask)
        
        # 添加内部阴影
        shadow_points = [(px+1, py+1) for px, py in points]
        gradient_draw.polygon(shadow_points, outline=100)
        
        # 添加高光
        highlight_points = [(px-1, py-1) for px, py in points]
        gradient_draw.polygon(highlight_points, outline=200)
        
        return points, gradient_mask

    def draw_smooth_curve(self, draw, x1, y1, x2, y2, x3, y3, width=1):
        """绘制平滑的贝塞尔曲线"""
        # 生成贝塞尔曲线的点
        points = []
        steps = 10
        for t in range(steps + 1):
            t = t / steps
            # 二次贝塞尔曲线公式
            px = (1-t)**2 * x1 + 2*(1-t)*t * x2 + t**2 * x3
            py = (1-t)**2 * y1 + 2*(1-t)*t * y2 + t**2 * y3
            points.append((px, py))
        
        # 绘制曲线
        for i in range(len(points)-1):
            draw.line([points[i], points[i+1]], fill=100, width=width)

    def draw_tab_highlight(self, draw, x, y, size, strength, horizontal=True):
        """绘制凸起部分的渐变高光效果"""
        steps = 10
        for i in range(steps):
            opacity = int((1 - i/steps) * 255 * strength)
            if horizontal:
                draw.line(
                    [(x - size + i, y - size/2), (x + size - i, y - size/2)],
                    fill=255-opacity,
                    width=1
                )
            else:
                draw.line(
                    [(x + size/2, y - size + i), (x + size/2, y + size - i)],
                    fill=255-opacity,
                    width=1
                )

    def int_to_rgb(self, color_int: int) -> tuple:
        """将整数颜色值转换为RGB元组"""
        r = (color_int >> 16) & 255
        g = (color_int >> 8) & 255
        b = color_int & 255
        return (r, g, b)

    def apply_effect(self, image: torch.Tensor, piece_size: int, num_missing: int,
                    stroke_opacity: float, emboss_strength: float) -> Tuple[torch.Tensor]:
        # 转换为PIL图像
        image_np = image.cpu().numpy()
        image_pil = Image.fromarray((image_np[0] * 255).astype(np.uint8))
        
        width, height = image_pil.size
        
        # 创建白色背景
        result = Image.new('RGB', (width, height), (255, 255, 255))
        
        # 计算拼图块数量和位置
        columns = width // piece_size
        rows = height // piece_size
        total_pieces = columns * rows
        
        # 生成缺失拼图块的位置
        if num_missing > 0:
            missing_pieces = set(random.sample(range(total_pieces), min(num_missing, total_pieces)))
        else:
            missing_pieces = set()
        
        # 为每个非缺失的拼图块创建遮罩
        piece_index = 0
        for y in range(0, height, piece_size):
            for x in range(0, width, piece_size):
                if piece_index not in missing_pieces:
                    # 创建拼图块遮罩和渐变遮罩
                    piece_mask = Image.new('L', (width, height), 0)
                    draw = ImageDraw.Draw(piece_mask)
                    
                    # 绘制拼图块形状和获取渐变遮罩
                    points, gradient_mask = self.draw_puzzle_piece(draw, x, y, piece_size, width, height)
                    
                    # 应用图像
                    result.paste(image_pil, mask=piece_mask)
                    
                    # 应用渐变效果
                    gradient_strength = max(0.3, emboss_strength)  # 确保始终有最小的渐变效果
                    gradient = Image.new('RGB', (width, height), (255, 255, 255))
                    
                    # 修改这部分代码
                    adjusted_mask = Image.eval(
                        gradient_mask,
                        lambda x: int(x * gradient_strength)
                    )
                    result.paste(gradient, mask=adjusted_mask)
                    
                    # 绘制边框
                    draw = ImageDraw.Draw(result)
                    draw.polygon(points, outline=(200, 200, 200), width=max(1, int(2 * emboss_strength)))
                
                piece_index += 1
        
        # 转换回torch张量
        result_np = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)
        
        return (result_tensor,)

    def draw_highlight(self, draw, x, y, radius, strength):
        """绘制高光效果"""
        # 创建渐变高光效果
        for i in range(int(radius)):
            opacity = int((1 - i/radius) * 255 * strength)
            draw.arc([x-i, y-i, x+i, y+i], 0, 360, fill=opacity)

class RegionBoundaryEffect:
    def __init__(self):
        self.type = "RegionBoundaryEffect"
        self.output_node = True
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segments": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 500,
                    "step": 10
                }),
                "compactness": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "line_width": ("INT", {  # 添加线条宽度控制
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "image/effects"

    def apply_effect(self, image: torch.Tensor, segments: int, 
                    compactness: int, line_width: int) -> Tuple[torch.Tensor]:
        try:
            from skimage.segmentation import slic
        except ImportError:
            raise ImportError("Please install scikit-image: pip install scikit-image")
            
        # 将torch张量转换为numpy数组
        image_np = image.cpu().numpy()
        image_pil = Image.fromarray((image_np[0] * 255).astype(np.uint8))
        
        # 转换为numpy数组并进行分割
        img_array = np.array(image_pil)
        segments = slic(img_array, n_segments=segments, compactness=compactness)
        
        # 创建新图像用于绘制边界
        result = image_pil.copy()
        draw = ImageDraw.Draw(result)
        
        # 绘制区域边界
        height, width = segments.shape
        for y in range(1, height):
            for x in range(1, width):
                if segments[y, x] != segments[y-1, x] or segments[y, x] != segments[y, x-1]:
                    # 使用白色线条，并应用指定的线条宽度
                    for i in range(line_width):
                        for j in range(line_width):
                            if (x+i < width and y+j < height):
                                draw.point((x+i, y+j), fill=(255, 255, 255))
        
        # 转换回torch张量
        result_np = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)
        
        return (result_tensor,)
