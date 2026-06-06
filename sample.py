import json
import torchmetrics
import time
import importlib
import argparse
import torch
import os
from dataset import TestDataset
from util import calc_psnr, tensor_rgb2y
import torch.nn as nn
from PIL import Image
from typing import Tuple
from thop import profile, clever_format
# from torchinfo import summary
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="超分模型推理与评估脚本")
    parser.add_argument("--model", type=str, default='CAMnet_v4', help="模型名称")
    parser.add_argument("--ckpt_path", type=str, default='./checkpoints/loss/0423A_DF2K_XR_CHA_X4_50w_500000.pth.tar',
                        help="模型权重路径")
    parser.add_argument("--group", type=int, default=4, help="模型分组数")
    parser.add_argument("--sample_dir", type=str, default='sample/0423A_DF2K_XR_CHA_X4_50w/Manga109',
                        help="结果保存根目录")
    parser.add_argument("--test_data_dir", type=str, default="./dataset/Manga109/image_SRF_4",
                        help="测试数据集路径")
    parser.add_argument("--scale", type=int, default=4, help="超分放大倍数")
    parser.add_argument("--shave", type=int, default=4, help="图像裁剪边距（避免边界效应）")
    parser.add_argument("--num_gpu", type=int, default=4, help="GPU数量（预留参数）")
    parser.add_argument("--warmup_runs", type=int, default=50,
                        help="GPU预热迭代次数（消除初始推理耗时波动）")
    parser.add_argument("--warmup_h", type=int, default=64, help="预热输入图像高度")
    parser.add_argument("--warmup_w", type=int, default=64, help="预热输入图像宽度")
    # --------------------- 新增FLOPs统计参数 ---------------------
    parser.add_argument("--flops_input_h", type=int, default=180, help="统计FLOPs用的输入高度（匹配论文标准）")
    parser.add_argument("--flops_input_w", type=int, default=320, help="统计FLOPs用的输入宽度（匹配论文标准）")
    return parser.parse_args()


def save_image(tensor: torch.Tensor, filename: str) -> None:
    """
    将张量保存为图像
    Args:
        tensor: 形状为(C, H, W)的张量，值范围[0,1]
        filename: 保存路径
    """
    try:
        tensor = tensor.cpu()
        ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(ndarr)
        im.save(filename)
    except Exception as e:
        raise RuntimeError(f"保存图像{filename}失败: {str(e)}")


def infer_large_image(net: nn.Module, lr: torch.Tensor, cfg: argparse.Namespace) -> torch.Tensor:
    """
    大图像分块推理（避免OOM）
    Args:
        net: 超分模型
        lr: 低分辨率图像张量 (C, H, W)
        cfg: 配置参数
    Returns:
        sr: 超分后图像张量 (C, H*scale, W*scale)
    """
    scale = cfg.scale
    shave = cfg.shave
    h, w = lr.size()[1:]
    h_half, w_half = h // 2, w // 2
    h_chop, w_chop = h_half + shave, w_half + shave

    # 切分4个patch
    lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop).cuda()
    lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
    lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
    lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
    lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])

    # 模型推理
    with torch.no_grad():
        sr_patches = net(lr_patch, scale)

    # 计算超分后尺寸
    h_sr, h_half_sr, h_chop_sr = h * scale, h_half * scale, h_chop * scale
    w_sr, w_half_sr, w_chop_sr = w * scale, w_half * scale, w_chop * scale

    # 合并patch
    sr = torch.FloatTensor(3, h_sr, w_sr).cuda()
    sr[:, 0:h_half_sr, 0:w_half_sr].copy_(sr_patches[0, :, 0:h_half_sr, 0:w_half_sr])
    sr[:, 0:h_half_sr, w_half_sr:w_sr].copy_(sr_patches[1, :, 0:h_half_sr, w_chop_sr - w_sr + w_half_sr:w_chop_sr])
    sr[:, h_half_sr:h_sr, 0:w_half_sr].copy_(sr_patches[2, :, h_chop_sr - h_sr + h_half_sr:h_chop_sr, 0:w_half_sr])
    sr[:, h_half_sr:h_sr, w_half_sr:w_sr].copy_(sr_patches[3, :, h_chop_sr - h_sr + h_half_sr:h_chop_sr,
                                                w_chop_sr - w_sr + w_half_sr:w_chop_sr])
    return sr


def infer_normal_image(net: nn.Module, lr: torch.Tensor, cfg: argparse.Namespace) -> torch.Tensor:
    """
    常规尺寸图像推理
    Args:
        net: 超分模型
        lr: 低分辨率图像张量 (C, H, W)
        cfg: 配置参数
    Returns:
        sr: 超分后图像张量 (C, H*scale, W*scale)
    """
    lr = lr.unsqueeze(0).cuda()
    with torch.no_grad():
        sr = net(lr, cfg.scale)
    return sr.squeeze(0)


def calculate_metrics(sr: torch.Tensor, hr: torch.Tensor, scale: int,
                      ssim_calculator: torchmetrics.StructuralSimilarityIndexMeasure,
                      device: torch.device) -> Tuple[float, float]:
    """
    计算Y通道的PSNR和SSIM指标
    Args:
        sr: 超分图像张量 (C, H, W)
        hr: 高清图像张量 (C, H, W)
        scale: 超分倍数
        ssim_calculator: SSIM计算实例
        device: 计算设备
    Returns:
        psnr: PSNR值
        ssim: SSIM值
    """
    # 转换为(B, C, H, W)格式并限制范围[0,1]
    sr_rgb = torch.clamp(sr.detach().unsqueeze(0), 0.0, 1.0).to(device)
    hr_rgb = torch.clamp(hr.detach().unsqueeze(0), 0.0, 1.0).to(device)

    # 转换到Y通道
    sr_y = tensor_rgb2y(sr_rgb)
    hr_y = tensor_rgb2y(hr_rgb)

    # 计算PSNR（Y通道）
    psnr = calc_psnr(sr_y, hr_y, scale, 1, benchmark=True)

    # 计算SSIM（裁剪边界后，Y通道）
    shave = scale
    sr_y_shaved = sr_y[:, :, shave:-shave, shave:-shave]
    hr_y_shaved = hr_y[:, :, shave:-shave, shave:-shave]
    ssim = ssim_calculator(sr_y_shaved, hr_y_shaved).item()

    return psnr, ssim


def setup_output_dirs(cfg: argparse.Namespace, model_name: str) -> Tuple[str, str]:
    """
    创建输出目录
    Args:
        cfg: 配置参数
        model_name: 模型名称（从权重路径提取）
    Returns:
        sr_dir: 超分图像保存目录
        hr_dir: 高清图像保存目录
    """
    dataset_name = os.path.basename(cfg.test_data_dir)
    base_dir = os.path.join(cfg.sample_dir, model_name, dataset_name, f"x{cfg.scale}")
    sr_dir = os.path.join(base_dir, "SR")
    hr_dir = os.path.join(base_dir, "HR")

    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)
    return sr_dir, hr_dir


# --------------------- 计算FLOPs和参数量的核心函数 ---------------------
def calculate_flops_params(net: nn.Module, cfg: argparse.Namespace, device: torch.device) -> Tuple[str, str, str]:
    """
    计算模型的FLOPs、参数量、Multi-Adds（强制适配scale参数）
    Args:
        net: 已加载权重的模型
        cfg: 配置参数（含FLOPs统计用的输入尺寸）
        device: 计算设备
    Returns:
        flops: 格式化的FLOPs（如 128.50M）
        params: 格式化的参数量（如 480.00K）
        madds: 格式化的Multi-Adds（如 64.25M）
    """
    # 构建标准输入张量（batch_size=1，匹配论文统计方式）
    input_size = (1, 3, cfg.flops_input_h, cfg.flops_input_w)  # (B, C, H, W)
    dummy_input = torch.randn(input_size).to(device)

    class ModelWrapper(nn.Module):
        def __init__(self, model, scale):
            super().__init__()
            self.model = model
            self.scale = scale

        def forward(self, x):
            return self.model(x, self.scale)

    wrapped_net = ModelWrapper(net, cfg.scale).to(device)

    # 统计FLOPs和参数量
    try:
        # 兼容写法：移除 ignore_modules，避免版本冲突
        # 激活函数等轻量级操作的 FLOPs 占比极低，直接统计不影响宏观评估
        flops, params = profile(
            wrapped_net,
            inputs=(dummy_input,),
            verbose=False
        )
    except Exception as e:
        # 如果 profile 依然失败（极少见），返回默认值避免程序崩溃
        print(f"警告：FLOPs 统计遇到未知错误 ({str(e)})，将返回估算值 0")
        flops, params = 0, 0

    # 格式化单位（自动转换 K/M/G）
    flops_formatted, params_formatted = clever_format([flops, params], "%.2f")

    # 换算Multi-Adds（SISR论文通用：Multi-Adds ≈ FLOPs / 2）
    flops_raw = float(flops_formatted[:-1])
    flops_unit = flops_formatted[-1]
    madds_formatted = f"{flops_raw / 2:.2f}{flops_unit}"

    return flops_formatted, params_formatted, madds_formatted

def sample(net: nn.Module, dataset: TestDataset, cfg: argparse.Namespace,
           flops: str, params: str, madds: str) -> None:
    """
    推理并评估数据集（新增FLOPs参数传递）
    Args:
        net: 超分模型
        dataset: 测试数据集
        cfg: 配置参数
        flops: 格式化的FLOPs
        params: 格式化的参数量
        madds: 格式化的Multi-Adds
    """
    # 初始化指标
    avg_psnr = 0.0
    total_inference_time = 0.0
    num_images = 0
    total_ssim = 0.0

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssim_calculator = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 提取模型名称
    model_name = os.path.splitext(os.path.basename(cfg.ckpt_path))[0]

    # 打印FLOPs/参数量信息（仅打印一次）
    print("\n=== 模型计算量统计 ===")
    print(f"参数量 (Params): {params}")
    print(f"浮点运算量 (FLOPs): {flops}")
    print(f"乘加操作数 (Multi-Adds): {madds}")
    print(f"超分倍数: X{cfg.scale} | FLOPs统计输入尺寸: {cfg.flops_input_h}x{cfg.flops_input_w}")
    print("-" * 50)

    for step, (hr, lr, img_name) in enumerate(dataset):
        # 推理
        start_time = time.time()
        if "urban100" in dataset.name:
            sr = infer_large_image(net, lr, cfg)
        else:
            sr = infer_normal_image(net, lr, cfg)
        inference_time = time.time() - start_time

        # 更新统计
        total_inference_time += inference_time
        num_images += 1

        # 创建输出目录
        sr_dir, hr_dir = setup_output_dirs(cfg, model_name)

        # 保存图像
        img_basename = os.path.basename(img_name)
        sr_save_path = os.path.join(sr_dir, img_basename.replace("HR", "SR"))
        hr_save_path = os.path.join(hr_dir, img_basename)
        save_image(sr, sr_save_path)
        save_image(hr, hr_save_path)

        # 计算指标
        psnr, ssim = calculate_metrics(sr, hr, cfg.scale, ssim_calculator, device)
        avg_psnr += psnr / len(dataset)
        total_ssim += ssim
        avg_ssim = total_ssim / num_images

        # 单张图像日志
        lr_h, lr_w = lr.shape[1], lr.shape[2]
        sr_h, sr_w = sr.shape[1], sr.shape[2]
        print(f"[{step + 1}/{len(dataset)}] 保存: {sr_save_path} | "
              f"输入尺寸: {lr_h}x{lr_w} | 输出尺寸: {sr_h}x{sr_w} | "
              f"推理耗时: {inference_time:.3f}s | "
              f"PSNR: {psnr:.4f} | SSIM: {ssim:.4f}")

    # 最终统计日志
    avg_fps = num_images / total_inference_time if num_images > 0 else 0.0
    print("\n=== 最终评估结果 ===")
    print(f"超分倍数: X{cfg.scale}")
    print(f"处理图像数量: {num_images}")
    print(f"平均PSNR: {avg_psnr:.4f}")
    print(f"平均SSIM: {avg_ssim:.4f}")
    print(f"总推理时间: {total_inference_time:.2f}s")
    print(f"平均FPS: {avg_fps:.2f} img/s")
    # 补充打印FLOPs/参数量到最终结果
    print(f"模型参数量: {params}")
    print(f"模型FLOPs: {flops}")
    print(f"模型Multi-Adds: {madds}")


def warmup_gpu(net: nn.Module, cfg: argparse.Namespace, device: torch.device) -> None:
    """
    GPU预热（消除初始推理耗时波动）
    Args:
        net: 超分模型
        cfg: 配置参数
        device: 计算设备
    """
    if device.type != 'cuda' or cfg.warmup_runs <= 0:
        return

    print(f"开始GPU预热，迭代次数: {cfg.warmup_runs}")
    dummy_input = torch.randn(1, 3, cfg.warmup_h, cfg.warmup_w).to(device)
    net.eval()
    with torch.no_grad():
        for _ in range(cfg.warmup_runs):
            _ = net(dummy_input, cfg.scale)
    torch.cuda.synchronize()
    print("GPU预热完成")


def main(cfg: argparse.Namespace) -> None:
    """主函数"""
    # 打印配置
    print("=== 运行配置 ===")
    print(json.dumps(vars(cfg), indent=4, ensure_ascii=False))

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 加载模型
    try:
        module = importlib.import_module(cfg.model)
        net = module.Net(multi_scale=False, group=cfg.group, scale=cfg.scale)

        # 加载权重
        checkpoint = torch.load(cfg.ckpt_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(device)
        net.eval()
        print("模型加载成功")
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")

    print("\n开始统计模型计算量...")
    flops, params, madds = calculate_flops_params(net, cfg, device)
    print("模型计算量统计完成")

    # GPU预热
    warmup_gpu(net, cfg, device)

    # 加载数据集
    try:
        dataset = TestDataset(cfg.test_data_dir, cfg.scale)
        print(f"\n数据集加载成功，共 {len(dataset)} 张图像")
    except Exception as e:
        raise RuntimeError(f"数据集加载失败: {str(e)}")

    # 推理与评估（传递FLOPs参数）
    sample(net, dataset, cfg, flops, params, madds)

if __name__ == "__main__":
    try:
        cfg = parse_args()
        main(cfg)
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        exit(1)