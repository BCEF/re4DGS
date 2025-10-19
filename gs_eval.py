import torch
import torchvision
import os
import argparse
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

# 导入必要的模块
from scene import Scene, GaussianModel
from gaussian_renderer import render_bribg as render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

# 尝试导入 lpips
try:
    from lpipsPyTorch import lpips
    LPIPS_AVAILABLE = True
except:
    print("Warning: lpipsPyTorch not available, LPIPS metric will be skipped")
    LPIPS_AVAILABLE = False

# 尝试导入 fused_ssim
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
    print("Warning: fused_ssim not available, using standard SSIM")


def evaluate_checkpoint(dataset, hyper, pipe, opt, start_checkpoint):
    """
    评估保存的checkpoint模型
    """
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {start_checkpoint}")
    print(f"{'='*60}\n")
    
    # 初始化高斯模型和场景
    gaussians = GaussianModel(dataset.sh_degree, hyper, opt.optimizer_type)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    
    # 加载checkpoint
    checkpoint_path = start_checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        # 尝试查找可用的checkpoint
        available_ckpts = [f for f in os.listdir(dataset.model_path) if f.startswith("chkpnt") and f.endswith(".pth")]
        if available_ckpts:
            print(f"Available checkpoints: {available_ckpts}")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model_params, iteration = torch.load(checkpoint_path, weights_only=False, map_location="cuda")
    gaussians.restore(model_params, opt)
    print(f"Checkpoint loaded successfully (iteration: {iteration})")
    
    # 设置背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 初始化指标统计
    results = {'psnr': [], 'ssim': [], 'l1': []}
    if LPIPS_AVAILABLE:
        results['lpips'] = []
    
    # 获取所有可用的训练相机信息
    all_train_cameras = scene.getAvailableCamInfos()['train_cameras']
    
    # 加载所有训练相机
    print(f"\nLoading all training cameras ({len(all_train_cameras)} views)...")
    scene.loadTrainCameras(all_train_cameras, dataset.rscale)
    
    # 初始化变形图
    if gaussians.dg is None:
        dg_path = os.path.join(dataset.source_path, 'deformation_graph.json')
        if os.path.exists(dg_path):
            print(f"Loading deformation graph from: {dg_path}")
            gaussians.deform_init(dg_path)
        else:
            print(f"Warning: Deformation graph not found at {dg_path}")
    
    # 获取训练相机
    cameras = scene.getTrainCameras(dataset.rscale)
    
    # 按kid分组相机
    cameras_by_kid = {}
    for viewpoint in cameras:
        kid = viewpoint.kid
        if kid not in cameras_by_kid:
            cameras_by_kid[kid] = []
        cameras_by_kid[kid].append(viewpoint)
    
    print(f"\n{'='*60}")
    print(f"Starting Evaluation on Training Cameras")
    print(f"Total views: {len(cameras)}")
    print(f"Found {len(cameras_by_kid)} unique keyframes (kids)")
    print(f"Kid distribution: {[(kid, len(cams)) for kid, cams in sorted(cameras_by_kid.items())]}")
    print(f"{'='*60}\n")
    
    # 创建保存目录
    save_dir = os.path.join(dataset.model_path, f"eval_iter_{iteration}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}\n")
    
    # 创建PLY保存目录
    ply_save_dir = os.path.join(save_dir, "deformed_gaussians")
    os.makedirs(ply_save_dir, exist_ok=True)
    
    saved_kids = set()
    
    # 遍历所有训练视角
    for idx, viewpoint in enumerate(tqdm(cameras, desc="Evaluating views")):
        kid = viewpoint.kid
        
        # 更新变形后的高斯点
        if hasattr(viewpoint, 'deformer_path') and hasattr(viewpoint, 'timecode'):
            gaussians.update_deformed_gaussians(viewpoint.deformer_path, viewpoint.timecode)
        
        # 如果这个kid还没保存过，则保存变形后的高斯点
        if kid not in saved_kids:
            ply_path = os.path.join(ply_save_dir, f"deformed_gaussians_kid_{kid}.ply")
            gaussians.save_ply(ply_path)
            saved_kids.add(kid)
            tqdm.write(f"  → Saved deformed gaussians for kid {kid}")
        
        # 处理背景图像
        bg = background
        if dataset.use_background_image and hasattr(viewpoint, 'bg_path') and os.path.exists(viewpoint.bg_path):
            bg = scene.get_background_image(viewpoint)
        
        # 渲染图像
        with torch.no_grad():
            render_pkg = render(viewpoint, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        
        # 应用 alpha mask（如果存在）
        if viewpoint.alpha_mask is not None:
            alpha_mask = viewpoint.alpha_mask.cuda()
            image = image * alpha_mask
            gt_image = gt_image * alpha_mask
        
        # 计算指标
        l1_val = l1_loss(image, gt_image).mean().item()
        psnr_val = psnr(image, gt_image).mean().item()
        
        if FUSED_SSIM_AVAILABLE:
            ssim_val = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
        else:
            ssim_val = ssim(image, gt_image).mean().item()
        
        # 保存指标
        results['l1'].append(l1_val)
        results['psnr'].append(psnr_val)
        results['ssim'].append(ssim_val)
        
        if LPIPS_AVAILABLE:
            lpips_val = lpips(image, gt_image, net_type='vgg').mean().item()
            results['lpips'].append(lpips_val)
        
        # 保存渲染图像（每10张保存一次以节省空间）
        if idx % 10 == 0 or idx < 5:
            img_name = viewpoint.image_name
            torchvision.utils.save_image(image, os.path.join(save_dir, f"{img_name}_render.png"))
            if idx < 5:  # 只保存前5张ground truth
                torchvision.utils.save_image(gt_image, os.path.join(save_dir, f"{img_name}_gt.png"))
    
    # 计算统计信息
    avg_l1 = np.mean(results['l1'])
    avg_psnr = np.mean(results['psnr'])
    avg_ssim = np.mean(results['ssim'])
    
    std_l1 = np.std(results['l1'])
    std_psnr = np.std(results['psnr'])
    std_ssim = np.std(results['ssim'])
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"Evaluation Results (Iteration {iteration}):")
    print(f"{'='*60}")
    print(f"  L1:    {avg_l1:.6f} ± {std_l1:.6f}")
    print(f"  PSNR:  {avg_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"  SSIM:  {avg_ssim:.6f} ± {std_ssim:.6f}")
    
    if LPIPS_AVAILABLE:
        avg_lpips = np.mean(results['lpips'])
        std_lpips = np.std(results['lpips'])
        print(f"  LPIPS: {avg_lpips:.6f} ± {std_lpips:.6f}")
    
    print(f"{'='*60}\n")
    
    # 保存详细结果
    results['avg'] = {
        'l1': float(avg_l1),
        'psnr': float(avg_psnr),
        'ssim': float(avg_ssim)
    }
    results['std'] = {
        'l1': float(std_l1),
        'psnr': float(std_psnr),
        'ssim': float(std_ssim)
    }
    
    if LPIPS_AVAILABLE:
        results['avg']['lpips'] = float(avg_lpips)
        results['std']['lpips'] = float(std_lpips)
    
    results['num_views'] = len(cameras)
    results['num_kids'] = len(cameras_by_kid)
    results['iteration'] = iteration
    results['saved_kids'] = sorted(list(saved_kids))
    
    # 保存结果到JSON文件
    results_file = os.path.join(save_dir, f"metrics.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_file}")
    
    # 生成markdown格式的结果表格
    markdown_file = os.path.join(save_dir, f"metrics.md")
    with open(markdown_file, 'w') as f:
        f.write(f"# Evaluation Results - Iteration {iteration}\n\n")
        f.write(f"**Model Path:** `{dataset.model_path}`\n\n")
        f.write(f"**Number of views:** {len(cameras)}\n\n")
        f.write(f"**Number of keyframes (kids):** {len(cameras_by_kid)}\n\n")
        f.write(f"**Saved kids:** {sorted(list(saved_kids))}\n\n")
        
        f.write("## Metrics\n\n")
        f.write("| Metric | Mean | Std |\n")
        f.write("|--------|------|-----|\n")
        f.write(f"| L1 | {results['avg']['l1']:.6f} | {results['std']['l1']:.6f} |\n")
        f.write(f"| PSNR (dB) | {results['avg']['psnr']:.4f} | {results['std']['psnr']:.4f} |\n")
        f.write(f"| SSIM | {results['avg']['ssim']:.6f} | {results['std']['ssim']:.6f} |\n")
        if LPIPS_AVAILABLE:
            f.write(f"| LPIPS | {results['avg']['lpips']:.6f} | {results['std']['lpips']:.6f} |\n")
        
        # 添加每个kid的相机数量信息
        f.write("\n## Keyframe Distribution\n\n")
        f.write("| Kid | Number of Cameras |\n")
        f.write("|-----|------------------|\n")
        for kid, cams in sorted(cameras_by_kid.items()):
            f.write(f"| {kid} | {len(cams)} |\n")
        
        # 添加保存的PLY文件列表
        f.write("\n## Saved Deformed Gaussian PLY Files\n\n")
        f.write(f"Location: `{ply_save_dir}`\n\n")
        for kid in sorted(list(saved_kids)):
            f.write(f"- `deformed_gaussians_kid_{kid}.ply`\n")
    
    print(f"Markdown report saved to: {markdown_file}")
    print(f"\nSaved {len(saved_kids)} deformed Gaussian PLY files to: {ply_save_dir}")
    print(f"  Kids saved: {sorted(list(saved_kids))}")
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate 3D Gaussian Splatting checkpoint")
    
    # 使用与训练脚本相同的参数解析器
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    op = OptimizationParams(parser)
    
    # 添加评估特定参数
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--start_checkpoint", type=str, default=None, help="Path to checkpoint file") 
    
    args = parser.parse_args()
    
    if args.start_checkpoint is None:
        print("Error: --start_checkpoint is required")
        exit(1)
    
    print(f"\nEvaluating model: {args.model_path}")
    print(f"Source path: {args.source_path}\n")
    
    # 提取参数
    dataset = lp.extract(args)
    hyper = hp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)
    
    # 运行评估
    results = evaluate_checkpoint(dataset, hyper, pipe, opt, args.start_checkpoint)
    
    if results is not None:
        print("\n✓ Evaluation completed successfully!")
    else:
        print("\n✗ Evaluation failed!")