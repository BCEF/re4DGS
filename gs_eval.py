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


def evaluate_checkpoint(dataset, hyper, pipe, opt, start_checkpoint, batch_size=None):
    """
    评估保存的checkpoint模型（分批处理以节省显存）
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
    
    # 初始化指标统计（全局）
    all_results = {'psnr': [], 'ssim': [], 'l1': []}
    if LPIPS_AVAILABLE:
        all_results['lpips'] = []
    
    # 获取所有可用的训练相机信息
    all_train_cameras = scene.getAvailableCamInfos()['train_cameras']
    
    # 计算batch参数（参考训练脚本）
    if batch_size is None:
        num_batches = dataset.batchnum - 1
        batch_size = max(1, len(all_train_cameras) // num_batches)
    else:
        num_batches = max(0, (len(all_train_cameras) - 1) // batch_size)
    
    print(f"\nTotal cameras: {len(all_train_cameras)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches + 1}")
    
    # 创建保存目录
    save_dir = os.path.join(dataset.model_path, f"eval_iter_{iteration}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}\n")
    
    # 创建PLY保存目录
    ply_save_dir = os.path.join(save_dir, "deformed_gaussians")
    os.makedirs(ply_save_dir, exist_ok=True)
    
    saved_kids = set()
    cameras_by_kid = {}
    total_processed = 0
    
    # 分批处理
    for batch_idx in range(num_batches + 1):
        batch_start = batch_idx * batch_size
        if batch_idx == num_batches:
            batch_end = len(all_train_cameras)
        else:
            batch_end = batch_start + batch_size
        
        if batch_start == batch_end:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx + 1}/{num_batches + 1}")
        print(f"Cameras {batch_start} to {batch_end - 1} ({batch_end - batch_start} views)")
        print(f"{'='*60}\n")
        
        # 加载当前batch的训练相机
        batch_cameras_info = all_train_cameras[batch_start:batch_end]
        scene.loadTrainCameras(batch_cameras_info, dataset.rscale)
        
        # 初始化变形图（只在第一个batch时）
        if batch_idx == 0 and gaussians.dg is None:
            dg_path = os.path.join(dataset.source_path, 'deformation_graph.json')
            if os.path.exists(dg_path):
                print(f"Loading deformation graph from: {dg_path}")
                gaussians.deform_init(dg_path)
            else:
                print(f"Warning: Deformation graph not found at {dg_path}")
        
        # 获取当前batch的相机
        cameras = scene.getTrainCameras(dataset.rscale)
        
        # 收集kid分布信息
        for viewpoint in cameras:
            kid = viewpoint.kid
            if kid not in cameras_by_kid:
                cameras_by_kid[kid] = 0
            cameras_by_kid[kid] += 1
        
        # 遍历当前batch的所有训练视角
        for idx, viewpoint in enumerate(tqdm(cameras, desc=f"Batch {batch_idx + 1} evaluation")):
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
            
            # 保存指标到全局结果
            all_results['l1'].append(l1_val)
            all_results['psnr'].append(psnr_val)
            all_results['ssim'].append(ssim_val)
            
            if LPIPS_AVAILABLE:
                lpips_val = lpips(image, gt_image, net_type='vgg').mean().item()
                all_results['lpips'].append(lpips_val)
            
            # 保存渲染图像（每10张保存一次以节省空间）
            global_idx = total_processed + idx
            if global_idx % 10 == 0 or global_idx < 5:
                img_name = viewpoint.image_name
                torchvision.utils.save_image(image, os.path.join(save_dir, f"{img_name}_render.png"))
                if global_idx < 5:  # 只保存前5张ground truth
                    torchvision.utils.save_image(gt_image, os.path.join(save_dir, f"{img_name}_gt.png"))
        
        total_processed += len(cameras)
        
        # 清空当前batch的相机以释放显存
        scene.clearCameras(dataset.rscale)
        torch.cuda.empty_cache()
        
        print(f"\nBatch {batch_idx + 1} completed. Total processed: {total_processed}/{len(all_train_cameras)}")
    
    # 计算全局统计信息
    avg_l1 = np.mean(all_results['l1'])
    avg_psnr = np.mean(all_results['psnr'])
    avg_ssim = np.mean(all_results['ssim'])
    
    std_l1 = np.std(all_results['l1'])
    std_psnr = np.std(all_results['psnr'])
    std_ssim = np.std(all_results['ssim'])
    
    # 打印最终结果
    print(f"\n{'='*60}")
    print(f"Final Evaluation Results (Iteration {iteration}):")
    print(f"{'='*60}")
    print(f"  Total views processed: {total_processed}")
    print(f"  L1:    {avg_l1:.6f} ± {std_l1:.6f}")
    print(f"  PSNR:  {avg_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"  SSIM:  {avg_ssim:.6f} ± {std_ssim:.6f}")
    
    if LPIPS_AVAILABLE:
        avg_lpips = np.mean(all_results['lpips'])
        std_lpips = np.std(all_results['lpips'])
        print(f"  LPIPS: {avg_lpips:.6f} ± {std_lpips:.6f}")
    
    print(f"{'='*60}\n")
    
    # 保存详细结果
    all_results['avg'] = {
        'l1': float(avg_l1),
        'psnr': float(avg_psnr),
        'ssim': float(avg_ssim)
    }
    all_results['std'] = {
        'l1': float(std_l1),
        'psnr': float(std_psnr),
        'ssim': float(std_ssim)
    }
    
    if LPIPS_AVAILABLE:
        all_results['avg']['lpips'] = float(avg_lpips)
        all_results['std']['lpips'] = float(std_lpips)
    
    all_results['num_views'] = total_processed
    all_results['num_kids'] = len(cameras_by_kid)
    all_results['iteration'] = iteration
    all_results['saved_kids'] = sorted(list(saved_kids))
    all_results['batch_size'] = batch_size
    all_results['num_batches'] = num_batches + 1
    
    # 保存结果到JSON文件
    results_file = os.path.join(save_dir, f"metrics.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to: {results_file}")
    
    # 生成markdown格式的结果表格
    markdown_file = os.path.join(save_dir, f"metrics.md")
    with open(markdown_file, 'w') as f:
        f.write(f"# Evaluation Results - Iteration {iteration}\n\n")
        f.write(f"**Model Path:** `{dataset.model_path}`\n\n")
        f.write(f"**Number of views:** {total_processed}\n\n")
        f.write(f"**Number of keyframes (kids):** {len(cameras_by_kid)}\n\n")
        f.write(f"**Batch size:** {batch_size}\n\n")
        f.write(f"**Number of batches:** {num_batches + 1}\n\n")
        f.write(f"**Saved kids:** {sorted(list(saved_kids))}\n\n")
        
        f.write("## Metrics\n\n")
        f.write("| Metric | Mean | Std |\n")
        f.write("|--------|------|-----|\n")
        f.write(f"| L1 | {all_results['avg']['l1']:.6f} | {all_results['std']['l1']:.6f} |\n")
        f.write(f"| PSNR (dB) | {all_results['avg']['psnr']:.4f} | {all_results['std']['psnr']:.4f} |\n")
        f.write(f"| SSIM | {all_results['avg']['ssim']:.6f} | {all_results['std']['ssim']:.6f} |\n")
        if LPIPS_AVAILABLE:
            f.write(f"| LPIPS | {all_results['avg']['lpips']:.6f} | {all_results['std']['lpips']:.6f} |\n")
        
        # 添加每个kid的相机数量信息
        f.write("\n## Keyframe Distribution\n\n")
        f.write("| Kid | Number of Cameras |\n")
        f.write("|-----|------------------|\n")
        for kid, count in sorted(cameras_by_kid.items()):
            f.write(f"| {kid} | {count} |\n")
        
        # 添加保存的PLY文件列表
        f.write("\n## Saved Deformed Gaussian PLY Files\n\n")
        f.write(f"Location: `{ply_save_dir}`\n\n")
        for kid in sorted(list(saved_kids)):
            f.write(f"- `{kid:06d}.ply`\n")
    
    print(f"Markdown report saved to: {markdown_file}")
    print(f"\nSaved {len(saved_kids)} deformed Gaussian PLY files to: {ply_save_dir}")
    print(f"  Kids saved: {sorted(list(saved_kids))}")
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}\n")
    
    return all_results


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
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for evaluation (default: use batchnum from config)")
    
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
    results = evaluate_checkpoint(dataset, hyper, pipe, opt, args.start_checkpoint, args.eval_batch_size)
    
    if results is not None:
        print("\n✓ Evaluation completed successfully!")
    else:
        print("\n✗ Evaluation failed!")