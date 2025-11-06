#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_bribg as render
from gaussian_renderer import network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,ModelHiddenParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try: 
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

#SUMO

import cv2
import matplotlib.pyplot as plt
import time

def training(dataset, hyper,opt, pipe, saving_iterations, checkpoint_iterations, checkpoint,debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,hyper,opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    #不优化标准空间
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint,weights_only=False,map_location="cuda")
        gaussians.restore(model_params,opt)
    

    all_train_cameras = scene.getAvailableCamInfos()['train_cameras']
    num_batches = dataset.batchnum-1
    batch_size = max(1, len(all_train_cameras) // num_batches)
    max_cycles = dataset.looptimes

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 引入全局iteration计数器
    global_iteration = first_iter

    for cycle in range(max_cycles):
        print(f"\n=== Starting cycle {cycle + 1}/{max_cycles} ===")
        
        for batch_idx in range(num_batches+1):
            batch_start = batch_idx * batch_size
            if batch_idx == num_batches:
                batch_end = len(all_train_cameras)
            else:
                batch_end = batch_start + batch_size
            
            if batch_start==batch_end:
                continue
            print(f"Loading batch {batch_idx + 1}/{num_batches}: cameras {batch_start} to {batch_end-1}")
            scene.loadTrainCameras(all_train_cameras[batch_start:batch_end], dataset.rscale)
            scene.loadTestCameras(scene.getAvailableCamInfos()['test_cameras'][batch_start:batch_end],dataset.rscale)

            if gaussians.dg is None:
                dg_path=os.path.join(dataset.source_path,'deformation_graph.json')
                gaussians.deform_init(dg_path)

            viewpoint_stack = scene.getTrainCameras(dataset.rscale).copy()

            #==================
            # for viewpoint in viewpoint_stack:
            #     gaussians.export_deformed_gaussian(viewpoint)
            #==================

            viewpoint_indices = list(range(len(viewpoint_stack)))
            ema_loss_for_log = 0.0
            ema_Ll1depth_for_log = 0.0

            # 计算当前batch的局部iteration范围
            batch_start_iter = global_iteration + 1
            batch_end_iter = global_iteration + opt.iterations*(batch_end-batch_start)//batch_size
            
            progress_bar = tqdm(range(batch_start_iter, batch_end_iter + 1), desc=f"Cycle{cycle+1} Batch{batch_idx+1}")
            
            for iteration in range(batch_start_iter, batch_end_iter + 1):
                # 更新全局iteration计数器
                global_iteration = iteration
                
                bg = torch.rand((3), device="cuda") if opt.random_background else background
                
                    
                if network_gui.conn == None:
                    network_gui.try_connect()
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                        if custom_cam != None:
                            net_image = render(custom_cam, gaussians, pipe, bg, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp)["render"]
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        network_gui.send(net_image_bytes, dataset.source_path)
                        if do_training and ((iteration < batch_end_iter) or not keep_alive):
                            break
                    except Exception as e:
                        network_gui.conn = None

                local_iteration = iteration - batch_start_iter + 1
                gaussians.update_learning_rate(local_iteration)

                if local_iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras(dataset.rscale).copy()
                    viewpoint_indices = list(range(len(viewpoint_stack)))
                
                rand_idx = randint(0, len(viewpoint_indices) - 1)
                viewpoint_cam = viewpoint_stack.pop(rand_idx)
                vind = viewpoint_indices.pop(rand_idx)
                
                if (local_iteration - 1) == debug_from:
                    pipe.debug = True

                
                #SUMO
                if dataset.use_background_image and os.path.exists(viewpoint_cam.bg_path):
                    bg=scene.get_background_image(viewpoint_cam)

                #SUMO
                gaussians.update_deformed_gaussians(viewpoint_cam.deformer_path,viewpoint_cam.timecode)


                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


                #保存缓冲
                if iteration % 500 == 1:
                    image_to_save =image.permute(1, 2, 0).detach().cpu().numpy()
                    # image_to_save = torch.clamp(image_to_save, 0, 1)
                    try:
                        os.makedirs(f"{dataset.model_path}/render_image/",exist_ok=True)
                        plt.imsave(f"{dataset.model_path}/render_image/rendered-image_{iteration-1}_{viewpoint_cam.image_name}.png", image_to_save)
                    except Exception as e:
                        print(e)

                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    image *= alpha_mask

                gt_image = viewpoint_cam.original_image.cuda()

                Ll1 = l1_loss(image, gt_image)
                if FUSED_SSIM_AVAILABLE:
                    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                else:
                    ssim_value = ssim(image, gt_image)

                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

                Ll1depth_pure = 0.0
                if depth_l1_weight(local_iteration) > 0 and viewpoint_cam.depth_reliable:
                    invDepth = render_pkg["depth"]
                    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    depth_mask = viewpoint_cam.depth_mask.cuda()

                    Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                    Ll1depth = depth_l1_weight(local_iteration) * Ll1depth_pure 
                    loss += Ll1depth
                    Ll1depth = Ll1depth.item()
                else:
                    Ll1depth = 0

                #SUMO
                tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
                loss += tv_loss


                loss.backward()


                with torch.no_grad():
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log


                    if local_iteration % 10 == 0:
                        progress_bar.set_postfix({
                            "Loss": f"{ema_loss_for_log:.{7}f}", 
                            # "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                            "Global Iter": global_iteration
                        })
                        progress_bar.update(10)
                    if local_iteration == opt.iterations:
                        progress_bar.close()

                    # 使用全局iteration进行报告和保存，但传递局部iteration给需要它的函数
                    # training_report(tb_writer, global_iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
                    
                    # Densification
                    if global_iteration < opt.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if global_iteration > opt.densify_from_iter and global_iteration % opt.densification_interval == 0:# and gaussians.get_xyz.shape[0]<360000:
                            size_threshold = 20 if global_iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                            # current_point_count = gaussians._xyz.shape[0]
                            # if hasattr(gaussians, 'vertex_deformer') and gaussians.vertex_deformer:
                            #     keys_to_remove = []
                            #     for cached_kid, cached_data in gaussians.vertex_deformer.items():
                            #         # 检查缓存的点数是否与当前点数匹配
                            #         # cached_data.shape[0] 是旧的点数，current_point_count 是新的点数
                            #         if cached_data.shape[0] != current_point_count:
                            #             keys_to_remove.append(cached_kid)
                                
                            #     if keys_to_remove:
                            #         print(f'  🔧 清理{len(keys_to_remove)}个不匹配的变形缓存（稠密化后点数变化）')
                            #         for kid_to_remove in keys_to_remove:
                            #             if kid_to_remove in gaussians.vertex_deformer:
                            #                 del gaussians.vertex_deformer[kid_to_remove]
                            #             if hasattr(gaussians, 'deformed_gaussian_xyz') and kid_to_remove in gaussians.deformed_gaussian_xyz:
                            #                 del gaussians.deformed_gaussian_xyz[kid_to_remove]
                            #             if hasattr(gaussians, 'deformed_gaussian_rot') and kid_to_remove in gaussians.deformed_gaussian_rot:
                            #                 del gaussians.deformed_gaussian_rot[kid_to_remove]
                            #         print(f'  ✅ 缓存已清理，将在下次使用时重建（只重建一次）')

                            print(f"Gaussian splatting points: {gaussians._xyz.shape[0]}")
                        
                        if global_iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and global_iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()
                    


                    # 使用全局iteration检查是否需要保存，避免文件覆盖
                    if (global_iteration in saving_iterations):
                        print(f"\n[GLOBAL ITER {global_iteration}] Saving Gaussians (Cycle {cycle+1}, Batch {batch_idx+1}, Local Iter {local_iteration})")
                        scene.save(global_iteration+viewpoint_cam.kid)

                    if local_iteration < opt.iterations:
                        gaussians.exposure_optimizer.step()
                        gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                        if use_sparse_adam:
                            visible = radii > 0
                            gaussians.optimizer.step(visible, radii.shape[0])
                            gaussians.optimizer.zero_grad(set_to_none = True)
                        else:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none = True)


                            
                    # 使用全局iteration进行检查点保存
                    if (global_iteration in checkpoint_iterations):
                        print(f"\n[GLOBAL ITER {global_iteration}] Saving Checkpoint (Cycle {cycle+1}, Batch {batch_idx+1}, Local Iter {local_iteration})")
                        torch.save((gaussians.capture(), global_iteration), scene.model_path + "/chkpnt" + str(global_iteration) + ".pth")

                    # print(f'G {time.time()-st}')
            
            gaussians.update_deformed_gaussians(viewpoint_cam.deformer_path,viewpoint_cam.timecode)
            scene.save(global_iteration+viewpoint_cam.kid)
            scene.clearCameras(dataset.rscale)
            torch.save((gaussians.capture(), global_iteration), scene.model_path + "/chkpnt" + str(global_iteration) + ".pth")
            

        #在每个cycle结束时保存一次模型 
        print(f"\n[GLOBAL ITER {global_iteration}] Saving Gaussians at the end of cycle {cycle+1}")
        torch.save((gaussians.capture(), global_iteration), scene.model_path + "/chkpnt" + str(global_iteration) + ".pth")
            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #SUMO
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(scene.scale)}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras(scene.scale)[idx % len(scene.getTrainCameras(scene.scale))] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    #SUMO
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[700,3_000,7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None) 

    #SUMO
    parser.add_argument("--deformation_graph",type=str,default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)


    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args),op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,args.debug_from)

    print("\nTraining complete.")