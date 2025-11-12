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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from deformation_tool.deformation_utils import get_deformation_info_fixed_influences
from utils.general_utils import build_quaternion
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

#SUMO
from deformation_tool import DeformationGraph,DeformationTransforms,apply_deformation_to_gaussians2,apply_deformation_to_gaussians_full
from deformation_tool.transform_gaussian import apply_deformation_to_gaussians_torch_batched
from scene.deformation import Deformation,quat_multiply
from scene.rotation_utils import rotation_6d_to_quaternion,quaternion_to_rotation_6d

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, args,optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        #SUMO
        self.args=args
        self.influ_nums=5
        self._deformation=Deformation()
        self.deformed_xyz=torch.empty(0)
        self.deformed_rot=torch.empty(0)
        self.deformed_scl=torch.empty(0)
        self.deformed_opa=torch.empty(0)
        self.deformed_shs=torch.empty(0)

        self.bg_image_dict={}
        self.deformed_gaussian_xyz={}
        self.deformed_gaussian_rot={}
        
        # ✅ 逆变换缓存
        self.inverse_deform_transforms={}
        
        self.dg=None
        self.base_xyz=None
        self.base_quat=None
        
        # ✅ 当前使用的deformer_path，用于densify时的逆变换
        self.current_deformer_path = None
        self.current_timecode=0
        # ✅ 新增：时间相关参数
        self._time_center = torch.empty(0)      # μₜ: 高斯出现的中心时间
        self._time_duration = torch.empty(0)    # s: 持续时间/带宽
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._time_center,      # ✅ 新增
            self._time_duration,    # ✅ 新增
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.dg,
            self.base_xyz,
            self.base_quat,
            self._deformation.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._time_center,       # ✅ 新增
        self._time_duration,     # ✅ 新增
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.dg,
        self.base_xyz,
        self.base_quat,
        deform_state) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self._deformation.load_state_dict(deform_state)
        
        # ✅ 恢复后清空缓存
        self._clear_all_caches()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_render_scaling(self):
        return self.scaling_activation(self.deformed_scl)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_render_rotation(self):
        return self.rotation_activation(self.deformed_rot)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_render_xyz(self):
        return self.deformed_xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_render_opacity(self):
        return self.opacity_activation(self.deformed_opa)
    
    @property
    def get_exposure(self):
        return self._exposure

    # ✅ 新增：时间参数的property
    @property
    def get_time_center(self):
        """获取时间中心（归一化到 [0, 1]）"""
        return torch.sigmoid(self._time_center)
    
    @property
    def get_time_duration(self):
        """获取时间持续时间（确保为正）"""
        return torch.exp(self._time_duration)

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # ✅ 新增：初始化时间参数
        # 默认所有点在整个时间段内可见
        num_points = fused_point_cloud.shape[0]
        # 使用 sigmoid 的逆函数初始化，使得 sigmoid(0) = 0.5
        time_centers = torch.zeros((num_points, 1), dtype=torch.float, device="cuda")  # sigmoid(0) = 0.5
        # 使用 log 的逆函数初始化，使得 exp(log(3.0)) = 3.0 (大带宽=始终可见)
        time_durations = torch.log(torch.tensor(3.0)) * torch.ones((num_points, 1), dtype=torch.float, device="cuda")
        
        self._time_center = nn.Parameter(time_centers.requires_grad_(True))
        self._time_duration = nn.Parameter(time_durations.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        self._deformation = self._deformation.to("cuda") 

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # ✅ 新增：时间参数的优化器组
            {'params': [self._time_center], 'lr': training_args.position_lr_init * self.spatial_lr_scale * 0.1, "name": "time_center"},
            {'params': [self._time_duration], 'lr': training_args.position_lr_init * self.spatial_lr_scale * 0.05, "name": "time_duration"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)  

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr_xyz = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr_xyz
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
        
        return lr_xyz

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # ✅ 新增：时间参数
        l.append('time_center')
        l.append('time_duration')
        return l


    # def save_ply(self, path):
    #     mkdir_p(os.path.dirname(path))

    #     xyz = self.deformed_xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self.deformed_opa.detach().cpu().numpy()
    #     scale = self.deformed_scl.detach().cpu().numpy()
    #     rotation = self.deformed_rot.detach().cpu().numpy()

    #     time_center = self.get_time_center.detach().cpu().numpy()
    #     time_duration = self.get_time_duration.detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, 
    #                                 time_center, time_duration), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)
    

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.deformed_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        # ✅ 从 deformed_shs 提取特征
        deformed_shs_np = self.deformed_shs.detach()

        # 分离 DC 部分 (第0个SH系数)
        f_dc = deformed_shs_np[:, 0:1, :]  # [N, 1, 3]

        # 分离 rest 部分 (剩余的SH系数)
        f_rest = deformed_shs_np[:, 1:, :]  # [N, 15, 3]

        f_dc=f_dc.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = f_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # # ✅ 使用 NumPy 的 reshape 代替 flatten(start_dim=1)
        # # 从 [N, 1, 3] -> [N, 3, 1] -> [N, 3]
        # f_dc = f_dc.transpose(0, 2, 1).reshape(f_dc.shape[0], -1)

        # # 从 [N, 15, 3] -> [N, 3, 15] -> [N, 45]
        # f_rest = f_rest.transpose(0, 2, 1).reshape(f_rest.shape[0], -1)

        opacities = self.deformed_opa.detach().cpu().numpy()
        scale = self.deformed_scl.detach().cpu().numpy()
        rotation = self.deformed_rot.detach().cpu().numpy()
        
        # ✅ 新增：保存时间参数（使用激活后的值）
        time_center = self.get_time_center.detach().cpu().numpy()
        time_duration = self.get_time_duration.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # ✅ 修改：添加时间参数到attributes
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, 
                                    time_center, time_duration), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # ✅ 新增：加载时间参数（如果存在）
        try:
            time_centers = np.asarray(plydata.elements[0]["time_center"])[..., np.newaxis]
            time_durations = np.asarray(plydata.elements[0]["time_duration"])[..., np.newaxis]
            has_time_params = True
            print("Time parameters loaded from ply file")
        except:
            # 如果文件中没有时间参数，使用默认值
            print("No time parameters found in ply file, using defaults")
            num_points = xyz.shape[0]
            time_centers = np.ones((num_points, 1)) * 0.5
            time_durations = np.ones((num_points, 1)) * 3.0
            has_time_params = False

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # ✅ 新增：加载时间参数
        # 使用逆激活函数（sigmoid的逆和log）
        if has_time_params:
            self._time_center = nn.Parameter(
                inverse_sigmoid(torch.tensor(time_centers, dtype=torch.float, device="cuda")).requires_grad_(True)
            )
            self._time_duration = nn.Parameter(
                torch.log(torch.tensor(time_durations, dtype=torch.float, device="cuda")).requires_grad_(True)
            )
        else:
            # 默认值：sigmoid(0)=0.5, exp(log(3.0))=3.0
            self._time_center = nn.Parameter(
                torch.zeros((xyz.shape[0], 1), dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._time_duration = nn.Parameter(
                torch.log(torch.tensor(3.0)) * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda").requires_grad_(True)
            )

        self.active_sh_degree = self.max_sh_degree

        self.base_xyz=self._xyz.detach().clone()
        self.base_quat=self._rotation.detach().clone()

    def fixup_params(self, cam_infos, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        self.tmp_radii = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation = self._deformation.to("cuda")
        
        # ✅ 新增：如果时间参数未初始化，使用默认值
        if self._time_center.numel() == 0:
            num_points = self.get_xyz.shape[0]
            self._time_center = nn.Parameter(
                torch.zeros((num_points, 1), dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._time_duration = nn.Parameter(
                torch.log(torch.tensor(3.0)) * torch.ones((num_points, 1), dtype=torch.float, device="cuda").requires_grad_(True)
            )
            print("Initialized time parameters with defaults")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group.get("name", "") == "deformation":
                continue
            if group.get("name", "") == "grid":
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._time_center = optimizable_tensors["time_center"]        # ✅ 新增
        self._time_duration = optimizable_tensors["time_duration"]    # ✅ 新增

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

        # ✅ 更新base并清空缓存
        self.base_xyz=self.base_xyz[valid_points_mask]
        self.base_quat=self.base_quat[valid_points_mask]
        self._clear_all_caches()
        # self._update_base_and_clear_cache()
        self.update_deformed_gaussians_for_render()
    
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group.get("name", "") == "deformation":
                continue
            if group.get("name", "") == "grid":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, 
                             new_scaling, new_rotation, new_tmp_radii,
                             new_time_center, new_time_duration):  # ✅ 新增参数
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "time_center": new_time_center,      # ✅ 新增
            "time_duration": new_time_duration   # ✅ 新增
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._time_center = optimizable_tensors["time_center"]        # ✅ 新增
        self._time_duration = optimizable_tensors["time_duration"]    # ✅ 新增

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # ✅ 更新base并清空缓存
        
        self.base_xyz=torch.cat((self.base_xyz,new_xyz),dim=0)
        self.base_quat=torch.cat((self.base_quat,new_rotation),dim=0)
        self._clear_all_caches()
        # self._update_base_and_clear_cache()
        self.update_deformed_gaussians_for_render()

    # ✅ 新增：清空所有缓存的统一方法
    def _clear_all_caches(self):
        """清空所有变形相关的缓存"""
        self.deformed_gaussian_xyz.clear()
        self.deformed_gaussian_rot.clear()
        # self.inverse_deform_transforms.clear()
        # self.current_deformer_path = None
        # print(f"[Cache Clear] Cleared all deformation caches")

    # ✅ 新增：统一的更新和清理函数
    def _update_base_and_clear_cache(self):
        """更新base_xyz/base_quat并清空所有变形缓存"""
        self.base_xyz = self._xyz.detach().clone()
        self.base_quat = self._rotation.detach().clone()
        
        # 清空所有缓存
        self._clear_all_caches()
        
        # print(f"[Cache Clear] Updated base_xyz to {self.base_xyz.shape[0]} points and cleared all caches")

    # ✅ 修改：densify_and_split 使用缓存的逆变换
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_render_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 在变形空间采样新点
        stds = self.get_render_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz_deformed = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_rotation_deformed = self._rotation[selected_pts_mask].repeat(N,1)
        # ✅ 关键修改：使用已缓存的逆变换
        if self.current_deformer_path is not None and self.current_deformer_path in self.inverse_deform_transforms:
            try:
                inv_transform = self.inverse_deform_transforms[self.current_deformer_path]

                new_xyz,_=apply_deformation_to_gaussians_torch_batched(self.dg,new_xyz_deformed,new_rotation_deformed,inv_transform)

                # new_xyz_canonical = apply_deformation_to_gaussians2(
                #     self.dg, 
                #     new_xyz_deformed.detach().cpu().numpy(), 
                #     inv_transform
                # )
                # new_xyz = torch.as_tensor(new_xyz_canonical, dtype=torch.float, device="cuda")
                # print(f"[Densify Split] Applied cached inverse transform, new points: {new_xyz.shape[0]}")
            except Exception as e:
                # print(f"[Warning] Inverse transform failed: {e}")
                # print(f"[Warning] Falling back to deformed space (may cause inconsistency)")
                # new_xyz = new_xyz_deformed
                raise ValueError(f"Inverse transform failed: {e}")
        else:
            print(f"[Warning] No cached inverse transform available for {self.current_deformer_path}")
            print(f"[Warning] Using deformed xyz directly (may cause inconsistency)")
            new_xyz = new_xyz_deformed

        #极限测试，无变形图模拟 TODO 恢复
        # new_xyz = new_xyz_deformed

        
        # 其他属性直接从canonical空间复制
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        
        # ✅ 新增：继承父点的时间参数
        new_time_center = self._time_center[selected_pts_mask].repeat(N, 1)
        new_time_duration = self._time_duration[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, 
                                  new_scaling, new_rotation, new_tmp_radii,
                                  new_time_center, new_time_duration)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_render_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # Clone操作直接从canonical空间复制，无需逆变换
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        
        # ✅ 新增：复制时间参数
        new_time_center = self._time_center[selected_pts_mask]
        new_time_duration = self._time_duration[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, 
                                  new_scaling, new_rotation, new_tmp_radii,
                                  new_time_center, new_time_duration)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        min_time_duration=0.01
        max_time_duration=3.5

        time_center = self.get_time_center.squeeze()  # (N,)
        time_duration = self.get_time_duration.squeeze()  # (N,)
        
        # 条件A: 时间中心超出有效范围 [0, 1]（允许小幅越界）
        time_out_of_range = (time_center < -0.15) | (time_center > 1.15)
        
        # 条件B: 时间持续时间过短（几乎瞬时出现，可能是噪声）
        time_too_short = time_duration < min_time_duration
        
        # 条件C: 时间持续时间过长（接近静态，可能是冗余点）
        time_too_long = time_duration > max_time_duration
        
        # 合并所有时间相关的剪枝条件
        time_prune_mask = time_out_of_range | time_too_short | time_too_long 
        
        # 3. 综合所有剪枝条件
        # prune_mask = torch.logical_or(prune_mask, time_prune_mask)

        self.prune_points(prune_mask)
        # print(f"[Densify] Gaussian after prune: {self._xyz.shape[0]}")

        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def set_base_xyz(self):

        self.base_xyz = self._xyz.detach().clone()
        self.base_quat = self._rotation.detach().clone()
        self._clear_all_caches()

    # ✅ 新增：时间不透明度计算函数
    def compute_temporal_opacity(self, t):
        """
        计算时间调制的不透明度 (论文公式4)
        σ(t) = exp(-0.5 * ((t - μₜ) / s)²)
        
        Args:
            t: 当前归一化时间 [0, 1]
        
        Returns:
            time_mask: (N, 1) 时间不透明度
        """
        # 确保 t 是 tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=self._time_center.device)
        
        # 获取激活后的时间参数
        time_center = self.get_time_center  # (N, 1) in [0, 1]
        time_duration = self.get_time_duration  # (N, 1), positive
        
        # 计算时间差异
        time_diff = t - time_center  # (N, 1)
        
        # 计算高斯衰减
        normalized_diff = time_diff / time_duration
        time_mask = torch.exp(-0.5 * normalized_diff ** 2)
        
        return time_mask

    # ✅ 修改：update_deformed_gaussians 记录当前deformer_path并应用时间调制
    def update_deformed_gaussians(self, deformer_path, t):
        """
        Args:
            deformer_path: 变形路径
            t: 归一化时间 [0, 1]
        """
        # 记录当前使用的deformer_path，用于densify时的逆变换
        self.current_deformer_path = deformer_path
        self.current_timecode=t
        
        time = torch.tensor(t).to(self._xyz.device).repeat(self._xyz.shape[0], 1)
        temp_xyz, temp_rot = self.get_deformed_gaussians(deformer_path)
        
        dx, ds, dr, do, dshs = self._deformation(temp_xyz.detach(), time)
        
        self.deformed_xyz = temp_xyz + dx
        self.deformed_scl = self._scaling + ds
        self.deformed_rot = rotation_6d_to_quaternion(temp_rot + dr)
        
        # ✅ 关键修改：应用时间调制
        # 先计算空间变形后的不透明度（在logit空间）
        if self.args.no_do:
            base_opacity_logit=self._opacity
        else:
            base_opacity_logit = self._opacity + do
        
        # 计算时间mask
        time_mask = self.compute_temporal_opacity(t)
        
        # 最终不透明度 = 空间不透明度 × 时间mask
        # 在logit空间：logit(p1 * p2) = logit(p1) + log(p2)
        # 为了数值稳定，添加小的epsilon
        #timer
        # self.deformed_opa = base_opacity_logit + torch.log(time_mask + 1e-8)
        self.deformed_opa = base_opacity_logit

        if self.args.no_dshs:
            self.deformed_shs=self.get_features
        else:
            self.deformed_shs = self.get_features + dshs
    
    def update_deformed_gaussians_for_render(self):
        """
        Args:
            deformer_path: 变形路径
            t: 归一化时间 [0, 1]
        """

        
        time = torch.tensor(self.current_timecode).to(self._xyz.device).repeat(self._xyz.shape[0], 1)
        temp_xyz, temp_rot = self.get_deformed_gaussians(self.current_deformer_path)
        
        dx, ds, dr, do, dshs = self._deformation(temp_xyz.detach(), time.detach())
        
        self.deformed_xyz = temp_xyz.detach() + dx.detach()
        self.deformed_scl = self._scaling.detach() + ds.detach()
        self.deformed_rot = rotation_6d_to_quaternion(temp_rot.detach() + dr.detach())
        
        # ✅ 关键修改：应用时间调制
        # 先计算空间变形后的不透明度（在logit空间）
        if self.args.no_do:
            base_opacity_logit=self._opacity.detach()
        else:
            base_opacity_logit = self._opacity.detach() + do.detach()
        
        # 计算时间mask
        time_mask = self.compute_temporal_opacity(self.current_timecode)
        
        # 最终不透明度 = 空间不透明度 × 时间mask
        # 在logit空间：logit(p1 * p2) = logit(p1) + log(p2)
        # 为了数值稳定，添加小的epsilon
        #timer
        # self.deformed_opa = base_opacity_logit + torch.log(time_mask + 1e-8)

        self.deformed_opa = base_opacity_logit
        if self.args.no_dshs:
            self.deformed_shs=self.get_features.detach()
        else:
            self.deformed_shs = self.get_features.detach() + dshs.detach()
    
    
    def deform_init(self, dg_path):
        self.dg_path = dg_path
        self.dg = DeformationGraph()
        self.dg.load(dg_path)

    # ✅ 关键修改：在get_deformed_gaussians中一次性加载逆变换
    def get_deformed_gaussians(self, deformer_path):
        if self.dg is None:
            raise NameError("self.dg not initialize!")
        if self.base_xyz is None:
            self.set_base_xyz()
            
        if deformer_path not in self.deformed_gaussian_xyz:
            # 加载正向变换
            transforms = DeformationTransforms()
            transforms.load(deformer_path)

            self.deformed_gaussian_xyz[deformer_path]=transforms
            #变形图
            # input_gs = {"xyz": self.base_xyz.cpu().numpy(), "rotations": self.base_quat.cpu().numpy()}
            # deformed_gaussian = apply_deformation_to_gaussians_full(self.dg, input_gs, transforms)
            # deformed_points = torch.as_tensor(deformed_gaussian["xyz"]).to(self._xyz.device)
            # deformed_rots = torch.as_tensor(deformed_gaussian["rotations"]).to(self._xyz.device)

            

            # deformed_rots = quaternion_to_rotation_6d(deformed_rots)

            # self.deformed_gaussian_xyz[deformer_path] = deformed_points
            # self.deformed_gaussian_rot[deformer_path] = deformed_rots

            #极限测试，无变形图模拟 TODO 恢复
            # self.deformed_gaussian_xyz[deformer_path] = self.base_xyz
            # self.deformed_gaussian_rot[deformer_path] = quaternion_to_rotation_6d(self.base_quat)
            

            # ✅ 关键修改：同时加载并缓存逆变换
            inv_trans_path = os.path.join(
                os.path.dirname(deformer_path), 
                "inv_transforms.json"
            )
            
            if os.path.exists(inv_trans_path):
                inv_transform = DeformationTransforms()
                inv_transform.load(inv_trans_path)
                self.inverse_deform_transforms[deformer_path] = inv_transform
                # print(f"[Deformation Cache] Cached forward & inverse transform for: {deformer_path}")
            else:
                print(f"[Warning] Inverse transform not found: {inv_trans_path}")
                print(f"[Warning] Densification may cause inconsistency for this frame")
        
        #有梯度无变形
        # return self._xyz, quaternion_to_rotation_6d(self._rotation)
        
        #有梯度变形场
        deformed_points,deformed_rots=apply_deformation_to_gaussians_torch_batched(self.dg,self._xyz,self._rotation,self.deformed_gaussian_xyz[deformer_path])
        return deformed_points, quaternion_to_rotation_6d(deformed_rots)
        
        #无梯度变形场
        # return self.deformed_gaussian_xyz[deformer_path], self.deformed_gaussian_rot[deformer_path]
    
    def compute_deformed_gaussian(self, deformer_path):
        transforms = DeformationTransforms()
        transforms.load(deformer_path)
        deformed_points = apply_deformation_to_gaussians2(self.dg, self._xyz.detach().clone().cpu().numpy(), transforms)
        deformed_points = torch.as_tensor(deformed_points).to(self._xyz.device)
        return deformed_points

    def save_ply_with_xyz(self, xyz, rots, path):
        mkdir_p(os.path.dirname(path))

        xyz = xyz.cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = rots.cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def export_deformed_gaussian(self, viewpoint):
        if self.dg is None:
            raise NameError("self.dg not initialize!")
        deformer_path = viewpoint.deformer_path
        transforms = DeformationTransforms()
        transforms.load(deformer_path)
        input_gs = {"xyz": self.base_xyz.cpu().numpy(), "rotations": self.base_quat.cpu().numpy()}
        deformed_gaussian = apply_deformation_to_gaussians_full(self.dg, input_gs, transforms)
        deformed_points = torch.as_tensor(deformed_gaussian["xyz"]).to(self._xyz.device)
        deformed_rots = torch.as_tensor(deformed_gaussian["rotations"]).to(self._xyz.device)
        self.deformed_gaussian_xyz[deformer_path] = deformed_points
        self.deformed_gaussian_rot[deformer_path] = deformed_rots
        
        self.save_ply_with_xyz(self.deformed_gaussian_xyz[deformer_path], self.deformed_gaussian_rot[deformer_path], deformer_path.replace("json","ply"))

    def _plane_regulation(self):
        multi_res_grids = self._deformation.grid.grids
        total = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _time_regulation(self):
        multi_res_grids = self._deformation.grid.grids
        total = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _l1_regulation(self):
        multi_res_grids = self._deformation.grid.grids
        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    
    # ✅ 新增：时间相关的正则化（可选）
    def _temporal_sparsity_regulation(self):
        """
        鼓励时间持续时间的稀疏性
        较小的duration意味着高斯只在特定时间段可见
        """
        # 负对数：鼓励较小的duration值
        return -torch.mean(torch.log(self.get_time_duration + 1e-8))
    
    def _temporal_smoothness_regulation(self):
        """
        鼓励相邻点的时间参数相似
        需要KNN信息，这里提供接口，实际使用时需要传入邻域信息
        """
        # 这是一个示例，实际使用需要根据空间邻域计算
        # 这里返回0，用户可以根据需要实现
        return torch.tensor(0.0, device=self._time_center.device)
    
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight,
                          temporal_sparsity_weight=0.0, temporal_smooth_weight=0.0):
        """
        Args:
            temporal_sparsity_weight: 时间稀疏性权重（鼓励短持续时间）
            temporal_smooth_weight: 时间平滑性权重（鼓励相邻点时间一致）
        """
        base_reg = (plane_tv_weight * self._plane_regulation() + 
                   time_smoothness_weight * self._time_regulation() + 
                   l1_time_planes_weight * self._l1_regulation())
        
        # ✅ 新增：时间正则化项
        temporal_reg = (temporal_sparsity_weight * self._temporal_sparsity_regulation() +
                       temporal_smooth_weight * self._temporal_smoothness_regulation())
        
        return base_reg + temporal_reg


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    first_difference = t[..., 1:, :] - t[..., :h-1, :]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]
    return torch.square(second_difference).mean()
