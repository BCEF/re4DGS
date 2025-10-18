import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init 
from .hexplane import HexPlaneField

# 配置参数
bounds=1.6
kplanes_config = {
                'grid_dimensions': 2,
                'input_coordinate_dim': 4,
                'output_coordinate_dim': 32,
                'resolution': [64, 64, 64, 25]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                }
multires = [1, 2, 4, 8] # multi resolution of voxel grid
        

def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


# 在文件开头添加工具函数
def quat_multiply(q1, q2):
    """Hamilton四元数乘法"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)



class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9):
        super(Deformation, self).__init__()
        # D是网络的深度，W是网络的宽度
        self.D = D
        self.W = W
        # input_ch是输入的维度
        # input_ch_time是时间的维度
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        
        # grid是一个HexPlaneField对象
        self.grid = HexPlaneField(bounds, kplanes_config, multires)
        
          
        # 创建网络 初始化
        self.create_net()
        self.init_weights()


    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self,xyz_max,xyz_min):
        self.grid.set_aabb(xyz_max, xyz_min)
    
    # 创建变形的神经网络   
    def create_net(self): 
        
        grid_out_dim = self.grid.feat_dim
        
    
        # 创建MLP网络
        self.feature_out = [nn.Linear(grid_out_dim ,self.W)]
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        
        # 创建多头输出层
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        
        
          
 
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def init_weights(self):
        """
        初始化原则：
        - 主干 feature_out: Kaiming-Normal (ReLU)
        - 各输出头：前几层 Kaiming，最后一层全 0（残差=0，等价于“不变形”）
        - grid(可学习 HexPlane): 建议小值或 0（不引入初始偏置）
        """

        # 2.1 初始化主干 MLP
        for m in self.feature_out.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        # 2.2 初始化各个“残差头”
        # 2.2 初始化各个“残差头”
        def init_head(seq: nn.Sequential, mode: str = "residual"):
            """
            mode:
            - "residual": 残差头，最后一层全 0
            - "absolute_quat": 绝对四元数头，最后一层小权重 + bias=[1,0,0,0]
            - "absolute_vec": 绝对向量头，最后一层小权重 + bias=0
            """
            # 先把所有 Linear 做 Kaiming + bias=0
            linears = [m for m in seq.modules() if isinstance(m, nn.Linear)]
            for m in linears:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

            # 处理最后一层
            last = linears[-1]
            if mode == "residual":
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

            elif mode == "absolute_quat":
                # 仅用于 rotations_deform 这类四元数输出
                nn.init.normal_(last.weight, mean=0, std=1e-4)  # 改用小方差的正态分布 
                last.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 单位四元数

            elif mode == "absolute_vec":
                nn.init.constant_(last.weight, 1e-3)
                nn.init.zeros_(last.bias)
            
            elif mode == "6d_rotation": 
                for m in self.mlp:
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                        nn.init.constant_(m.bias, 0.0)
                # 让最后一层更“保守”，靠近身份旋转（6D 全 0 经正交化趋近单位）
                last = self.mlp[-1]
                nn.init.uniform_(last.weight, -1e-3, 1e-3)
                nn.init.constant_(last.bias, 0.0)

            elif mode == "axis_angle":
                # 关键：用更大的标准差，让初始旋转分散
                nn.init.normal_(m.weight, mean=0, std=0.1)  # 从 0.01 改为 0.1
                nn.init.uniform_(m.bias, -0.1, 0.1)  # 随机偏置

            elif mode == "quat_identity":  # 新增模式
                nn.init.zeros_(last.weight)
                last.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0])

            init_head(self.pos_deform,    mode="residual")
            init_head(self.scales_deform, mode="absolute_vec")
            init_head(self.rotations_deform, mode="quat_identity")  # ← 只改这个
            init_head(self.opacity_deform, mode="residual")
            init_head(self.shs_deform,    mode="residual")





    # 进行hexplane查询
    def query_time(self, rays_pts_emb, time_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
        hidden = self.feature_out(grid_feature)   

        return hidden
     
    
    def forward(self, rays_pts_emb, time_emb=None):
        #if time_emb is None:
        #    return self.forward_static(rays_pts_emb[:,:3])
        #else:
        #    return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

        return self.forward_dynamic(rays_pts_emb, time_emb)



    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    


    def forward_dynamic(self,pts_emb,  time_emb):

        hidden = self.query_time(pts_emb, time_emb)
         
         
        # breakpoint()
        dx = self.pos_deform(hidden)
        #pts = torch.zeros_like(pts_emb[:,:3])
        #pts = pts_emb[:,:3]*mask + dx

        
        dr = self.rotations_deform(hidden)
        #rotations = torch.zeros_like(rotations_emb[:,:4])
        #rotations = rotations_emb[:,:4] + dr


        
        ds = self.scales_deform(hidden)
        #ds = 1.0 * torch.tanh(ds_raw)   # 残差范围 (-1, 1)

        #alpha = 0.25 #比较稳
        #ds = alpha * torch.tanh(ds_raw) # 让变形尺度更平滑一些

        #scales = torch.zeros_like(scales_emb[:,:3])
        #scales = scales_emb[:,:3]*mask + ds
        
        
        do = self.opacity_deform(hidden) 
        #do = 0.5 * torch.tanh(do_raw)   # 残差范围 (-0.5, 0.5)

        #opacity = torch.zeros_like(opacity_emb[:,:1])
        #opacity = opacity_emb[:,:1]*mask + do
        

        dshs = self.shs_deform(hidden).reshape([pts_emb.shape[0],16,3])
        # shs = torch.zeros_like(shs_emb)
        #shs = shs_emb*mask.unsqueeze(-1) + dshs

        return dx, ds, dr, do, dshs
    


    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    

 