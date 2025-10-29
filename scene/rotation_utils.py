import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
    
def rotation_6d_to_quaternion(d6: torch.Tensor) -> torch.Tensor:
    """
    6D连续旋转 -> 四元数 (一步到位)
    
    参数:
        d6: (..., 6) 6D旋转表示
        
    返回:
        q: (..., 4) 四元数 [w, x, y, z]
    """
    R = rotation_6d_to_matrix(d6)
    q = rotation_matrix_to_quaternion(R)
    return q


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    将6D连续旋转表示转换为旋转矩阵
    
    参数:
        d6: (..., 6) 形状的张量，表示两个3D向量 [a1, a2]
        
    返回:
        R: (..., 3, 3) 旋转矩阵
        
    原理:
        1. 前3个数是向量a1，后3个数是向量a2
        2. 用Gram-Schmidt正交化得到正交基 [b1, b2, b3]
        3. 组合成旋转矩阵
    
    优点:
        - 连续、无奇异点
        - 不需要归一化
        - 梯度友好
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    
    # Gram-Schmidt正交化
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    # 组合成旋转矩阵 (..., 3, 3)
    R = torch.stack([b1, b2, b3], dim=-1)
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为四元数 (稳定版本)
    
    参数:
        R: (..., 3, 3) 旋转矩阵
        
    返回:
        q: (..., 4) 四元数 [w, x, y, z]
    """
    # 保存原始形状
    original_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    batch_size = R_flat.size(0)
    
    q = torch.zeros((batch_size, 4), device=R.device, dtype=R.dtype)
    
    # 计算矩阵的迹
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    
    # 情况1: trace > 0 (最稳定)
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s
        q[mask1, 2] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s
        q[mask1, 3] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s
    
    # 情况2: R[0,0] 最大
    mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2
        q[mask2, 0] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s
        q[mask2, 3] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s
    
    # 情况3: R[1,1] 最大
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2
        q[mask3, 0] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s
        q[mask3, 1] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s
    
    # 情况4: R[2,2] 最大
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2
        q[mask4, 0] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s
        q[mask4, 1] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s
        q[mask4, 2] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s
    
    # 恢复原始形状
    q = q.reshape(*original_shape, 4)
    
    # 归一化（保险起见）
    q = F.normalize(q, p=2, dim=-1)
    
    return q