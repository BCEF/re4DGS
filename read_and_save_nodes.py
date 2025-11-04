import json
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import os


def read_positions_from_json(json_path):
    """
    从deformation_graph.json文件中读取node_positions作为xyz坐标
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        xyz: numpy数组，shape为(N, 3)，包含所有node_positions坐标
    """
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取node_positions
    positions = data['node_positions']
    
    # 转换为numpy数组
    xyz = np.array(positions, dtype=np.float32)
    
    print(f"成功读取 {xyz.shape[0]} 个position点")
    print(f"节点数量: {data.get('node_count', 'N/A')}")
    print(f"顶点数量: {data.get('vertex_count', 'N/A')}")
    print(f"节点半径: {data.get('node_radius', 'N/A')}")
    print(f"\n坐标范围:")
    print(f"  X: [{xyz[:,0].min():.4f}, {xyz[:,0].max():.4f}]")
    print(f"  Y: [{xyz[:,1].min():.4f}, {xyz[:,1].max():.4f}]")
    print(f"  Z: [{xyz[:,2].min():.4f}, {xyz[:,2].max():.4f}]")
    
    return xyz


def read_positions_as_tensor(json_path, device='cuda'):
    """
    读取position并返回torch tensor
    
    Args:
        json_path: JSON文件路径
        device: 'cuda' 或 'cpu'
        
    Returns:
        xyz: torch tensor，shape为(N, 3)
    """
    xyz_np = read_positions_from_json(json_path)
    xyz_tensor = torch.from_numpy(xyz_np).float().to(device)
    return xyz_tensor


def inverse_sigmoid(x):
    """逆sigmoid函数"""
    return torch.log(x / (1 - x))


def save_ply_with_xyz(xyz, path, fixed_scale=0.00005, max_sh_degree=0):
    """
    保存xyz坐标为红色的高斯点云PLY文件
    
    Args:
        xyz: 点云坐标，shape为(N, 3)的tensor或numpy数组
        path: 保存路径
        fixed_scale: 固定的scale值，默认0.01
        max_sh_degree: 球谐函数最大阶数，默认0
    """
    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 转换为numpy数组
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    
    # 初始化normals（法向量）
    normals = np.zeros_like(xyz)
    
    # 使用固定的scale值
    scales = np.log(np.full((xyz.shape[0], 3), fixed_scale, dtype=np.float32))
    print(f"使用固定scale值: {fixed_scale}")
    
    # 初始化rotations（四元数旋转）
    rots = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    rots[:, 0] = 1  # w=1, x=y=z=0 表示无旋转
    
    # 初始化opacities（不透明度）
    opacities_tensor = inverse_sigmoid(0.9 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
    opacities = opacities_tensor.cpu().numpy()
    
    # 创建红色的颜色特征（球谐函数系数）
    SH_C0 = 0.28209479177387814  # sqrt(1/(4*pi))
    red_color = np.array([1.0, 0.0, 0.0])  # RGB红色
    fused_color = (red_color - 0.5) / SH_C0
    
    # 初始化球谐特征
    features_dc = np.tile(fused_color, (xyz.shape[0], 1)).reshape(xyz.shape[0], 3, 1)
    features_rest = np.zeros((xyz.shape[0], 3, (max_sh_degree + 1) ** 2 - 1), dtype=np.float32)
    
    # 展平球谐特征用于保存
    f_dc = features_dc.reshape(xyz.shape[0], 3)
    f_rest = features_rest.reshape(xyz.shape[0], -1)
    
    # 构造属性列表（标准3DGS格式）
    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # 添加球谐系数
    for i in range(3):
        attributes.append(f'f_dc_{i}')
    for i in range((max_sh_degree + 1) ** 2 - 1):
        attributes.append(f'f_rest_{i}')
    attributes.append('opacity')
    for i in range(3):
        attributes.append(f'scale_{i}')
    for i in range(4):
        attributes.append(f'rot_{i}')
    
    dtype_full = [(attribute, 'f4') for attribute in attributes]
    
    # 组合所有属性
    all_attributes = np.concatenate([
        xyz,           # x, y, z
        normals,       # nx, ny, nz
        f_dc,          # f_dc_0, f_dc_1, f_dc_2
        f_rest,        # f_rest_*
        opacities,     # opacity
        scales,        # scale_0, scale_1, scale_2
        rots           # rot_0, rot_1, rot_2, rot_3
    ], axis=1)
    
    # 创建结构化数组
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, all_attributes))
    
    # 写入PLY文件
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    
    print(f"成功保存 {xyz.shape[0]} 个红色高斯点到 {path}")


def convert_json_to_ply(json_path, output_ply_path, fixed_scale=0.01):
    """
    将deformation_graph.json转换为红色高斯点云PLY文件
    
    Args:
        json_path: 输入的JSON文件路径
        output_ply_path: 输出的PLY文件路径
        fixed_scale: 固定的scale值，默认0.01
    """
    print("="*60)
    print("开始转换 JSON -> PLY")
    print("="*60)
    
    # 读取位置
    xyz = read_positions_from_json(json_path)
    
    print("\n" + "="*60)
    print("保存为红色高斯点云")
    print("="*60)
    
    # 保存为红色高斯点云
    save_ply_with_xyz(xyz, output_ply_path, fixed_scale=fixed_scale)
    
    print("\n" + "="*60)
    print(f"转换完成！")
    print(f"输入: {json_path}")
    print(f"输出: {output_ply_path}")
    print("="*60)


# 使用示例
if __name__ == "__main__":
    # # 方法1：读取为numpy数组
    # print("方法1：读取为numpy数组")
    # print("-"*60)
    # xyz = read_positions_from_json('/home/momo/Desktop/tao_data_2/deformation_graph.json')
    # print(f"\nNumpy数组 shape: {xyz.shape}")
    # print(f"前3个点:\n{xyz[:3]}")
    
    # # 方法2：读取为tensor
    # print("\n\n方法2：读取为tensor")
    # print("-"*60)
    # xyz_tensor = read_positions_as_tensor('/mnt/user-data/uploads/deformation_graph.json', device='cpu')
    # print(f"Tensor shape: {xyz_tensor.shape}")
    # print(f"Tensor device: {xyz_tensor.device}")
    
    # 方法3：完整转换流程
    print("\n\n方法3：转换为PLY文件")
    print("-"*60)
    convert_json_to_ply(
        '/home/momo/Desktop/tao_data_2/deformation_graph.json',
        '/home/momo/Desktop/tao_data_2/dg.ply',
        fixed_scale=0.005  # 可以调整这个值来改变点的大小
    )
