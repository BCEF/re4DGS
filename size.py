import torch
import numpy as np
import os

def get_size_mb(tensor):
    """计算张量的内存大小（MB）"""
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement() / (1024 ** 2)
    elif isinstance(tensor, np.ndarray):
        return tensor.nbytes / (1024 ** 2)
    else:
        # 对于其他类型，尝试估算
        import sys
        return sys.getsizeof(tensor) / (1024 ** 2)

def calculate_total_size(value):
    """
    递归计算一个值的总内存大小
    
    参数:
        value: 要计算的值
    返回:
        总大小（MB）
    """
    total = 0
    
    if isinstance(value, torch.Tensor):
        total += get_size_mb(value)
    elif isinstance(value, dict):
        for v in value.values():
            total += calculate_total_size(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            total += calculate_total_size(item)
    
    return total

def print_value_info(key, value, indent=0, total_size_tracker=None, max_depth=None, current_depth=0):
    """
    递归打印值的信息
    
    参数:
        key: 键名
        value: 值
        indent: 缩进级别
        total_size_tracker: 用于累计总大小的列表
        max_depth: 最大递归深度，None表示无限制
        current_depth: 当前深度
    """
    if total_size_tracker is None:
        total_size_tracker = [0]
    
    # 检查是否超过最大深度
    if max_depth is not None and current_depth >= max_depth:
        prefix = "  " * indent
        
        if isinstance(value, torch.Tensor):
            size_mb = get_size_mb(value)
            shape = str(tuple(value.shape))
            dtype = str(value.dtype).replace('torch.', '')
            print(f"{prefix}{key:<{40-2*indent}} Tensor{shape} [{dtype}] {size_mb:>10.4f} MB")
            total_size_tracker[0] += size_mb
        elif isinstance(value, (dict, list, tuple)):
            type_name = type(value).__name__
            length = len(value)
            # 计算这个容器的总大小
            container_size = calculate_total_size(value)
            print(f"{prefix}{key:<{40-2*indent}} {type_name} (长度: {length}) {container_size:>10.4f} MB [已达最大深度]")
            total_size_tracker[0] += container_size
        else:
            print(f"{prefix}{key:<{40-2*indent}} {type(value).__name__}")
        return total_size_tracker[0]
    
    prefix = "  " * indent
    
    if isinstance(value, torch.Tensor):
        size_mb = get_size_mb(value)
        shape = str(tuple(value.shape))
        dtype = str(value.dtype).replace('torch.', '')
        print(f"{prefix}{key:<{40-2*indent}} Tensor{shape} [{dtype}] {size_mb:>10.4f} MB")
        total_size_tracker[0] += size_mb
        
    elif isinstance(value, dict):
        # 先计算字典的总大小
        dict_total_size = calculate_total_size(value)
        print(f"{prefix}{key} (字典，包含 {len(value)} 个项，总计 {dict_total_size:.4f} MB)")
        print(f"{prefix}{'-' * (80 - 2*indent)}")
        for sub_key, sub_value in value.items():
            print_value_info(sub_key, sub_value, indent + 1, total_size_tracker, max_depth, current_depth + 1)
        print(f"{prefix}{'-' * (80 - 2*indent)}")
        
    elif isinstance(value, (list, tuple)):
        type_name = "列表" if isinstance(value, list) else "元组"
        # 先计算列表/元组的总大小
        container_total_size = calculate_total_size(value)
        print(f"{prefix}{key} ({type_name}，包含 {len(value)} 个项，总计 {container_total_size:.4f} MB)")
        print(f"{prefix}{'-' * (80 - 2*indent)}")
        for i, item in enumerate(value):
            print_value_info(f"[{i}]", item, indent + 1, total_size_tracker, max_depth, current_depth + 1)
        print(f"{prefix}{'-' * (80 - 2*indent)}")
        
    elif isinstance(value, (int, float, str, bool)):
        print(f"{prefix}{key:<{40-2*indent}} {type(value).__name__}: {value}")
        
    else:
        # 对于自定义对象，尝试检查其属性
        print(f"{prefix}{key:<{40-2*indent}} {type(value).__name__}")
        # 尝试获取对象的所有属性
        try:
            obj_dict = vars(value)
            if obj_dict:
                obj_size = calculate_total_size(obj_dict)
                print(f"{prefix}  └─ 对象属性总计: {obj_size:.4f} MB")
                total_size_tracker[0] += obj_size
        except:
            pass
    
    return total_size_tracker[0]

def print_checkpoint_info(checkpoint_path, max_depth=None):
    """
    读取checkpoint文件并打印每个属性的大小
    
    参数:
        checkpoint_path: checkpoint文件路径
        max_depth: 最大递归深度，None表示无限制
    """
    print(f"正在读取checkpoint: {checkpoint_path}")
    print(f"文件大小: {os.path.getsize(checkpoint_path) / (1024**2):.2f} MB\n")
    
    # 读取checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 检查checkpoint的类型
    print(f"Checkpoint类型: {type(checkpoint).__name__}")
    if max_depth is not None:
        print(f"最大显示深度: {max_depth}")
    
    print("=" * 80)
    print(f"{'属性路径':<40} {'类型/形状':<25} {'大小(MB)':<15}")
    print("=" * 80)
    
    total_size_tracker = [0]
    
    # 根据类型处理
    if isinstance(checkpoint, (tuple, list)):
        print(f"根级别 ({type(checkpoint).__name__}，包含 {len(checkpoint)} 个元素)")
        print("-" * 80)
        for i, item in enumerate(checkpoint):
            print_value_info(f"element_{i}", item, indent=0, total_size_tracker=total_size_tracker, max_depth=max_depth, current_depth=0)
    elif isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            print_value_info(key, value, indent=0, total_size_tracker=total_size_tracker, max_depth=max_depth, current_depth=0)
    else:
        print_value_info("root", checkpoint, indent=0, total_size_tracker=total_size_tracker, max_depth=max_depth, current_depth=0)
    
    print("=" * 80)
    print(f"{'张量总内存占用:':<60} {total_size_tracker[0]:>10.4f} MB")
    print("=" * 80)

def print_checkpoint_info_limited(checkpoint_path):
    """
    读取checkpoint文件并只打印到第2层（15个子元素这一级）
    
    参数:
        checkpoint_path: checkpoint文件路径
    """
    print_checkpoint_info(checkpoint_path, max_depth=2)

# 使用示例
if __name__ == "__main__":
    # 请替换为您的checkpoint文件路径
    checkpoint_path = "/home/momo/Desktop/tao_data_2_output_newHP_10/chkpnt450000.pth"
    
    if os.path.exists(checkpoint_path):
        # 方式1: 打印完整结构（无限深度）- 用这个找出所有数据
        print("\n" + "="*80)
        print("完整结构分析（深度不限）:")
        print("="*80)
        print_checkpoint_info(checkpoint_path)
        
        print("\n\n" + "="*80)
        print("简化视图（只到第2层）:")
        print("="*80)
        # 方式2: 只打印到15个子元素这一级（深度=2）
        print_checkpoint_info_limited(checkpoint_path)
        
        # 方式3: 自定义深度
        # print_checkpoint_info(checkpoint_path, max_depth=3)
    else:
        print(f"错误: 文件 '{checkpoint_path}' 不存在")
        print("\n使用方法:")
        print("1. print_checkpoint_info(path) - 打印完整结构")
        print("2. print_checkpoint_info_limited(path) - 只打印到第2层")
        print("3. print_checkpoint_info(path, max_depth=N) - 打印到第N层")