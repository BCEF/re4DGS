import numpy as np
import time
from .deformation_graph import DeformationGraph
from .pointcloud_fast_marching import FastMarchingPointCloud

def generate_deformation_graph_pointcloud(points, 
                                         node_num=100, 
                                         radius_coef=2.1, 
                                         node_nodes_num=4, 
                                         v_nodes_num=6,
                                         k_neighbors=15,
                                         initial_seed=None):
    """
    从点云生成变形图
    
    参数:
        points: 点云坐标数组 (n x 3)
        node_num: 需要的控制节点数量
        radius_coef: 节点影响半径系数
        node_nodes_num: 每个节点连接的最大节点数
        v_nodes_num: 每个顶点连接的最大节点数
        k_neighbors: k近邻图的邻居数量
        initial_seed: 初始种子点（默认随机选择）
    
    返回:
        dg: 生成的变形图对象
    """
    print("=" * 60)
    print("开始从点云生成变形图...")
    print(f"点云大小: {len(points)} 个点")
    print(f"目标节点数: {node_num}")
    print(f"k近邻数: {k_neighbors}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 创建变形图对象
    dg = DeformationGraph()
    
    # 创建点云Fast Marching工具
    fm = FastMarchingPointCloud(k_neighbors=k_neighbors)
    fm.set_pointcloud(points)
    
    # 使用最远点采样选择节点
    print("\n[1/4] 使用最远点采样选择控制节点...")
    nodes, max_distance = fm.farthest_point_sampling(node_num, initial_point=initial_seed)
    dg.nodes = nodes
    
    # 设置节点位置
    dg.node_positions = points[nodes]
    
    # 设置节点影响半径
    influence_radius = max_distance * radius_coef
    dg.node_radius = influence_radius
    print(f"节点影响半径设置为: {influence_radius:.4f}")
    
    # 初始化连接数据结构
    n_nodes = len(nodes)
    n_points = len(points)
    dg.node_nodes = [[] for _ in range(n_nodes)]
    dg.v_nodes = [[] for _ in range(n_points)]
    
    # 节点索引映射
    node_index_map = {node: i for i, node in enumerate(nodes)}
    
    # 计算节点-节点连接
    print(f"\n[2/4] 计算节点-节点连接...")
    for i, node in enumerate(nodes):
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{n_nodes}")
        
        # 计算从当前节点到所有点的距离
        fm.compute_distance([node], max_distance=influence_radius)
        
        # 为当前节点找到邻近节点
        for other_node in nodes:
            if other_node != node and fm.distance[other_node] < influence_radius:
                dg.node_nodes[i].append((node_index_map[other_node], fm.distance[other_node]))
    
    # 统计节点连接信息
    node_conn_counts = [len(conns) for conns in dg.node_nodes]
    print(f"  节点平均连接数: {np.mean(node_conn_counts):.2f}")
    print(f"  连接数范围: [{np.min(node_conn_counts)}, {np.max(node_conn_counts)}]")
    
    # 计算顶点-节点连接
    print(f"\n[3/4] 计算顶点-节点连接...")
    for node_idx, node in enumerate(nodes):
        if (node_idx + 1) % 10 == 0:
            print(f"  进度: {node_idx + 1}/{n_nodes}")
        
        # 计算从当前节点到所有点的距离
        fm.compute_distance([node], max_distance=influence_radius)
        
        # 为每个顶点添加当前节点的影响
        for v in range(n_points):
            if fm.distance[v] < influence_radius:
                # 使用高斯权重函数
                weight = np.exp(-2.0 * (fm.distance[v] / influence_radius) ** 2)
                dg.v_nodes[v].append((node_idx, weight))
    
    # 统计顶点连接信息
    vertex_conn_counts = [len(conns) for conns in dg.v_nodes]
    print(f"  顶点平均连接数: {np.mean(vertex_conn_counts):.2f}")
    print(f"  连接数范围: [{np.min(vertex_conn_counts)}, {np.max(vertex_conn_counts)}]")
    
    # 检查孤立顶点
    isolated_vertices = sum(1 for conns in dg.v_nodes if len(conns) == 0)
    if isolated_vertices > 0:
        print(f"  警告: 发现 {isolated_vertices} 个孤立顶点（未连接到任何节点）")
    
    # 减少连接数量
    print(f"\n[4/4] 优化连接数量...")
    dg.reduce_connections(node_nodes_num, v_nodes_num)
    
    # 归一化顶点-节点权重
    dg.normalize_weights()
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"变形图生成完成！")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"节点数: {len(dg.nodes)}")
    print(f"节点位置数: {len(dg.node_positions)}")
    print("=" * 60)
    
    return dg


def generate_deformation_graph_auto(vertices, faces=None, **kwargs):
    """
    自动选择点云或网格模式生成变形图
    
    参数:
        vertices: 点坐标数组 (n x 3)
        faces: 面索引数组（可选，如果提供则使用网格模式）
        **kwargs: 其他参数
    
    返回:
        dg: 生成的变形图对象
    """
    if faces is not None and len(faces) > 0:
        print("检测到面信息，使用网格模式...")
        from .generate_deformation_graph import generate_deformation_graph
        return generate_deformation_graph(vertices, faces, **kwargs)
    else:
        print("未检测到面信息，使用点云模式...")
        return generate_deformation_graph_pointcloud(vertices, **kwargs)