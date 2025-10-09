import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh,os
from deformation_graph import DeformationGraph
from generate_deformation_graph import generate_deformation_graph
from deformation_utils import compute_deformation_transforms, apply_deformation, DeformationTransforms
def visualize_deformation_graph(mesh, dg):
    """可视化网格和变形图"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制网格边缘
    edges = mesh.edges_unique
    for edge in edges:
        p1, p2 = mesh.vertices[edge]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', alpha=0.1)
    
    # 绘制控制节点
    ax.scatter(dg.node_positions[:, 0], 
               dg.node_positions[:, 1], 
               dg.node_positions[:, 2], 
               c='r', s=50, label='Control Nodes')
    
    # 绘制前10个节点的连接
    for i in range(min(10, len(dg.node_nodes))):
        for j, _ in dg.node_nodes[i]:
            p1 = dg.node_positions[i]
            p2 = dg.node_positions[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', alpha=0.5)
    
    ax.set_title(f" ({len(dg.nodes)} )")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

def visualize_meshes(original, target, deformed, faces):
    """可视化原始网格、目标网格和变形后的网格"""
    fig = plt.figure(figsize=(18, 6))
    
    # 原始网格
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('mesh A')
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            ax1.plot([original[v1, 0], original[v2, 0]], 
                    [original[v1, 1], original[v2, 1]], 
                    [original[v1, 2], original[v2, 2]], 'b-', alpha=0.5)
    
    # 目标网格
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('mesh B')
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            ax2.plot([target[v1, 0], target[v2, 0]], 
                    [target[v1, 1], target[v2, 1]], 
                    [target[v1, 2], target[v2, 2]], 'r-', alpha=0.5)
    
    # 变形后的网格
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('deform A→B')
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            ax3.plot([deformed[v1, 0], deformed[v2, 0]], 
                    [deformed[v1, 1], deformed[v2, 1]], 
                    [deformed[v1, 2], deformed[v2, 2]], 'g-', alpha=0.5)
    
    # 设置相同的视角和缩放
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    plt.tight_layout()
    plt.savefig('deformation_result_separated.png', dpi=300)
    plt.show()

def compute_transforms(mesh_a_path, mesh_b_path,deformation_graph_path,out_deformation_transforms_path):
    """
    第一部分：计算从网格A到网格B的变形变换
    
    参数:
        mesh_a_path: 源网格A的OBJ文件路径
        mesh_b_path: 目标网格B的OBJ文件路径
    """
    print("===== 第一部分：计算变形变换 =====")
    
    # 1. 加载变形图
    dg = DeformationGraph()
    dg.load(deformation_graph_path)
    
    # 2. 加载原始网格A
    print(f"加载源网格: {mesh_a_path}")
    mesh_a = trimesh.load(mesh_a_path,process=False)
    vertices_a = np.array(mesh_a.vertices)
    faces = np.array(mesh_a.faces)
    
    # 3. 加载目标网格B
    print(f"加载目标网格: {mesh_b_path}")
    mesh_b = trimesh.load(mesh_b_path,process=False)
    vertices_b = np.array(mesh_b.vertices)
    
    # 确保两个网格有相同的拓扑结构(相同数量的顶点)
    if len(vertices_a) != len(vertices_b):
        raise ValueError(f"网格拓扑不匹配: 网格A有{len(vertices_a)}个顶点，而网格B有{len(vertices_b)}个顶点")
    
    # 验证面的数量也相同(进一步确认拓扑一致)
    if len(mesh_a.faces) != len(mesh_b.faces):
        print(f"警告: 网格面数不匹配: 网格A有{len(mesh_a.faces)}个面，而网格B有{len(mesh_b.faces)}个面")
    
    print(f"网格顶点数: {len(vertices_a)}, 面数: {len(faces)}")
    
    # 4. 计算从A到B的变形变换
    print("计算变形变换...")
    transforms = compute_deformation_transforms(dg, vertices_a, vertices_b)
    
    # 5. 保存变换参数到文件
    transforms.save(out_deformation_transforms_path)
    
    print("变形变换已计算并保存。")
    return vertices_a, vertices_b, faces

def apply_transforms(deformation_graph_path,deformation_transforms_path,mesh_a_path,output_mesh_path):
    """第二部分：将预先计算的变形变换应用到网格A"""
    print("\n===== 第二部分：应用变形变换 =====")
    
    # 1. 加载变形图
    dg = DeformationGraph()
    dg.load(deformation_graph_path)
    
    # 2. 加载原始网格A
    mesh_a = trimesh.load(mesh_a_path,process=False)
    vertices_a = np.array(mesh_a.vertices)
    faces = np.array(mesh_a.faces)
    
    # 3. 加载预计算的变换参数
    transforms = DeformationTransforms()
    transforms.load(deformation_transforms_path)
    
    # 4. 应用变形变换到网格A
    deformed_vertices = apply_deformation(dg, vertices_a, transforms)
    
    # 5. 保存变形后的网格
    deformed_mesh = trimesh.Trimesh(vertices=deformed_vertices, faces=faces)
    deformed_mesh.export(output_mesh_path)
    
    print(f"变形已应用，结果已保存为{output_mesh_path}")
    
    # 创建目标网格B (用于可视化比较)
    vertices_b = vertices_a.copy()
    vertices_b[:, 0] *= 1.5
    vertices_b[:, 1] *= 0.7
    mask = vertices_b[:, 2] > 0
    vertices_b[mask, 2] *= 1.3
    
    return vertices_a, vertices_b, deformed_vertices, faces

# 在主函数中调用
def main(mesh_a,mesh_b,output_folder):
    os.makedirs(output_folder, exist_ok=True)

    deformation_graph_path=os.path.join(output_folder,"deformation_graph.json")
    deformation_transforms_path=os.path.join(output_folder,os.path.basename(mesh_b).replace(".obj",".json"))
    output_mesh_path=os.path.join(output_folder,"deformed_mesh.obj")

   

    # visualize_deformation_graph(mesh, dg)
    # 从两个OBJ文件计算变换
    vertices_a, vertices_b, faces = compute_transforms(
        mesh_a,  # 源网格OBJ文件路径
        mesh_b,   # 目标网格OBJ文件路径
        deformation_graph_path,
        deformation_transforms_path
    )
    import time
    st=time.time()
    # 应用变换
    
    # vertices_a, vertices_b, deformed_vertices, faces = apply_transforms(deformation_graph_path,deformation_transforms_path,mesh_a,output_mesh_path)
    print(time.time()-st)
    # 可视化结果
    # visualize_meshes(vertices_a, vertices_b, deformed_vertices, faces)

if __name__ == "__main__":

    #第一帧重拓扑obj
    mesh_a="/home/momo/mount_folder/home/momo/Desktop/qianqian_data/obj/Frame000075.obj"
    #所有帧重拓扑obj
    mesh_b_folder="/home/momo/mount_folder/home/momo/Desktop/qianqian_data/obj"
    #输出文件夹
    output_folder="/home/momo/mount_folder/home/momo/Desktop/qianqian_data/deform_graph"
    #变形图保存目录
    deformation_graph_path=os.path.join(output_folder,"deformation_graph.json")
    #根据mesh a生成变形图
    mesh = trimesh.load(mesh_a)
    print(f"网格加载完成: {len(mesh.vertices)} 个顶点, {len(mesh.faces)} 个面")
    
    # 生成变形图
    dg = generate_deformation_graph(
        vertices=mesh.vertices,
        faces=mesh.faces,
        node_num=200,            # 控制节点数量
        radius_coef=5.0,        # 影响半径系数
        node_nodes_num=8,       # 每个节点的最大连接节点数
        v_nodes_num=12           # 每个顶点的最大连接节点数
    )
    visualize_deformation_graph(mesh, dg)
        # 保存变形图
    dg.save(deformation_graph_path)
    
    #注意这里第一帧也计算一次，避免后续因i-1产生的麻烦
    #生成从前一帧到当前帧的变形
    mesh_b=os.path.join(mesh_b_folder,f"Frame{75:06d}.obj")
    main(mesh_a,mesh_b,output_folder)

    for i in range(76,1200):
        mesh_a=os.path.join(mesh_b_folder,f"Frame{i-1:06d}.obj")
        mesh_b=os.path.join(mesh_b_folder,f"Frame{i:06d}.obj")
        main(mesh_a,mesh_b,output_folder)