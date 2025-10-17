import numpy as np
import os

def load_smplx_vertices_from_npz(npz_path, smplx_model_path, gender='neutral', 
                                 num_betas=100, num_expression_coeffs=50):
    """从NPZ文件加载并生成SMPLX顶点"""
    try:
        import smplx,torch
    except ImportError:
        print("❌ 未安装 smplx 库，请运行: pip install smplx")
        return None, None
    
    body_model = smplx.create(
        model_path=smplx_model_path,
        model_type='smplx',
        gender=gender,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        use_pca=False,
        flat_hand_mean=True,
        ext='npz'
    )
    
    data = np.load(npz_path)
    
    betas = torch.from_numpy(data['betas']).float().unsqueeze(0)
    trans = torch.from_numpy(data['trans']).float().unsqueeze(0)
    root_orient = torch.from_numpy(data['root_orient']).float().unsqueeze(0)
    pose_body = torch.from_numpy(data['pose_body']).float().unsqueeze(0)
    pose_hand = torch.from_numpy(data['pose_hand']).float().unsqueeze(0)
    expression = torch.from_numpy(data['expression']).float().unsqueeze(0)
    
    left_hand_pose = pose_hand[:, :45]
    right_hand_pose = pose_hand[:, 45:]
    
    with torch.no_grad():
        output = body_model(
            betas=betas,
            global_orient=root_orient,
            body_pose=pose_body,
            transl=trans,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=expression,
            jaw_pose=torch.zeros(1, 3),
            leye_pose=torch.zeros(1, 3),
            reye_pose=torch.zeros(1, 3)
        )
        
        vertices = output.vertices.squeeze(0).cpu().numpy()
        faces = body_model.faces.astype(np.int32)
    
    return vertices,faces


from deformation_tool import generate_deformation_graph
from deformation_tool import compute_deformation_transforms
if __name__=="__main__":
    ref_model="/home/momo/Desktop/tao_data/000000/000000.npz"
    deformation_graph_path="/home/momo/Desktop/tao_data/"
    vertices,faces=load_smplx_vertices_from_npz(ref_model,"./models")
    dg=generate_deformation_graph(
        vertices=vertices,
        faces=faces,
        node_num=200,
        radius_coef=5.0,
        node_nodes_num=8,       # 每个节点的最大连接节点数
        v_nodes_num=12           # 每个顶点的最大连接节点数
    )
    dg.save(os.path.join(deformation_graph_path,"deformation_graph.json"))

    root_folder="/home/momo/Desktop/tao_data/"
    for folder in os.listdir(root_folder):
        vertices_a=vertices.copy()
        npz_path=os.path.join(root_folder,folder,folder+".npz")
        if not os.path.exists(npz_path):
            continue
        vertices_b,_=load_smplx_vertices_from_npz(npz_path,"./models")

        transforms = compute_deformation_transforms(dg, vertices_a, vertices_b)
        transforms.save(os.path.join(root_folder,folder,"transforms.json"))

        inv_transforms=compute_deformation_transforms(dg, vertices_b, vertices_a)
        inv_transforms.save(os.path.join(root_folder,folder,"inv_transforms.json"))
