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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

#SUMO
from dataclasses import dataclass
@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    #SUMO
    bg_path:str=""
    deformer_path:str=""
    kid:int=0
    timecode: float = 0.0 

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, background,depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    
    #SUMO
    for camera_info in cam_infos_unsorted:
        bg_reading_dir="bg" if background==None else background
        camera_info.bg_path=os.path.join(path,bg_reading_dir,camera_info.image_name)

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

#SUMO
def readDeformSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8,colmap_folder=None,root_folder=None,deformer_path=None,bg_img_folder=None,kid=0,timecode=0.0):
    reading_dir = "images" if images == None else images

    sparse_folder=os.path.join(path, "sparse/0") if colmap_folder is None else colmap_folder
    tao_camera_path=os.path.join(root_folder,"cam_params.json")
    if os.path.exists(sparse_folder):
        cameras_extrinsic_file = os.path.join(sparse_folder, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_folder, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

        depth_params_file = os.path.join(sparse_folder, "depth_params.json")
        ## if depth_params_file isnt there AND depths file is here -> throw error
        depths_params = None
        if depths != "":
            try:
                with open(depth_params_file, "r") as f:
                    depths_params = json.load(f)
                all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
                if (all_scales > 0).sum():
                    med_scale = np.median(all_scales[all_scales > 0])
                else:
                    med_scale = 0
                for key in depths_params:
                    depths_params[key]["med_scale"] = med_scale

            except FileNotFoundError:
                print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
                sys.exit(1)

        if eval:
            if "360" in path:
                llffhold = 8
            if llffhold:
                print("------------LLFF HOLD-------------")
                cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
                cam_names = sorted(cam_names)
                test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
            else:
                with open(os.path.join(sparse_folder, "test.txt"), 'r') as file:
                    test_cam_names_list = [line.strip() for line in file]
        else:
            test_cam_names_list = []

        
        cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
            images_folder=os.path.join(path, reading_dir), 
            depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list,
            )
        
        ply_path=os.path.join(colmap_folder,"points3D.ply")
        bin_path=os.path.join(colmap_folder,"points3D.bin")
        txt_path=os.path.join(colmap_folder,"points3D.txt")

        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    elif os.path.exists(tao_camera_path):
        with open(tao_camera_path, 'r') as f:
            cameras = json.load(f)
        
        if eval:
            if "360" in path:
                llffhold = 8
            if llffhold:
                print("------------LLFF HOLD-------------")
                cam_names = cameras.get('all_cam_names', list(cameras.keys()))
                cam_names = sorted(cam_names)
                test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
            else:
                with open(os.path.join(sparse_folder, "test.txt"), 'r') as file:
                    test_cam_names_list = [line.strip() for line in file]
        else:
            test_cam_names_list = []
        
        cam_infos_unsorted = readTaoCameras(
            cameras=cameras, 
            images_folder=os.path.join(path, reading_dir), 
            depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list,
            )
        
        ply_path=os.path.join(root_folder,"input.ply")
        if not os.path.exists(ply_path):
            # npz_path="/home/momo/Documents/xwechat_files/wxid_46mm76y10kh221_6c39/msg/file/2025-10/000000.npz"
            npz_path=os.path.join(root_folder,"smplx.npz")
            vertices=load_smplx_vertices_from_npz(npz_path,'./models/')
            # 为顶点分配颜色（皮肤色）
            num_pts = len(vertices)
            colors = np.tile([200, 150, 100], (num_pts, 1))  # 皮肤色
            
            storePly(ply_path, vertices, colors)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    
    #读取变形
    deformer_path=os.path.join(path,"transforms.json") if deformer_path==None else deformer_path
    for camera_info in cam_infos_unsorted:
        bg_reading_dir="bg" if bg_img_folder==None else bg_img_folder
        camera_info.bg_path=os.path.join(path,bg_reading_dir,camera_info.image_name)
        camera_info.deformer_path=deformer_path
        camera_info.kid=kid
        camera_info.timecode=timecode

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

#SUMO
def readTaoCameras(cameras, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    all_camera_names=cameras.get('all_cam_names', list(cameras.keys()))
    for uid,cam_name in enumerate(all_camera_names):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}".format(cam_name))
        sys.stdout.flush()
        cam_info = cameras[cam_name]
        K = np.array(cam_info['K'], dtype=np.float32)
        dist = np.array(cam_info['D'], dtype=np.float32).ravel()
        R = np.array(cam_info['R'], dtype=np.float32)
        T = np.array(cam_info['T'], dtype=np.float32).ravel()
        
        ###################333
        height = int(cam_info.get('height', 1024))
        width = int(cam_info.get('width', 1024))
        M = np.eye(3)
        w_ = K[0, 2] - width / 2
        h_ = K[1, 2] - height / 2
        M[0, 2] = (w_) / K[0, 0]
        M[1, 2] = (h_) / K[1, 1]
        K[0, 2] = width / 2
        K[1, 2] = height / 2
        R = M @ R
        T = M @ T
        ###################333

        # 转换为 3DGS 需要的格式
        R = np.transpose(R)  # 3DGS 使用转置的旋转矩阵
        
        # 计算视场角
        height = int(cam_info.get('height', 1024))
        width = int(cam_info.get('width', 1024))
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)


        image_name = cam_name+".jpg"
        image_path = os.path.join(images_folder, image_name)
        
        depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=None,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

# def readTaoAvatarSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8, train_views=None, train_frames=None, val_views=None, val_frames=None):
#     """
#     读取 TaoAvatar/x_avatar 格式的数据集
#     数据格式：
#     path/
#         ├── cam_params.json
#         ├── 1/
#         │   ├── 000000.jpg
#         │   ├── 000000.png
#         │   └── ...
#         ├── 2/
#         └── models/
#             ├── 000000.npz
#             └── ...
#     """
#     import cv2
    
#     # 读取相机参数
#     cam_params_file = os.path.join(path, 'cam_params.json')
#     with open(cam_params_file, 'r') as f:
#         cameras = json.load(f)
    
#     # 读取 SMPL-X 参数文件列表
#     models_dir = os.path.join(path, 'models')
#     model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.npz')])
    
#     # 确定训练和测试的相机和帧
#     all_cam_names = cameras.get('all_cam_names', list(cameras.keys()))
    
#     if train_views is None and val_views is None:
#         # 使用 llffhold 自动分配训练/测试相机
#         if llffhold > 0:
#             # 类似 LLFF 的方式：每 llffhold 个相机选一个作为测试
#             sorted_cam_names = sorted(all_cam_names)
#             val_views = [name for idx, name in enumerate(sorted_cam_names) if idx % llffhold == 0]
#             train_views = [name for name in sorted_cam_names if name not in val_views]
#             print(f"使用 llffhold={llffhold} 自动分配相机:")
#             print(f"  训练相机 ({len(train_views)}个): {train_views}")
#             print(f"  测试相机 ({len(val_views)}个): {val_views}")
#         else:
#             # llffhold=0: 所有相机都用于训练
#             train_views = all_cam_names
#             val_views = []
#             print(f"llffhold=0: 使用所有 {len(train_views)} 个相机进行训练")
#     else:
#         # 手动指定了 train_views 或 val_views
#         if train_views is None:
#             train_views = all_cam_names
#         if val_views is None:
#             val_views = train_views[:1] if train_views else []
    
#     # 帧范围：[start, end, step] 或直接的帧列表
#     if train_frames is None:
#         train_frames = [0, len(model_files), 1]
#     if val_frames is None:
#         val_frames = [0, len(model_files), 10]
    
#     # 解析帧范围或帧列表
#     if len(train_frames) == 3 and all(isinstance(x, int) for x in train_frames):
#         # 范围模式: [start, end, step]
#         start_frame, end_frame, sampling_rate = train_frames
#         if end_frame == 0:
#             end_frame = len(model_files)
#         train_frame_indices = list(range(start_frame, end_frame, sampling_rate))
#     else:
#         # 列表模式: 直接使用提供的帧索引
#         train_frame_indices = train_frames
#         print(f"使用帧列表模式，训练帧: {train_frame_indices}")
    
#     if eval:
#         # 解析验证帧：支持范围或列表
#         if len(val_frames) == 3 and all(isinstance(x, int) for x in val_frames):
#             # 范围模式
#             val_start, val_end, val_step = val_frames
#             if val_end == 0:
#                 val_end = len(model_files)
#             val_frame_indices = list(range(val_start, val_end, val_step))
#         else:
#             # 列表模式
#             val_frame_indices = val_frames
#             print(f"使用帧列表模式，验证帧: {val_frame_indices}")
#         test_cam_names = val_views
#     else:
#         val_frame_indices = []
#         test_cam_names = []
    
#     # 构建相机信息列表
#     cam_infos_unsorted = []
#     uid = 0
    
#     # 训练相机
#     for cam_name in train_views:
#         if cam_name not in cameras:
#             print(f"Warning: Camera {cam_name} not found in cam_params.json")
#             continue
            
#         cam_info = cameras[cam_name]
#         K = np.array(cam_info['K'], dtype=np.float32)
#         dist = np.array(cam_info['D'], dtype=np.float32).ravel()
#         R = np.array(cam_info['R'], dtype=np.float32)
#         T = np.array(cam_info['T'], dtype=np.float32).ravel()
        
#         # 转换为 3DGS 需要的格式
#         R = np.transpose(R)  # 3DGS 使用转置的旋转矩阵
        
#         # 计算视场角
#         height = int(cam_info.get('height', 1024))
#         width = int(cam_info.get('width', 1024))
#         focal_length_x = K[0, 0]
#         focal_length_y = K[1, 1]
#         FovY = focal2fov(focal_length_y, height)
#         FovX = focal2fov(focal_length_x, width)
        
#         # 为每一帧创建相机信息
#         for frame_idx in train_frame_indices:
#             model_file = model_files[frame_idx]
#             image_file = os.path.join(path, cam_name, f"{frame_idx:06d}.jpg")
#             # mask_file = os.path.join(path, cam_name, f"{frame_idx:06d}.png")
            
#             if not os.path.exists(image_file):
#                 continue
            
#             image_name = f"{cam_name}_{frame_idx:06d}"
            
#             # 获取变形场路径（优先使用环境变量中的路径）
#             deform_base_path = os.environ.get('DEFORM_PATH', None)
#             if deform_base_path and os.path.exists(deform_base_path):
#                 # 使用变形场的 transforms.json
#                 deformer_path = os.path.join(deform_base_path, f"{frame_idx:06d}", "transforms.json")
#             else:
#                 # 回退到 models 目录（向后兼容）
#                 deformer_path = os.path.join(models_dir, model_file)
            
#             # 创建 CameraInfo
#             cam_infos_unsorted.append(CameraInfo(
#                 uid=uid,
#                 R=R,
#                 T=T,
#                 FovY=FovY,
#                 FovX=FovX,
#                 depth_params=None,
#                 image_path=image_file,
#                 image_name=image_name,
#                 depth_path="",
#                 width=width,
#                 height=height,
#                 is_test=False,
#                 # TaoAvatar 特有字段
#                 kid=frame_idx,
#                 timecode=frame_idx / len(model_files),
#                 deformer_path=deformer_path,
#                 bg_path=None  # TaoAvatar 数据集没有背景图
#             ))
#             uid += 1
    
#     # 测试相机（如果需要）
#     for cam_name in test_cam_names:
#         if cam_name not in cameras:
#             continue
            
#         cam_info = cameras[cam_name]
#         K = np.array(cam_info['K'], dtype=np.float32)
#         R = np.array(cam_info['R'], dtype=np.float32)
#         T = np.array(cam_info['T'], dtype=np.float32).ravel()
#         R = np.transpose(R)
        
#         height = int(cam_info.get('height', 1024))
#         width = int(cam_info.get('width', 1024))
#         focal_length_x = K[0, 0]
#         focal_length_y = K[1, 1]
#         FovY = focal2fov(focal_length_y, height)
#         FovX = focal2fov(focal_length_x, width)
        
#         for frame_idx in val_frame_indices:
#             model_file = model_files[frame_idx]
#             image_file = os.path.join(path, cam_name, f"{frame_idx:06d}.jpg")
#             mask_file = os.path.join(path, cam_name, f"{frame_idx:06d}.png")
            
#             if not os.path.exists(image_file):
#                 continue
            
#             image_name = f"{cam_name}_{frame_idx:06d}"
            
#             # 获取变形场路径（优先使用环境变量中的路径）
#             deform_base_path = os.environ.get('DEFORM_PATH', None)
#             if deform_base_path and os.path.exists(deform_base_path):
#                 # 使用变形场的 transforms.json
#                 deformer_path = os.path.join(deform_base_path, f"{frame_idx:06d}", "transforms.json")
#             else:
#                 # 回退到 models 目录（向后兼容）
#                 deformer_path = os.path.join(models_dir, model_file)
            
#             cam_infos_unsorted.append(CameraInfo(
#                 uid=uid,
#                 R=R,
#                 T=T,
#                 FovY=FovY,
#                 FovX=FovX,
#                 depth_params=None,
#                 image_path=image_file,
#                 image_name=image_name,
#                 depth_path="",
#                 width=width,
#                 height=height,
#                 is_test=True,
#                 flame_params=None,
#                 kid=frame_idx,
#                 timecode=frame_idx / len(model_files),
#                 deformer_path=deformer_path,
#                 bg_path=None  # TaoAvatar 数据集没有背景图
#             ))
#             uid += 1
    
#     # 排序相机信息
#     cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    
#     train_cam_infos = [c for c in cam_infos if not c.is_test]
#     test_cam_infos = [c for c in cam_infos if c.is_test]
    
#     # 计算 NeRF 归一化参数
#     nerf_normalization = getNerfppNorm(train_cam_infos)
    
#     # 生成初始点云（从第一个 SMPL-X 模型）
#     ply_path = os.path.join(path, "points3d.ply")
#     if not os.path.exists(ply_path) and len(model_files) > 0:
#         print("Generating point cloud from SMPL-X model...")
#         model_file_path = os.path.join(models_dir, model_files[0])
#         # model_dict = np.load(model_file_path)
        
#         # # 获取最小形状顶点
#         # if 'minimal_shape' in model_dict:
#         #     vertices = model_dict['minimal_shape'].astype(np.float32)
#         # elif 'vertices' in model_dict:
#         #     vertices = model_dict['vertices'].astype(np.float32)
#         # else:
#         #     # 如果没有顶点，创建随机点云
#         #     print("No vertices found in SMPL-X model, using random point cloud")
#         #     num_pts = 10000
#         #     vertices = np.random.random((num_pts, 3)) * 2.0 - 1.0
        
#         vertices=load_smplx_vertices_from_npz(model_file_path,'./models/smplx/')
        
#         # 为顶点分配颜色（皮肤色）
#         num_pts = len(vertices)
#         colors = np.tile([200, 150, 100], (num_pts, 1))  # 皮肤色
        
#         storePly(ply_path, vertices, colors)
    
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None
    
#     scene_info = SceneInfo(
#         point_cloud=pcd,
#         train_cameras=train_cam_infos,
#         test_cameras=test_cam_infos,
#         nerf_normalization=nerf_normalization,
#         ply_path=ply_path,
#         is_nerf_synthetic=False
#     )
    
#     return scene_info

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
    

    
    return vertices

#SUMO
def readMultiFrameNerfSyntheticCameras(path, transformsfile, depths_folder, white_background, 
                                        is_test, extension=".jpg", start_frame=0, end_frame=9999):
    """
    读取多帧NeRF Synthetic数据集的相机参数
    所有帧的数据都在一个transforms文件中
    
    Args:
        path: 数据集根目录
        transformsfile: transforms文件名（如 "transforms_train.json"）
        depths_folder: 深度图文件夹路径
        white_background: 是否使用白色背景
        is_test: 是否为测试集
        extension: 图像扩展名
        start_frame: 起始帧
        end_frame: 结束帧
    """
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        
        # 按帧分组
        frame_groups = {}
        for frame_data in frames:
            file_path = frame_data["file_path"]
            # 提取帧编号，如 "/000000/images/001.jpg" -> "000000"
            frame_id_str = file_path.split('/')[1] if file_path.startswith('/') else file_path.split('/')[0]
            
            # 转换为整数帧ID
            try:
                frame_id = int(frame_id_str)
            except ValueError:
                print(f"Warning: Cannot parse frame ID from path: {file_path}")
                continue
            
            # 过滤帧范围
            if frame_id < start_frame or frame_id > end_frame:
                continue
            
            if frame_id not in frame_groups:
                frame_groups[frame_id] = []
            frame_groups[frame_id].append(frame_data)
        
        print(f"Found {len(frame_groups)} frames in range [{start_frame}, {end_frame}]")
        
        # 处理每一帧
        total_frames = len(frame_groups)
        for frame_id in sorted(frame_groups.keys()):
            frame_list = frame_groups[frame_id]
            timecode = float(frame_id) / max(total_frames, 1) if total_frames > 1 else 0.0
            
            # 如果JSON中有time字段，优先使用
            if len(frame_list) > 0 and "time" in frame_list[0]:
                timecode = frame_list[0]["time"]
            
            # 查找该帧的deformer_path
            frame_folder_name = f"{frame_id:06d}"  # 格式化为000000, 000001等
            frame_folder_path = os.path.join(path, frame_folder_name)
            frame_deformer = os.path.join(frame_folder_path, "transforms.json")
            
            deformer_path = frame_deformer if os.path.exists(frame_deformer) else None
            
            print(f"Processing frame {frame_id} (timecode: {timecode:.3f}, {len(frame_list)} cameras, deformer: {deformer_path is not None})")
            
            # 处理该帧的所有相机
            for idx, frame_data in enumerate(frame_list):
                file_path = frame_data["file_path"]
                
                # 处理路径：移除开头的'/'
                if file_path.startswith('/'):
                    file_path = file_path[1:]
                
                # 构建完整路径
                cam_name = file_path if file_path.endswith(extension) else file_path + extension
                image_path = os.path.join(path, cam_name)
                image_name = os.path.basename(cam_name)
                
                # NeRF 'transform_matrix' 是相机到世界的变换
                c2w = np.array(frame_data["transform_matrix"])
                # 从OpenGL/Blender坐标系转换到COLMAP坐标系
                c2w[:3, 1:3] *= -1

                # 获取世界到相机的变换
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])  # R存储为转置（因为CUDA代码中的glm）
                T = w2c[:3, 3]

                # 读取图像
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                    
                image = Image.open(image_path)

                # 处理RGBA图像
                if image.mode == 'RGBA':
                    im_data = np.array(image.convert("RGBA"))
                    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
                    norm_data = im_data / 255.0
                    arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                    image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")
                elif image.mode != 'RGB':
                    image = image.convert("RGB")

                # 计算FOV
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx

                # 深度图路径
                depth_path = ""
                if depths_folder != "":
                    # 深度图与图像对应
                    depth_full_path = os.path.join(depths_folder, file_path.replace(extension, ".png"))
                    if os.path.exists(depth_full_path):
                        depth_path = depth_full_path

                # 生成唯一ID
                uid = frame_id * 10000 + idx
                
                # 背景图路径
                bg_path = image_path.replace('/images/', '/bg/')
                if not os.path.exists(bg_path):
                    bg_path = None
                
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    bg_path=bg_path,
                    width=image.size[0],
                    height=image.size[1],
                    depth_path=depth_path,
                    depth_params=None,
                    is_test=is_test
                )
                
                # 添加多帧相关信息
                cam_info.kid = frame_id
                cam_info.timecode = timecode
                cam_info.deformer_path = deformer_path  # 设置该帧的deformer路径
                
                cam_infos.append(cam_info)

    return cam_infos


def readMultiFrameNerfSyntheticInfo(path, images, depths, eval, white_background, 
                                     extension=".jpg", start_frame=0, end_frame=9999,
                                     train_test_exp=False):
    """
    读取多帧NeRF Synthetic数据集
    
    Args:
        path: 数据集根目录
        images: 图像文件夹（未使用，保持接口一致）
        depths: 深度图文件夹名称
        eval: 是否为评估模式
        white_background: 是否使用白色背景
        extension: 图像扩展名
        start_frame: 起始帧
        end_frame: 结束帧
        train_test_exp: 是否扩展训练集（包含测试集）
    """
    depths_folder = os.path.join(path, depths) if depths != "" else ""
    
    print("="*60)
    print("Reading Multi-Frame NeRF Synthetic Dataset")
    print("="*60)
    
    # 读取训练相机
    print("\nReading Training Transforms (All Frames)...")
    train_cam_infos = readMultiFrameNerfSyntheticCameras(
        path, "transforms_train.json", depths_folder, white_background, 
        False, extension, start_frame, end_frame
    )
    
    # 读取测试相机
    print("\nReading Test Transforms (All Frames)...")
    test_cam_infos = readMultiFrameNerfSyntheticCameras(
        path, "transforms_test.json", depths_folder, white_background,
        True, extension, start_frame, end_frame
    )
    
    # 评估模式处理
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    
    # 如果需要扩展训练集
    if train_test_exp and eval:
        all_cam_infos = train_cam_infos + test_cam_infos
        train_cam_infos = all_cam_infos
        test_cam_infos = [c for c in all_cam_infos if c.is_test]

    # 读取或生成点云
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"\nGenerating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print(f"Warning: Failed to load point cloud: {e}")
        pcd = None

    # 计算归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)

    print("\n" + "="*60)
    print(f"Multi-Frame NeRF Synthetic Loading Complete")
    print(f"Total train cameras: {len(train_cam_infos)}")
    print(f"Total test cameras: {len(test_cam_infos)}")
    
    # 统计每帧的相机数量
    train_frames = {}
    for cam in train_cam_infos:
        train_frames[cam.kid] = train_frames.get(cam.kid, 0) + 1
    print(f"Train frames: {len(train_frames)} frames")
    
    test_frames = {}
    for cam in test_cam_infos:
        test_frames[cam.kid] = test_frames.get(cam.kid, 0) + 1
    print(f"Test frames: {len(test_frames)} frames")
    
    # 统计deformer使用情况
    deformer_count = sum(1 for cam in train_cam_infos + test_cam_infos if cam.deformer_path is not None)
    print(f"Cameras with deformer: {deformer_count}/{len(train_cam_infos) + len(test_cam_infos)}")
    print("="*60 + "\n")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=True
    )
    
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Deform":readDeformSceneInfo,
    "MultiFrameNerfSynthetic": readMultiFrameNerfSyntheticInfo,
}