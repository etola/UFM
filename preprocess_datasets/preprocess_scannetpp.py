#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the scannet++ dataset.
# Usage:
# python3 datasets_preprocess/preprocess_scannetpp.py --scannetpp_dir /path/to/scannetpp --precomputed_pairs /path/to/scannetpp_pairs --pyopengl-platform egl
# --------------------------------------------------------
import argparse
import json
import os
import os.path as osp
import re

import cv2
import numpy as np
import PIL.Image as Image
import pyrender
import trimesh
import trimesh.exchange.ply
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import uniflowmatch.utils.geometry as geometry
from uniflowmatch.datasets.utils.cropping import rescale_image_depthmap

inv = np.linalg.inv
norm = np.linalg.norm
REGEXPR_DSLR = re.compile(r"^DSC(?P<frameid>\d+).JPG$")
REGEXPR_IPHONE = re.compile(r"frame_(?P<frameid>\d+).jpg$")

DEBUG_VIZ = None  # 'iou'
if DEBUG_VIZ is not None:
    import matplotlib.pyplot as plt  # noqa


OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannetpp_dir", type=str, default="/jet/home/yzhang25/match_anything/data/scannet++_v2")
    parser.add_argument(
        "--precomputed_anymap_pairs",
        type=str,
        default="/jet/home/yzhang25/match_anything/data/anymap_dataset_metadata/train/snpp_aggregated_metadata_train.npz",
    )
    parser.add_argument(
        "--output_dir", default="/jet/home/yzhang25/match_anything/data/scannetpp_anymap_processed/train"
    )
    parser.add_argument("--target_resolution", default=980, type=int, help="images resolution")
    parser.add_argument("--pyopengl-platform", type=str, default="", help="PyOpenGL env variable")
    return parser


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world


def get_frame_number(name, cam_type="dslr"):

    name = name.split("_")[-1]

    if cam_type == "dslr":
        regex_expr = REGEXPR_DSLR
    elif cam_type == "iphone":
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f"wrong {cam_type=} for get_frame_number")
    matches = re.match(regex_expr, name)
    return matches["frameid"]


def load_sfm(sfm_dir, cam_type="dslr"):
    # load cameras
    with open(osp.join(sfm_dir, "cameras.txt"), "r") as f:
        raw = f.read().splitlines()[3:]  # skip header

    intrinsics = {}
    for camera in tqdm(raw, position=1, leave=False):
        camera = camera.split(" ")
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]

    # load images
    with open(os.path.join(sfm_dir, "images.txt"), "r") as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith("#")]  # skip header

    img_idx = {}
    img_infos = {}
    for image, points in tqdm(zip(raw[0::2], raw[1::2]), total=len(raw) // 2, position=1, leave=False):
        image = image.split(" ")
        points = points.split(" ")

        idx = image[0]
        img_name = image[-1]
        assert img_name not in img_idx, "duplicate db image: " + img_name
        img_idx[img_name] = idx  # register image name

        current_points2D = {
            int(i): (float(x), float(y)) for i, x, y in zip(points[2::3], points[0::3], points[1::3]) if i != "-1"
        }
        img_infos[idx] = dict(
            intrinsics=intrinsics[int(image[-2])],
            path=img_name,
            frame_id=get_frame_number(img_name, cam_type),
            cam_to_world=pose_from_qwxyz_txyz(image[1:-2]),
            sparse_pts2d=current_points2D,
        )

    # load 3D points
    with open(os.path.join(sfm_dir, "points3D.txt"), "r") as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith("#")]  # skip header

    points3D = {}
    observations = {idx: [] for idx in img_infos.keys()}
    for point in tqdm(raw, position=1, leave=False):
        point = point.split()
        point_3d_idx = int(point[0])
        points3D[point_3d_idx] = tuple(map(float, point[1:4]))
        if len(point) > 8:
            for idx, point_2d_idx in zip(point[8::2], point[9::2]):
                observations[idx].append((point_3d_idx, int(point_2d_idx)))

    return img_idx, img_infos, points3D, observations


def subsample_img_infos(img_infos, num_images, allowed_name_subset=None):
    img_infos_val = [(idx, val) for idx, val in img_infos.items()]
    if allowed_name_subset is not None:
        img_infos_val = [(idx, val) for idx, val in img_infos_val if val["path"] in allowed_name_subset]

    if len(img_infos_val) > num_images:
        img_infos_val = sorted(img_infos_val, key=lambda x: x[1]["frame_id"])
        kept_idx = np.round(np.linspace(0, len(img_infos_val) - 1, num_images)).astype(int).tolist()
        img_infos_val = [img_infos_val[idx] for idx in kept_idx]
    return {idx: val for idx, val in img_infos_val}


def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = geometry.colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 1, (width, height), True)
        map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

    undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    undistorted_mask = cv2.remap(
        mask, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    new_K = geometry.opencv_to_colmap_intrinsics(new_K)
    return width, height, new_K, undistorted_image, undistorted_mask


def adjacency_list_to_array(adj_list):
    edges = set()  # Use a set to store unique edges
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            edge = tuple(sorted((node, neighbor)))  # Ensure (a, b) and (b, a) are treated the same
            edges.add(edge)

    return np.array(list(edges))  # Convert set to NumPy array


def process_scenes(root, pairs_npz, output_dir, target_resolution, rank=0, world_size=1):
    os.makedirs(output_dir, exist_ok=True)

    pairs_data = np.load(pairs_npz, allow_pickle=True)
    pairs_data = {k.split("/")[-1]: v for k, v in pairs_data.items()}

    pairs_keys_list = sorted(list(pairs_data.keys()))

    # default values from
    # https://github.com/scannetpp/scannetpp/blob/main/common/configs/render.yml
    znear = 0.05
    zfar = 20.0

    pairs_keys_list = np.array_split(pairs_keys_list, world_size)[rank]

    renderer = pyrender.OffscreenRenderer(0, 0)
    for scene_id in pairs_keys_list:
        # Scannetpp scene dir
        data_dir = os.path.join(root, "data", scene_id)
        dir_dslr = os.path.join(data_dir, "dslr")
        dir_iphone = os.path.join(data_dir, "iphone")
        dir_scans = os.path.join(data_dir, "scans")

        # The images we need to process, according to the AnyMap pairs
        scene_pairs = pairs_data[scene_id]
        adjacency_list = scene_pairs.item()["adjacency_list"]
        file_names_original = scene_pairs.item()["file_names"]
        total_number_of_edges = scene_pairs.item()["total_number_of_edges"]

        file_names = [x.split("_")[-1] for x in file_names_original]

        # set up the output paths
        output_dir_scene = os.path.join(output_dir, scene_id)

        output_dir_scene_rgb = os.path.join(output_dir_scene, "images")
        output_dir_scene_depth = os.path.join(output_dir_scene, "depth")
        os.makedirs(output_dir_scene_rgb, exist_ok=True)
        os.makedirs(output_dir_scene_depth, exist_ok=True)

        ply_path = os.path.join(dir_scans, "mesh_aligned_0.05.ply")

        sfm_dir_dslr = os.path.join(dir_dslr, "colmap")
        rgb_dir_dslr = os.path.join(dir_dslr, "resized_images")
        mask_dir_dslr = os.path.join(dir_dslr, "resized_anon_masks")

        try:
            # load the mesh
            with open(ply_path, "rb") as f:
                mesh_kwargs = trimesh.exchange.ply.load_ply(f)
            mesh_scene = trimesh.Trimesh(**mesh_kwargs)

            # read colmap reconstruction, we will only use the intrinsics and pose here

            img_idx_dslr, img_infos_dslr, points3D_dslr, observations_dslr = load_sfm(sfm_dir_dslr, cam_type="dslr")

            dslr_paths = {
                "in_colmap": sfm_dir_dslr,
                "in_rgb": rgb_dir_dslr,
                "in_mask": mask_dir_dslr,
            }

            # img_idx_iphone, img_infos_iphone, points3D_iphone, observations_iphone = load_sfm(
            #     sfm_dir_iphone, cam_type="iphone"
            # )

        except FileNotFoundError:
            print(f"Skipping {scene_id} as colmap reconstruction not found")
            continue

        mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False)
        pyrender_scene = pyrender.Scene()
        pyrender_scene.add(mesh)

        selection_dslr = [x.split(".")[0] for x in file_names_original]

        selection_cam = selection_dslr
        img_idx = {k.split(".")[0]: v for k, v in img_idx_dslr.items()}
        img_infos = img_infos_dslr
        paths_data = dslr_paths

        # resize the image to a more manageable size and render depth
        rgb_dir = paths_data["in_rgb"]
        mask_dir = paths_data["in_mask"]

        intrinsics_undistorted = []
        for imgname in tqdm(selection_cam, position=1, leave=False):
            imgidx = img_idx[imgname]
            img_infos_idx = img_infos[imgidx]
            rgb = np.array(Image.open(os.path.join(rgb_dir, img_infos_idx["path"])))
            mask = np.array(Image.open(os.path.join(mask_dir, img_infos_idx["path"][:-3] + "png")))

            _, _, K, rgb, mask = undistort_images(img_infos_idx["intrinsics"], rgb, mask)

            # rescale_image_depthmap assumes opencv intrinsics
            intrinsics = geometry.colmap_to_opencv_intrinsics(K)
            image, mask, intrinsics = rescale_image_depthmap(
                rgb, mask, intrinsics, (target_resolution, target_resolution * 3.0 / 4)
            )

            W, H = image.size

            # saving intrinsics in opencv format, so we disable the conversion
            # intrinsics = geometry.opencv_to_colmap_intrinsics(intrinsics)

            # save the intrinsics
            intrinsics_undistorted.append(intrinsics)

            # update inpace img_infos_idx
            img_infos_idx["intrinsics"] = intrinsics
            rgb_outpath = os.path.join(output_dir_scene_rgb, img_infos_idx["path"][:-3] + "jpg")
            image.save(rgb_outpath)

            depth_outpath = os.path.join(output_dir_scene_depth, img_infos_idx["path"][:-3] + "png")
            # render depth image
            renderer.viewport_width, renderer.viewport_height = W, H
            fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
            camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=znear, zfar=zfar)
            camera_node = pyrender_scene.add(camera, pose=img_infos_idx["cam_to_world"] @ OPENGL_TO_OPENCV)

            depth = renderer.render(pyrender_scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            pyrender_scene.remove_node(camera_node)  # dont forget to remove camera

            depth = (depth * 1000).astype("uint16")
            # invalidate depth from mask before saving
            depth_mask = mask < 255
            depth[depth_mask] = 0
            Image.fromarray(depth).save(depth_outpath)

        # nerfstudio_json = osp.join(dir_dslr, "nerfstudio", "transforms.json")

        # extrinsics in COLMAP IS from nerfstudio. It is not necessary to load it from json.
        # with open(nerfstudio_json, "r") as f:
        #     nerfstudio_data = json.load(f)

        # extrinsics = {x["file_path"].split(".")[0] : np.array(x["transform_matrix"]) for x in nerfstudio_data["frames"]}
        trajectories = []
        # intrinsics = []
        for imgname in selection_dslr:
            imgidx = img_idx_dslr[imgname + ".JPG"]
            img_infos_idx = img_infos_dslr[imgidx]

            # intrinsics_ = img_infos_idx["intrinsics"]
            cam_to_world_ = img_infos_idx["cam_to_world"]

            # intrinsics.append(intrinsics_)
            trajectories.append(cam_to_world_)

        intrinsics_undistorted = np.stack(intrinsics_undistorted, axis=0)
        trajectories = np.stack(trajectories, axis=0)

        # unroll the adjacency lists into pairs

        # pairs[:, 0] = index_map[pairs[:, 0].astype(np.int64)]
        # pairs[:, 1] = index_map[pairs[:, 1].astype(np.int64)]
        pairs = adjacency_list_to_array(adjacency_list)

        # save metadata for this scene
        scene_metadata_path = osp.join(output_dir_scene, "scene_metadata.npz")
        np.savez(
            scene_metadata_path,
            trajectories=trajectories,
            intrinsics=intrinsics_undistorted,
            images=selection_cam,
            pairs=pairs,
        )

        del img_infos
        del pyrender_scene

    # concat all scene_metadata.npz into a single file
    scene_data = {}
    for scene_subdir in pairs_keys_list:
        scene_metadata_path = osp.join(output_dir, scene_subdir, "scene_metadata.npz")

        if not osp.exists(scene_metadata_path):
            print(f"Scene metadata not found for {scene_subdir}, skipping")
            continue

        with np.load(scene_metadata_path) as data:
            trajectories = data["trajectories"]
            intrinsics = data["intrinsics"]
            images = data["images"]
            pairs = data["pairs"]
        scene_data[scene_subdir] = {
            "trajectories": trajectories,
            "intrinsics": intrinsics,
            "images": images,
            "pairs": pairs,
        }

    offset = 0
    counts = []
    scenes = []
    sceneids = []
    images = []
    intrinsics = []
    trajectories = []
    pairs = []
    for scene_idx, (scene_subdir, data) in enumerate(scene_data.items()):
        num_imgs = data["images"].shape[0]
        img_pairs = data["pairs"]

        scenes.append(scene_subdir)
        sceneids.extend([scene_idx] * num_imgs)

        images.append(data["images"])

        intrinsics.append(data["intrinsics"])
        trajectories.append(data["trajectories"])

        # offset pairs
        img_pairs[:, 0:2] += offset
        pairs.append(img_pairs)
        counts.append(offset)

        offset += num_imgs

    images = np.concatenate(images, axis=0)
    intrinsics = np.concatenate(intrinsics, axis=0)
    trajectories = np.concatenate(trajectories, axis=0)
    pairs = np.concatenate(pairs, axis=0)
    np.savez(
        osp.join(output_dir, "all_metadata.npz"),
        counts=counts,
        scenes=scenes,
        sceneids=sceneids,
        images=images,
        intrinsics=intrinsics,
        trajectories=trajectories,
        pairs=pairs,
    )
    print("all done")


if __name__ == "__main__":
    parser = get_parser()

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)

    args = parser.parse_args()

    os.environ["PYOPENGL_PLATFORM"] = "egl"

    if args.pyopengl_platform.strip():
        os.environ["PYOPENGL_PLATFORM"] = args.pyopengl_platform
    process_scenes(
        args.scannetpp_dir,
        args.precomputed_anymap_pairs,
        args.output_dir,
        args.target_resolution,
        args.rank,
        args.world_size,
    )
