"""
Script to preprocess ETH3D dataset for benchmarking
"""

import argparse
import os

import cv2
import numpy as np
import PIL.Image
import pycolmap
from natsort import natsorted
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def load_eth3d_raw_image(path):
    "Function to load ETH3D raw image"
    img = Image.open(path)
    return img


def resize_pil_image(img, long_edge_size):
    "Function to resize PIL image"
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_eth3d_raw_depth(path):
    "Function to load ETH3D raw depth"
    height, width = 4032, 6048
    depth = np.fromfile(path, dtype=np.float32).reshape(height, width)
    depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0, nan=0.0)
    depth = np.expand_dims(depth, -1)  # (H, W, 1)
    return depth


def pose_matrix_from_quaternion(pvec):
    """
    Get 4x4 pose matrix from quaternion (t, q)
    t = (tx, ty, tz)
    q = (qw, qx, qy, qz)
    """
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_quat(pvec[3:], scalar_first=True).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose


def resize_image_and_adjust_camera_params_step1(rgb_image, depth_image, camera_params, target_size):
    """
    Resize RGB and depth images and adjust camera parameters for step 1.
    """
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = camera_params
    # Determine the scale factor
    w, h = rgb_image.size
    scale = target_size / max(w, h)
    # Resize images
    new_size = (int(round(w * scale)), int(round(h * scale)))
    long_edge_size = max(new_size)
    resized_rgb = resize_pil_image(rgb_image, long_edge_size)
    resized_depth = cv2.resize(depth_image, new_size, interpolation=cv2.INTER_NEAREST)
    # Adjust camera parameters
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = cx * scale
    new_cy = cy * scale
    adjusted_camera_params = [new_fx, new_fy, new_cx, new_cy, k1, k2, p1, p2, k3, k4, sx1, sy1]
    return resized_rgb, resized_depth, adjusted_camera_params


def resize_image_and_adjust_camera_params_step2(rgb_image, depth_image, camera_params, target_size):
    """
    Resize RGB and depth images and adjust camera parameters for step 2.
    """
    fx, fy, cx, cy = camera_params[:4]
    # Determine the scale factor
    w, h = rgb_image.size
    scale = target_size / max(w, h)
    # Resize images
    new_size = (int(round(w * scale)), int(round(h * scale)))
    long_edge_size = max(new_size)
    resized_rgb = resize_pil_image(rgb_image, long_edge_size)
    resized_depth = cv2.resize(depth_image, new_size, interpolation=cv2.INTER_NEAREST)
    # Adjust camera parameters
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = cx * scale
    new_cy = cy * scale
    if len(camera_params) == 9:
        adjusted_camera_params = [new_fx, new_fy, new_cx, new_cy] + camera_params[4:]
    elif len(camera_params) == 4:
        adjusted_camera_params = [new_fx, new_fy, new_cx, new_cy]
    else:
        raise ValueError("Invalid number of camera params")
    return resized_rgb, resized_depth, adjusted_camera_params


def process_step1(root, out_root, target_size=600):
    """Process step 1: basic preprocessing"""
    scene_list = natsorted(os.listdir(root))
    for scene_name in tqdm(scene_list):
        scene_folder = os.path.join(root, scene_name)
        # Read images.txt
        images_txt_path = os.path.join(scene_folder, "dslr_calibration_jpg", "images.txt")
        with open(images_txt_path, "r") as f:
            lines = f.readlines()[4:]  # Skip header
        # Read cameras.txt
        cameras_txt_path = os.path.join(scene_folder, "dslr_calibration_jpg", "cameras.txt")
        with open(cameras_txt_path, "r") as f:
            camera_lines = f.readlines()[3:]  # Skip header
        # Create camera_id to camera_params mapping
        camera_params_dict = {}
        for line in camera_lines:
            parts = line.strip().split()
            camera_id = int(parts[0])
            params = list(map(float, parts[4:]))
            camera_params_dict[camera_id] = params
        # Create output directories
        out_scene_folder = os.path.join(out_root, scene_name)
        os.makedirs(os.path.join(out_scene_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_scene_folder, "depths"), exist_ok=True)
        os.makedirs(os.path.join(out_scene_folder, "camera_params"), exist_ok=True)
        os.makedirs(os.path.join(out_scene_folder, "poses"), exist_ok=True)
        # Process each image
        for i in tqdm(range(0, len(lines), 2)):
            parts = lines[i].strip().split()
            _, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts[:10]
            camera_id = int(camera_id)
            # Load image and depth map
            image_path = os.path.join(scene_folder, "images", name)
            depth_path = os.path.join(scene_folder, "ground_truth_depth", name)
            rgb_image = load_eth3d_raw_image(image_path)
            depth_image = load_eth3d_raw_depth(depth_path)
            # Get camera pose and parameters
            pose = pose_matrix_from_quaternion([tx, ty, tz, qw, qx, qy, qz])
            camera_params = camera_params_dict[camera_id]
            # Resize image and depth map
            resized_rgb, resized_depth, adjusted_camera_params = resize_image_and_adjust_camera_params_step1(
                rgb_image, depth_image, camera_params, target_size
            )
            # Save resized image, depth map, pose, and camera parameters
            file_name = name.split(".")[0].split("/")[-1]
            resized_image_path = os.path.join(out_scene_folder, "images", f"{file_name}.jpg")
            resized_depth_path = os.path.join(out_scene_folder, "depths", f"{file_name}.npy")
            pose_path = os.path.join(out_scene_folder, "poses", f"{file_name}.npy")
            camera_params_path = os.path.join(out_scene_folder, "camera_params", f"{file_name}.npy")
            # Save the resized RGB image
            resized_rgb.save(resized_image_path)
            # Save the resized depth map
            np.save(resized_depth_path, resized_depth)
            # Save the camera pose
            np.save(pose_path, pose)
            # Save the adjusted camera parameters
            np.save(camera_params_path, adjusted_camera_params)


def process_step2(root, out_root, target_size=600):
    """Process step 2: undistortion and preprocessing"""
    scene_list = natsorted(os.listdir(root))
    for scene_name in tqdm(scene_list):
        scene_folder = os.path.join(root, scene_name)

        print("Computing the undistortion mapping ...")
        # Read cameras.txt for distorted cameras
        distorted_cameras_txt_path = os.path.join(scene_folder, "dslr_calibration_jpg", "cameras.txt")
        with open(distorted_cameras_txt_path, "r") as f:
            distorted_camera_lines = f.readlines()[3:]  # Skip header

        # Create camera_id to camera_params mapping for distorted cameras
        distorted_camera_params_dict = {}
        for line in distorted_camera_lines:
            parts = line.strip().split()
            distorted_camera_id = int(parts[0])
            distorted_params = list(map(float, parts[4:]))
            distorted_camera_params_dict[distorted_camera_id] = pycolmap.Camera(
                model="THIN_PRISM_FISHEYE", width=int(parts[2]), height=int(parts[3]), params=distorted_params
            )

        # Read cameras.txt for undistorted cameras
        undistorted_cameras_txt_path = os.path.join(scene_folder, "dslr_calibration_undistorted", "cameras.txt")
        with open(undistorted_cameras_txt_path, "r") as f:
            undistorted_camera_lines = f.readlines()[3:]  # Skip header

        # Create camera_id to camera_params mapping for undistorted cameras
        undistorted_camera_params_dict = {}
        for line in undistorted_camera_lines:
            parts = line.strip().split()
            undistorted_camera_id = int(parts[0])
            undistorted_params = list(map(float, parts[4:]))
            undistorted_camera_params_dict[undistorted_camera_id] = pycolmap.Camera(
                model="PINHOLE", width=int(parts[2]), height=int(parts[3]), params=undistorted_params
            )

        # Precompute distorted image coordinates for each camera ID
        distorted_img_coords_dict = {}
        for camera_id, undistorted_camera in undistorted_camera_params_dict.items():
            # Generate image points grid
            height, width = undistorted_camera.height, undistorted_camera.width
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            image_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

            # Compute world points from image points using the undistorted camera
            pinhole_world_pts = undistorted_camera.cam_from_img(image_points)

            # Get the distorted camera
            distorted_camera = distorted_camera_params_dict[camera_id]

            # Project world points to distorted image plane
            distorted_img_coords = distorted_camera.img_from_cam(pinhole_world_pts)
            distorted_img_coords = np.clip(distorted_img_coords, 0, [width - 1, height - 1])
            distorted_img_coords = distorted_img_coords.astype(int)

            # Store the precomputed coordinates
            distorted_img_coords_dict[camera_id] = distorted_img_coords

        # Read images.txt for distorted images
        distorted_images_txt_path = os.path.join(scene_folder, "dslr_calibration_jpg", "images.txt")
        with open(distorted_images_txt_path, "r") as f:
            distorted_image_lines = f.readlines()[4:]  # Skip header

        # Process each image
        print("Undistorting the depth maps ...")
        os.makedirs(os.path.join(scene_folder, "ground_truth_depth", "dslr_images_undistorted"), exist_ok=True)
        for i in tqdm(range(0, len(distorted_image_lines), 2)):
            parts = distorted_image_lines[i].strip().split()
            _, _, _, _, _, _, _, _, camera_id, image_name = parts[:10]
            camera_id = int(camera_id)

            # Load the corresponding depth map
            depth_map_path = os.path.join(scene_folder, "ground_truth_depth", image_name)
            depth_map = load_eth3d_raw_depth(depth_map_path)

            # Retrieve precomputed distorted image coordinates
            distorted_img_coords = distorted_img_coords_dict[camera_id]

            # Get undistorted depth values
            undistorted_depth = depth_map[distorted_img_coords[:, 1], distorted_img_coords[:, 0]]

            # Fetch the height and width specific to the current camera_id
            undistorted_camera = undistorted_camera_params_dict[camera_id]
            height, width = undistorted_camera.height, undistorted_camera.width

            # Save the undistorted depth map
            saved_depth_name = image_name.replace("dslr_images", "dslr_images_undistorted")
            saved_depth_name = saved_depth_name.replace(".JPG", ".npy")
            undistorted_depth_path = os.path.join(scene_folder, "ground_truth_depth", saved_depth_name)
            np.save(undistorted_depth_path, undistorted_depth.reshape(height, width))

        print("Resizing the undistorted data and saving it to output directory ...")
        # Read images.txt
        images_txt_path = os.path.join(scene_folder, "dslr_calibration_undistorted", "images.txt")
        with open(images_txt_path, "r") as f:
            lines = f.readlines()[4:]  # Skip header
        # Read cameras.txt
        cameras_txt_path = os.path.join(scene_folder, "dslr_calibration_undistorted", "cameras.txt")
        with open(cameras_txt_path, "r") as f:
            camera_lines = f.readlines()[3:]  # Skip header
        # Create camera_id to camera_params mapping
        camera_params_dict = {}
        for line in camera_lines:
            parts = line.strip().split()
            camera_id = int(parts[0])
            params = list(map(float, parts[4:]))
            camera_params_dict[camera_id] = params
        # Create output directories
        out_scene_folder = os.path.join(out_root, scene_name)
        os.makedirs(os.path.join(out_scene_folder, "undistorted_images"), exist_ok=True)
        os.makedirs(os.path.join(out_scene_folder, "undistorted_depths"), exist_ok=True)
        os.makedirs(os.path.join(out_scene_folder, "undistorted_camera_params"), exist_ok=True)
        # Process each image
        for i in tqdm(range(0, len(lines), 2)):
            parts = lines[i].strip().split()
            _, _, _, _, _, _, _, _, camera_id, name = parts[:10]
            camera_id = int(camera_id)
            # Load image and depth map
            image_path = os.path.join(scene_folder, "images", name)
            depth_path = os.path.join(scene_folder, "ground_truth_depth", name.replace(".JPG", ".npy"))
            rgb_image = load_eth3d_raw_image(image_path)
            depth_image = np.load(depth_path)
            # Get camera parameters
            camera_params = camera_params_dict[camera_id]
            # Resize image and depth map
            resized_rgb, resized_depth, adjusted_camera_params = resize_image_and_adjust_camera_params_step2(
                rgb_image, depth_image, camera_params, target_size
            )
            # Save resized image, depth map, pose, and camera parameters
            file_name = name.split(".")[0].split("/")[-1]
            resized_image_path = os.path.join(out_scene_folder, "undistorted_images", f"{file_name}.jpg")
            resized_depth_path = os.path.join(out_scene_folder, "undistorted_depths", f"{file_name}.npy")
            camera_params_path = os.path.join(out_scene_folder, "undistorted_camera_params", f"{file_name}.npy")
            # Save the resized RGB image
            resized_rgb.save(resized_image_path)
            # Save the resized depth map
            np.save(resized_depth_path, resized_depth)
            # Save the adjusted camera parameters
            np.save(camera_params_path, adjusted_camera_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path",
        type=str,
        help="Path to the raw ETH3D training dataset",
        default="/home/inf/UFM/data/eth3d_download",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Path to the output directory",
        default="/home/inf/UFM/data/eth3d_processed",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["step1", "step2", "both"],
        default="both",
        help="Which processing step to run",
    )
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    print("Processing ETH3D dataset...")

    if args.step == "step1" or args.step == "both":
        print("Running step 1...")
        process_step1(args.in_path, args.out_path)

    if args.step == "step2" or args.step == "both":
        print("Running step 2...")
        process_step2(args.in_path, args.out_path)

    print("Processing done.")
