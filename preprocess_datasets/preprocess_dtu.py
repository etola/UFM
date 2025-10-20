# Modified based on RobustMVD Benchmark & TransMVSNet
#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import re

import cv2
import numpy as np
import PIL.Image
from natsort import natsorted
from PIL import Image
from tqdm import tqdm


def cp(a, b, verbose=True, followLinks=False):
    os.system('cp -r %s %s "%s" "%s"' % ("-v" if verbose else "", "-L" if followLinks else "", a, b))


def copy_rectified_images(in_base, out_base):
    in_base = osp.join(in_base, "Rectified")
    scans = os.listdir(in_base)

    for scan in tqdm(scans, "Processed scans"):
        in_path = osp.join(in_base, scan)
        out_path = osp.join(out_base, scan)
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "images")
        os.makedirs(out_path, exist_ok=True)

        images = sorted([x for x in os.listdir(in_path) if x.endswith("_3_r5000.png")])

        for idx, image in enumerate(images):
            image_in = osp.join(in_path, image)
            image_out = osp.join(out_path, "{:08d}.png".format(idx))
            cp(image_in, image_out)


def copy_gt_depths(in_base, out_base):
    in_base = osp.join(in_base, "dtu", "Depths_raw")
    scans = os.listdir(in_base)

    for scan in tqdm(scans, "Processed scans"):
        in_path = osp.join(in_base, scan)
        out_path = osp.join(out_base, scan)
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "depths")
        os.makedirs(out_path, exist_ok=True)

        depths = sorted([x for x in os.listdir(in_path) if x.endswith(".pfm")])

        for idx, depth in enumerate(depths):
            if idx > 48:
                break
            depth_in = osp.join(in_path, depth)
            depth_out = osp.join(out_path, "{:08d}.pfm".format(idx))
            cp(depth_in, depth_out)


def copy_cams(in_base, out_base):
    scans = os.listdir(osp.join(in_base, "Rectified"))
    in_base = osp.join(in_base, "dtu", "Cameras_1")

    for scan in tqdm(scans, "Processed scans"):
        in_path = in_base
        out_path = osp.join(out_base, scan)
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "cams")
        os.makedirs(out_path, exist_ok=True)

        cams = sorted([x for x in os.listdir(in_path) if x.endswith(".txt")])

        for idx, cam in enumerate(cams):
            if idx > 48:
                break
            cam_in = osp.join(in_path, cam)
            cam_out = osp.join(out_path, "{:08d}.txt".format(idx))
            cp(cam_in, cam_out)


def copy_points(in_base, out_base):
    in_base = osp.join(in_base, "Points", "stl")
    scans = [x for x in os.listdir(in_base) if x.endswith(".ply")]

    for scan in tqdm(scans, "Processed scans"):
        in_path = osp.join(in_base, scan)
        scan_id = int(scan[3:6])
        out_path = osp.join(out_base, "scan{}".format(scan_id))
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "scan.ply")
        cp(in_path, out_path)


def load_image(path):
    "Function to load image"
    img = Image.open(path)
    return img


def readPFM(file):
    file = open(file, "rb")

    header = file.readline().rstrip()
    if header.decode("ascii") == "PF":
        color = True
    elif header.decode("ascii") == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = "<"
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    return data


def read_cam_file(file):
    with open(file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
    return intrinsics, extrinsics


def resize_pil_image(img, long_edge_size):
    "Function to resize PIL image"
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_image_and_adjust_camera_params(rgb_image, depth_image, camera_params, target_size):
    """
    Resize RGB and depth images and adjust camera parameters.
    :param rgb_image: Input RGB image (PIL image).
    :param depth_image: Input depth image (numpy array).
    :param camera_params: Camera parameters (numpy array of shape 3 x 3).
    :param target_size: Target size for the longest side of the image.
    :return: Resized RGB image, resized depth image, adjusted camera parameters.
    """
    # Determine the scale factor
    w, h = rgb_image.size
    scale = target_size / max(w, h)
    # Resize images
    new_size = (int(round(w * scale)), int(round(h * scale)))
    long_edge_size = max(new_size)
    resized_rgb = resize_pil_image(rgb_image, long_edge_size)
    resized_depth = cv2.resize(depth_image, new_size, interpolation=cv2.INTER_NEAREST)
    # Adjust camera parameters
    adjusted_camera_params = camera_params
    adjusted_camera_params[:2, :] = adjusted_camera_params[:2, :] * scale
    return resized_rgb, resized_depth, adjusted_camera_params


def process_dtu_dataset(in_path, out_path, target_size=600):
    """
    Process the DTU dataset by resizing images and adjusting camera parameters.
    Args:
        in_path (str): Path to the raw DTU dataset.
        out_path (str): Path to the output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    # Loop over each scan in the DTU dataset
    scans = os.listdir(os.path.join(in_path, "Rectified"))
    for scan in tqdm(scans, "Processed scans"):
        # Create output directory for the current scan
        scan_out_path = os.path.join(out_path, scan)
        os.makedirs(scan_out_path, exist_ok=True)
        # Create folders for images, depths, poses, and camera params
        images_path = os.path.join(scan_out_path, "images")
        depths_path = os.path.join(scan_out_path, "depths")
        poses_path = os.path.join(scan_out_path, "poses")
        camera_params_path = os.path.join(scan_out_path, "camera_params")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(depths_path, exist_ok=True)
        os.makedirs(poses_path, exist_ok=True)
        os.makedirs(camera_params_path, exist_ok=True)
        # Get the list of image, depth and camera files
        images_in_path = os.path.join(in_path, "Rectified", scan)
        depths_in_path = os.path.join(in_path, "dtu", "Depths_raw", scan)
        cams_in_path = os.path.join(in_path, "dtu", "Cameras_1")
        image_names = natsorted([x for x in os.listdir(images_in_path) if x.endswith("_3_r5000.png")])
        depth_names = natsorted([x for x in os.listdir(depths_in_path) if x.endswith(".pfm")])
        depth_names = depth_names[:49]
        cam_names = natsorted([x for x in os.listdir(cams_in_path) if x.endswith(".txt")])
        cam_names = cam_names[:49]
        # Loop over the necessary file numbers
        for idx, (image_name, depth_name, cam_name) in enumerate(zip(image_names, depth_names, cam_names)):
            # Load RGB image
            rgb_image_path = os.path.join(images_in_path, image_name)
            rgb_image = load_image(rgb_image_path)
            # Load depth image
            depth_image_path = os.path.join(depths_in_path, depth_name)
            depth_image = readPFM(depth_image_path)
            # Load camera txt file
            camera_txt_path = os.path.join(cams_in_path, cam_name)
            intrinsics, extrinsics = read_cam_file(camera_txt_path)
            # Resize RGB and depth images and adjust camera parameters
            resized_rgb, resized_depth, adjusted_intrinsics = resize_image_and_adjust_camera_params(
                rgb_image, depth_image, intrinsics, target_size
            )
            # Save the resized RGB, resized depth, adjusted intrinsics, and extrinsics
            resized_rgb_path = os.path.join(images_path, "{:08d}.png".format(idx))
            resized_depth_path = os.path.join(depths_path, "{:08d}.npy".format(idx))
            adjusted_intrinsics_path = os.path.join(camera_params_path, "{:08d}_intrinsics.npy".format(idx))
            extrinsics_path = os.path.join(poses_path, "{:08d}.npy".format(idx))
            resized_rgb.save(resized_rgb_path)
            np.save(resized_depth_path, resized_depth)
            np.save(adjusted_intrinsics_path, adjusted_intrinsics)
            np.save(extrinsics_path, extrinsics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path",
        type=str,
        help="Path to the raw DTU dataset",
        default="/mnt/xri_mapsresearch/data/nkeetha/dtu_raw",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Path to the output directory",
        default="/mnt/xri_mapsresearch/data/nkeetha/dtu_processed",
    )
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    # print("Copying rectified images:")
    # copy_rectified_images(args.in_path, args.out_path)

    # print("Copying GT depths:")
    # copy_gt_depths(args.in_path, args.out_path)

    # print("Copying camera poses:")
    # copy_cams(args.in_path, args.out_path)

    # print("Copying points:")
    # copy_points(args.in_path, args.out_path)

    print("Processing DTU dataset ...")
    process_dtu_dataset(args.in_path, args.out_path)

    print("Processing Done")
