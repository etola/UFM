#!/usr/bin/env python3
# --------------------------------------------------------
# Dataloader for preprocessed ETH3D dataset
# This file is copied from AnyMap repository
# Original Author: Nikhil Keetha
# --------------------------------------------------------
import os.path as osp

import numpy as np

from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from uniflowmatch.utils.image import imread_cv2


class ETH3D(BaseStereoViewDataset):
    """
    ETH3D dataset for benchmarking.
    """

    def __init__(self, *args, ROOT, pair_metadata_path, **kwargs):
        self.ROOT = ROOT
        self.pair_metadata_path = pair_metadata_path
        super().__init__(*args, **kwargs)
        self._load_data()
        self.is_metric_scale = True
        self.is_synthetic = False

    def _load_data(self):
        # Load pairs npz
        pairs_data = np.load(self.pair_metadata_path, allow_pickle=True)
        pairs = pairs_data["data"]

        # Assign pairs as class variable
        self.pairs = pairs

        # List of all scenes
        self.scenes = np.unique(self.pairs[:, 0])

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f"{len(self)} pairs from {len(self.scenes)} scenes"

    def _get_views(self, pair_idx, resolution, rng):
        # Get the pair data
        scene_name, img1_name, img2_name, score = self.pairs[pair_idx]

        # Get the scene folder
        scene_folder = osp.join(self.ROOT, scene_name)

        # Load the views
        views = []
        for view_name in [img1_name, img2_name]:
            # Load the RGB image
            img_path = osp.join(scene_folder, "undistorted_images", view_name + ".jpg")
            image = imread_cv2(img_path)
            # Load the depth data
            depth_path = osp.join(scene_folder, "undistorted_depths", view_name + ".npy")
            depthmap = np.load(depth_path).astype(np.float32)
            # Load the camera parameters
            params_path = osp.join(scene_folder, "undistorted_camera_params", view_name + ".npy")
            fx, fy, cx, cy = np.load(params_path).astype(np.float32)
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
            # Load the pose
            pose_path = osp.join(scene_folder, "poses", view_name + ".npy")
            w2c_pose = np.load(pose_path).astype(np.float32)
            c2w_pose = np.linalg.inv(w2c_pose)

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_folder, view_name)
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=c2w_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="ETH3D",
                    label=scene_name,
                    instance=view_name,
                    is_widebaseline=True,
                    is_synthetic=False,
                )
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/mnt/xri_mapsresearch/data/nkeetha/eth3d_processed", type=str)
    parser.add_argument(
        "-pmp",
        "--pair_metadata_path",
        default="/mnt/xri_mapsresearch/data/nkeetha/anymap_data_pairs/eth3d_valid_pairs.npz",
        type=str,
    )
    parser.add_argument("--viz", action="store_true")

    return parser


if __name__ == "__main__":
    import rerun as rr
    from anymap.datasets.base.base_stereo_view_dataset import view_name
    from anymap.utils.image import rgb
    from anymap.utils.viz import script_add_rerun_args

    parser = get_parser()
    script_add_rerun_args(parser)  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    dataset = ETH3D(
        ROOT=args.root_dir,
        pair_metadata_path=args.pair_metadata_path,
        resolution=(512, 384),
        seed=777,
        transform="imgnorm",
        data_norm_type="dust3r",
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, f"ETH3D_Dataloader")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    sampled_indices = np.random.choice(len(dataset), size=5, replace=False)

    for num, idx in enumerate(sampled_indices):
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        for view_idx in [0, 1]:
            image = rgb(views[view_idx]["img"], norm_type=views[view_idx]["data_norm_type"])
            depthmap = views[view_idx]["depthmap"]
            pose = views[view_idx]["camera_pose"]
            intrinsics = views[view_idx]["camera_intrinsics"]
            pts3d = views[view_idx]["pts3d"]
            valid_mask = views[view_idx]["valid_mask"]
            if args.viz:
                rr.set_time_seconds("stable_time", num * 0.1)
                if view_idx == 0:
                    base_name = "world/image"
                    pts_name = "world/pointcloud_image"
                else:
                    base_name = "world/paired_image"
                    pts_name = "world/pointcloud_paired_image"
                # Log camera info and loaded data
                height, width = image.shape[0], image.shape[1]
                rr.log(
                    base_name,
                    rr.Transform3D(
                        translation=pose[:3, 3],
                        mat3x3=pose[:3, :3],
                        from_parent=False,
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole",
                    rr.Pinhole(
                        image_from_camera=intrinsics,
                        height=height,
                        width=width,
                        camera_xyz=rr.ViewCoordinates.RDF,
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole/rgb",
                    rr.Image(image),
                )
                rr.log(
                    f"{base_name}/pinhole/depth",
                    rr.DepthImage(depthmap),
                )
                # Log points in 3D
                filtered_pts = pts3d[valid_mask]
                filtered_pts_col = image[valid_mask]
                rr.log(
                    pts_name,
                    rr.Points3D(
                        positions=filtered_pts.reshape(-1, 3),
                        colors=filtered_pts_col.reshape(-1, 3),
                    ),
                )
