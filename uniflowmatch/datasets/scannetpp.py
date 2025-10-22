# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# This file is copied from https://github.com/naver/dust3r/blob/main/dust3r/datasets/scannetpp.py
#
# Dataloader for preprocessed scannet++
# dataset at https://github.com/scannetpp/scannetpp - non-commercial research and educational purposes
# https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf
# See datasets_preprocess/preprocess_scannetpp.py
# --------------------------------------------------------
import os.path as osp

import cv2
import numpy as np

from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from uniflowmatch.datasets.base.flow_postprocessing import flow_occlusion_post_processing
from uniflowmatch.utils.image import imread_cv2


class ScanNetpp(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        with np.load(osp.join(self.ROOT, "all_metadata.npz")) as data:
            self.scenes = data["scenes"]
            self.sceneids = data["sceneids"]
            self.images = data["images"]
            self.intrinsics = data["intrinsics"].astype(np.float32)
            self.trajectories = data["trajectories"].astype(np.float32)
            self.pairs = data["pairs"][:, :2].astype(int)

    def __len__(self):
        train_pairs = int(0.95 * len(self.pairs))
        val_pairs = len(self.pairs) - train_pairs

        return train_pairs if self.split == "train" else val_pairs

    def _get_views(self, idx, resolution, rng):
        assert idx < self.__len__()

        image_idx1, image_idx2 = (
            self.pairs[idx] if self.split == "train" else self.pairs[idx + int(0.95 * len(self.pairs))]
        )

        views = []
        for view_idx in [image_idx1, image_idx2]:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_dir, "images", basename + ".jpg"))
            # Load depthmap
            depthmap = imread_cv2(osp.join(scene_dir, "depth", basename + ".png"), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="ScanNet++",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_widebaseline=True,
                    is_synthetic=False,
                    covisible_rendering_parameters=np.array([0.1, 0.1, 0.005], dtype=np.float32),
                    suitable_for_refinement=False,
                )
            )
        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="/uniflowmatch/data/scannetpp_anymap_processed/train", type=str
    )
    parser.add_argument("--viz", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Profile reading from the dataset with cProfile to find bottlenecks"
    )

    return parser


if __name__ == "__main__":
    import rerun as rr
    from torch.utils.data import DataLoader

    from uniflowmatch.datasets.base.base_stereo_view_dataset import view_name
    from uniflowmatch.utils.image import rgb
    from uniflowmatch.utils.viz import script_add_rerun_args, warp_image_with_flow

    parser = get_parser()
    script_add_rerun_args(parser)  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    dataset = ScanNetpp(
        split="train",
        ROOT=args.root_dir,
        resolution=(512, 384),
        # resolution=(224, 224),
        aug_crop="auto_crop_asis",
        aug_monocular=None,
        transform="colorjitter",
        data_norm_type="dust3r",
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.profile:
        import cProfile

        def profile_dataset(num_samples=100):
            # sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

            # for i, idx in enumerate(sampled_indices):
            #     dataset[idx]
            #     print(f"Processed {i+1}/{num_samples} samples", end="\r")

            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

            for i, views in enumerate(dataloader):
                if i > num_samples:
                    break
                print(f"Processed {i+1}/{num_samples} samples", end="\r")

                flow_occlusion_post_processing(
                    views,
                    depth_error_threshold=0.1,
                    depth_error_temperature=0.1,
                    relative_depth_error_threshold=0.005,
                    opt_iters=2,
                )

        # save the profile results
        cProfile.run(
            "profile_dataset()",
            filename="/uniflowmatch/outputs/datasets_profile/scannet_profile_results",
        )

    if args.viz:
        rr.script_setup(args, f"Scannetpp_Dataloader")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    for idx, views in enumerate(dataloader):
        if idx > 100:
            break

        # batch process for flow supervision
        flow_occlusion_post_processing(
            views,
            depth_error_threshold=0.1,
            depth_error_temperature=0.1,
            relative_depth_error_threshold=0.005,
            opt_iters=0,
        )

        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        for view_idx in [0, 1]:
            image = rgb(views[view_idx]["img"][0], norm_type=views[view_idx]["data_norm_type"][0])
            depthmap = views[view_idx]["depthmap"][0]
            pose = views[view_idx]["camera_pose"][0]
            intrinsics = views[view_idx]["camera_intrinsics"][0]
            pts3d = views[view_idx]["pts3d"][0]
            valid_mask = views[view_idx]["valid_mask"][0]

            # additional flow data
            flow = views[view_idx]["flow"][0]
            non_occluded_mask = views[view_idx]["non_occluded_mask"][0].numpy()

            depth_validity = views[view_idx]["depth_validity"][0].float()
            other_view_depth_validity = views[view_idx]["other_view_depth_validity"][0].float()

            fov_mask = views[view_idx]["fov_mask"][0].float()
            occlusion_supervision_mask = views[view_idx]["occlusion_supervision_mask"][0].float()

            other_image = rgb(views[1 - view_idx]["img"][0], norm_type=views[1 - view_idx]["data_norm_type"][0])

            if args.viz:
                rr.set_time_seconds("stable_time", idx * 0.1)
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
                filtered_occlusion = non_occluded_mask[valid_mask]

                rr.log(
                    pts_name,
                    rr.Points3D(
                        positions=filtered_pts.reshape(-1, 3),
                        colors=filtered_pts_col.reshape(-1, 3),
                    ),
                )

                # log occlusion as coloring of points
                occluded_point_coloring = filtered_pts_col.reshape(-1, 3)
                occluded_point_coloring[~filtered_occlusion.reshape(-1)] = np.array([[1, 0, 0]])
                rr.log(
                    pts_name + "_occ",
                    rr.Points3D(
                        positions=filtered_pts.reshape(-1, 3),
                        colors=occluded_point_coloring,
                    ),
                )

                # log gif between original and warped image
                warped_image = warp_image_with_flow(
                    image, non_occluded_mask[..., None], other_image, flow.permute(1, 2, 0).numpy()
                )

                rr.log(
                    f"{base_name}/pinhole/warped",
                    rr.Image(warped_image),
                )

                rr.log(
                    f"{base_name}/pinhole/covisible",
                    rr.Image(non_occluded_mask.astype(np.float32)),
                )

                rr.log(
                    f"{base_name}/pinhole/depth_validity",
                    rr.Image(depth_validity),
                )

                rr.log(
                    f"{base_name}/pinhole/other_view_depth_validity",
                    rr.Image(other_view_depth_validity),
                )

                rr.log(
                    f"{base_name}/pinhole/fov_mask",
                    rr.Image(fov_mask),
                )

                rr.log(
                    f"{base_name}/pinhole/occlusion_supervision_mask",
                    rr.Image(occlusion_supervision_mask),
                )
