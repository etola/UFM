# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed StaticThings3D
# dataset at https://github.com/lmb-freiburg/robustmvd/
# See datasets_preprocess/preprocess_staticthings3d.py
# --------------------------------------------------------
import os.path as osp

import numpy as np

from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from uniflowmatch.datasets.base.flow_postprocessing import flow_occlusion_post_processing
from uniflowmatch.utils.image import imread_cv2


class StaticThings3D(BaseStereoViewDataset):
    """Dataset of indoor scenes, 5 images each time"""

    def __init__(self, ROOT, *args, split, mask_bg=False, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        assert mask_bg in (True, False, "rand")
        self.mask_bg = mask_bg

        # loading all pairs
        assert self.split is None
        self.pairs = np.load(osp.join(ROOT, "staticthings_pairs.npy"))

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f"{len(self)} pairs"

    def _get_views(self, pair_idx, resolution, rng):
        scene, seq, cam1, im1, cam2, im2 = self.pairs[pair_idx]
        seq_path = osp.join("TRAIN", scene.decode("ascii"), f"{seq:04d}")

        views = []

        mask_bg = (self.mask_bg == True) or (self.mask_bg == "rand" and rng.choice(2))

        CAM = {b"l": "left", b"r": "right"}
        for cam, idx in [(CAM[cam1], im1), (CAM[cam2], im2)]:
            num = f"{idx:04n}"
            img = num + "_clean.jpg" if rng.choice(2) else num + "_final.jpg"
            image = imread_cv2(osp.join(self.ROOT, seq_path, cam, img))
            depthmap = imread_cv2(osp.join(self.ROOT, seq_path, cam, num + ".exr"))
            camera_params = np.load(osp.join(self.ROOT, seq_path, cam, num + ".npz"))

            intrinsics = camera_params["intrinsics"]
            camera_pose = camera_params["cam2world"]

            if mask_bg:
                depthmap[depthmap > 200] = 0

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, cam, img)
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="StaticThings3D",
                    label=seq_path,
                    instance=cam + "_" + img,
                    is_widebaseline=True,
                    is_synthetic=True,
                    covisible_rendering_parameters=np.array([0.1, 0.1, 0.005], dtype=np.float32),
                    suitable_for_refinement=False,
                )
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="/uniflowmatch/data/staticthings3d_processed", type=str
    )
    parser.add_argument(
        "-pmp",
        "--pair_metadata_path",
        default="/uniflowmatch/data/dust3r_data_pairs/staticthings_pairs.npy",
        type=str,
    )
    parser.add_argument("--viz", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Profile reading from the dataset with cProfile to find bottlenecks"
    )

    return parser


if __name__ == "__main__":
    import rerun as rr
    import torch.multiprocessing as mp
    from torch.utils.data import DataLoader

    from uniflowmatch.datasets.base.base_stereo_view_dataset import view_name
    from uniflowmatch.utils.image import rgb
    from uniflowmatch.utils.viz import script_add_rerun_args, warp_image_with_flow

    mp.set_start_method("spawn", force=True)

    parser = get_parser()
    script_add_rerun_args(parser)  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    dataset = StaticThings3D(
        split="train",
        ROOT=args.root_dir,
        # resolution=(224, 224),
        resolution=(560, 560),
        mask_bg=False,
        aug_crop=False,
        aug_monocular=None,
        transform="colorjitter",
        data_norm_type="dust3r",
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    if args.profile:
        import cProfile

        def profile_dataset(num_samples=1000):
            sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

            for i, idx in enumerate(sampled_indices):
                dataset[idx]
                print(f"Processed {i+1}/{num_samples} samples", end="\r")

        # save the profile results
        cProfile.run("profile_dataset()", filename="/uniflowmatch/data/blendedmvs_profile_results")

    if args.viz:
        rr.script_setup(args, f"StaticThings_Dataloader")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    for idx, views in enumerate(dataloader):
        if idx > 10:
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
