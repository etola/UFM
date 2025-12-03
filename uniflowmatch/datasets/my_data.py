#!/usr/bin/env python3
# --------------------------------------------------------
# Dataloader for preprocessed BlendedMVS dataset at https://github.com/YoYo000/BlendedMVS
# Please see:
# preprocess_datasets/preprocess_blendedmvs.py
# anymap/generate_pairs/blendedmvs.py
# Adopted from DUSt3R (Naver Corporation, CC BY-NC-SA 4.0 (non-commercial use only))
# Adopted from AnyMap
# --------------------------------------------------------
import os.path as osp

import numpy as np

from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from uniflowmatch.datasets.base.flow_postprocessing import flow_occlusion_post_processing
from uniflowmatch.datasets.utils.my_data_module import MY_DATA_Module
from uniflowmatch.utils.image import imread_cv2


class MY_DATA(BaseStereoViewDataset):
    """
    MY_DATA dataset containing object-centric and birds-eye-view images.
    """

    def __init__(self, *args, ROOT, pair_metadata_path, split=None, overfit_num_pairs=None, **kwargs):
        self.ROOT = ROOT
        self.pair_metadata_path = pair_metadata_path
        self.my_data_info = MY_DATA_Module(ROOT)
        super().__init__(
            *args,
            **kwargs,
        )
        self._load_data(split, overfit_num_pairs)
        self.is_metric_scale = False

    def _load_data(self, split, overfit_num_pairs):
        pairs = np.load(self.pair_metadata_path)
        # Split Dataset based on DUSt3R's train/val split
        if split is None:
            selection = slice(None)
        elif split in ["train", "overfit"]:
            # select 90% of all scenes
            selection = (pairs["seq_low"] % 10) > 0
        elif split == "val":
            # select 10% of all scenes
            selection = (pairs["seq_low"] % 10) == 0
        else:
            raise ValueError(f"Unknown split {split}, must be None, train, val or overfit")
        # Get valid sequences
        valid_seqs = self.my_data_info.sequences
        # IMPORTANT: Convert to numpy array with uint64 dtype to avoid precision loss in np.isin()
        # Using a Python list causes np.isin to convert to float64, losing precision for large integers
        valid_seqls = np.array([int(seq[8:], 16) for seq in valid_seqs], dtype=np.uint64)

        # Filter out invalid sequences
        filtered_selection = np.isin(pairs["seq_low"], valid_seqls) & selection
        self.pairs = pairs[filtered_selection]

        # If split is overfitting, select a particular number of pairs
        if split == "overfit":
            overfit_indices = np.linspace(0, len(self.pairs) - 1, overfit_num_pairs, dtype=int)
            self.pairs = self.pairs[overfit_indices]

        # List of all scenes
        self.scenes = np.unique(self.pairs["seq_low"])  # low is unique enough

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f"{len(self)} pairs from {len(self.scenes)} scenes"

    def _get_views(self, pair_idx, resolution, rng):
        seqh, seql, img1, img2, score = self.pairs[pair_idx]

        seq = f"{seqh:08x}{seql:016x}"
        seq_path = osp.join(self.ROOT, seq)

        views = []

        for view_index in [img1, img2]:
            impath = f"{view_index:08d}"
            image = imread_cv2(osp.join(seq_path, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(seq_path, impath + ".exr"))
            camera_params = np.load(osp.join(seq_path, impath + ".npz"))

            intrinsics = np.float32(camera_params["intrinsics"])
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = camera_params["R_cam2world"]
            camera_pose[:3, 3] = camera_params["t_cam2world"]

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, impath)
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="MY_DATA",
                    label=osp.relpath(seq_path, self.ROOT),
                    instance=impath,
                    is_widebaseline=True,
                    is_synthetic=False,
                    covisible_rendering_parameters=np.array([0.1, 0.1, 0.005], dtype=np.float32),
                    suitable_for_refinement=True,
                )
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="/uniflowmatch/data/my_data_processed", type=str
    )
    parser.add_argument(
        "-pmp",
        "--pair_metadata_path",
        default="/uniflowmatch/data/data_pairs/my_data_pairs.npy",
        type=str,
    )
    parser.add_argument("--viz", action="store_true")

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

    dataset = MY_DATA(
        split="train",
        ROOT=args.root_dir,
        pair_metadata_path=args.pair_metadata_path,
        resolution=(224, 224),
        aug_crop=False,  # "auto_crop_asis"
        aug_monocular=None,
        transform="imgnorm",
        data_norm_type="dust3r",
        seed=777,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    if args.viz:
        rr.script_setup(args, "MY_DATA_Dataloader")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    for idx, views in enumerate(dataloader):
        if idx > 1000:
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
