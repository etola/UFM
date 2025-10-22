import json
import os
from functools import lru_cache
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from uniflowmatch.datasets.base.base_optical_flow_dataset import BaseOpticalFlowDataset


class HD1K(BaseOpticalFlowDataset):
    def __init__(self, ROOT: str, split: str, *args, **kwargs):
        self.root = ROOT
        assert split == "train"
        super().__init__(split=split, *args, **kwargs)

        # scan for all npz files, since each npz file contains one pair of flow supervision
        self.npz_files = sorted(glob(os.path.join(self.root, "*.npz")))

    def __len__(self):
        return len(self.npz_files)

    def _get_raw_views(self, idx, rng: int):
        npz_filename = self.npz_files[idx]
        data = np.load(npz_filename)

        # read image file
        img0 = data["img0"]
        img1 = data["img1"]

        # read the npz file
        flow = data["flow"]
        fov_mask = data["valid"]

        flow[:, ~fov_mask] = 4000  # set invalid flow to a large value

        views = []

        torch_fov_mask = torch.from_numpy(fov_mask)
        view0 = {
            "img": torch.from_numpy(img0),
            "flow": torch.from_numpy(flow),
            "fov_mask": torch_fov_mask,
            "non_occluded_mask": torch_fov_mask.clone(),
            "occlusion_supervision_mask": torch.zeros_like(torch_fov_mask),
            "dataset": "hd1k",
            "label": f"hd1k_{idx}",
            "instance": f"{idx}_0",
            "is_widebaseline": False,
            "is_synthetic": False,
            "suitable_for_refinement": False,
        }

        view1 = {
            "img": torch.from_numpy(img1),
            "flow": 4000 * torch.ones_like(view0["flow"]), # no backward flow supervision
            "fov_mask": torch.zeros_like(torch_fov_mask),
            "non_occluded_mask": torch.zeros_like(torch_fov_mask),
            "occlusion_supervision_mask": torch.zeros_like(torch_fov_mask),
            "dataset": "hd1k",
            "label": f"hd1k_{idx}",
            "instance": f"{idx}_1",
            "is_widebaseline": False,
            "is_synthetic": False,
            "suitable_for_refinement": False,
        }

        views = [view0, view1]

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/uniflowmatch/data/hd1k_processed", type=str)

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

    dataset = HD1K(
        ROOT=args.root_dir,
        split="train",
        resolution=(224, 224),
        # mask_bg=False,
        aug_crop=True,
        aug_monocular=None,
        transform="colorjitter",
        data_norm_type="dust3r",
    )

    data = dataset[0]

    if args.profile:
        import cProfile

        def profile_dataset(num_samples=100):
            sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

            for i, idx in enumerate(sampled_indices):
                dataset[idx]
                print(f"Processed {i+1}/{num_samples} samples", end="\r")

        # save the profile results
        cProfile.run(
            "profile_dataset()",
            filename="/uniflowmatch/outputs/datasets_profile/spring_profile_results",
        )

    if args.viz:
        rr.script_setup(args, f"HD1K_Dataloader")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True)

    for idx, views in enumerate(dataloader):
        if idx > 100:
            break

        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        for view_idx in [0, 1]:
            image = rgb(views[view_idx]["img"][0], norm_type=views[view_idx]["data_norm_type"][0])
            # depthmap = views[view_idx]["depthmap"][0]
            # pose = views[view_idx]["camera_pose"][0]
            # intrinsics = views[view_idx]["camera_intrinsics"][0]
            # pts3d = views[view_idx]["pts3d"][0]
            # valid_mask = views[view_idx]["valid_mask"][0]

            # additional flow data
            flow = views[view_idx]["flow"][0]
            non_occluded_mask = views[view_idx]["non_occluded_mask"][0].numpy()

            # depth_validity = views[view_idx]["depth_validity"][0].float()
            # other_view_depth_validity = views[view_idx]["other_view_depth_validity"][0].float()

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
                # rr.log(
                #     base_name,
                #     rr.Transform3D(
                #         translation=pose[:3, 3],
                #         mat3x3=pose[:3, :3],
                #         from_parent=False,
                #     ),
                # )
                # rr.log(
                #     f"{base_name}/pinhole",
                #     rr.Pinhole(
                #         image_from_camera=intrinsics,
                #         height=height,
                #         width=width,
                #         camera_xyz=rr.ViewCoordinates.RDF,
                #     ),
                # )
                rr.log(
                    f"{base_name}/pinhole/rgb",
                    rr.Image(image),
                )
                # rr.log(
                #     f"{base_name}/pinhole/depth",
                #     rr.DepthImage(depthmap),
                # )
                # Log points in 3D
                # filtered_pts = pts3d[valid_mask]
                # filtered_pts_col = image[valid_mask]
                # filtered_occlusion = non_occluded_mask[valid_mask]

                # rr.log(
                #     pts_name,
                #     rr.Points3D(
                #         positions=filtered_pts.reshape(-1, 3),
                #         colors=filtered_pts_col.reshape(-1, 3),
                #     ),
                # )

                # # log occlusion as coloring of points
                # occluded_point_coloring = filtered_pts_col.reshape(-1, 3)
                # occluded_point_coloring[~filtered_occlusion.reshape(-1)] = np.array([[1, 0, 0]])
                # rr.log(
                #     pts_name + "_occ",
                #     rr.Points3D(
                #         positions=filtered_pts.reshape(-1, 3),
                #         colors=occluded_point_coloring,
                #     ),
                # )

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

                # rr.log(
                #     f"{base_name}/pinhole/depth_validity",
                #     rr.Image(depth_validity),
                # )

                # rr.log(
                #     f"{base_name}/pinhole/other_view_depth_validity",
                #     rr.Image(other_view_depth_validity),
                # )

                rr.log(
                    f"{base_name}/pinhole/fov_mask",
                    rr.Image(fov_mask),
                )

                rr.log(
                    f"{base_name}/pinhole/occlusion_supervision_mask",
                    rr.Image(occlusion_supervision_mask),
                )
