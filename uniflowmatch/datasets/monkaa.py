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
from uniflowmatch.datasets.utils.distributed_h5 import DistributedH5Reader


class Monkaa(BaseOpticalFlowDataset):
    def __init__(
        self, ROOT: str, split: str, aug_left_right=False, aug_clean_final=False, dstype="clean", *args, **kwargs
    ):
        self.root = ROOT
        self.dstype = dstype

        super().__init__(split=split, *args, **kwargs)

        self.aug_left_right = aug_left_right  # whether to randomly select from reading left/right camera images
        self.aug_clean_final = aug_clean_final  # whether to randomly select from reading clean/finalpass images

        self.dstype = dstype  # default image type to read
        self.other_dstype = "final" if self.dstype == "clean" else "clean"

        self.individual_quantity_folder = os.path.join(self.root, "individual_quantities")
        self.flow_folder = os.path.join(self.root, "flow")

        individual_quantity_index_path = os.path.join(self.individual_quantity_folder, "index.json")
        flow_index_path = os.path.join(self.flow_folder, "index.json")

        self.individual_quantity_index = pd.read_json(individual_quantity_index_path, orient="index")

        self.individual_quantity_index["h5_reader_index"] = self.individual_quantity_index.index
        individual_quantity_index_search = self.individual_quantity_index.set_index(["seq", "idx"])

        self.flow_index = pd.read_json(flow_index_path, orient="index")

        # precompute the h5 reader index for each pair stored in the flow index
        self.image_pairs = []
        for idx, row in self.flow_index.iterrows():
            img0_idx = individual_quantity_index_search.loc[(row["seq"], row["pair0_id"])]["h5_reader_index"]
            img1_idx = individual_quantity_index_search.loc[(row["seq"], row["pair1_id"])]["h5_reader_index"]
            self.image_pairs.append((int(img0_idx), int(img1_idx)))

        # initialize the flow & individual quantity h5 readers
        self.flow_reader = DistributedH5Reader(self.flow_folder)
        self.individual_quantity_reader = DistributedH5Reader(self.individual_quantity_folder)

    def __len__(self):
        return len(self.image_pairs)

    def _get_raw_views(self, idx, rng: int):
        img1_idx, img2_idx = self.image_pairs[idx]

        direction = "left" if (not self.aug_left_right or (rng.random() < 0.5)) else "right"

        img1_dstype_sel = self.dstype if (not self.aug_clean_final or (rng.random() < 0.5)) else self.other_dstype
        img2_dstype_sel = self.dstype if (not self.aug_clean_final or (rng.random() < 0.5)) else self.other_dstype

        views = []
        for img_idx, view_name, img_dstype in zip(
            [img1_idx, img2_idx], ["view0", "view1"], [img1_dstype_sel, img2_dstype_sel]
        ):
            img = self.individual_quantity_reader.read(
                img_idx,
                [f"frame_{img1_dstype_sel}pass_{direction}"],
            )[
                f"frame_{img1_dstype_sel}pass_{direction}"
            ][..., :3]

            # read paired flow quantities
            flow_data = self.flow_reader.read(
                idx,
                [
                    f"{direction}_{view_name}_flow",
                    f"{direction}_{view_name}_fov_mask",
                    f"{direction}_{view_name}_non_occluded_mask",
                    f"{direction}_{view_name}_occlusion_supervision_mask",
                ],
            )

            metadata = self.individual_quantity_index.loc[img_idx]

            views.append(
                {
                    "img": img,
                    "flow": flow_data[f"{direction}_{view_name}_flow"].permute(2, 0, 1),
                    "fov_mask": flow_data[f"{direction}_{view_name}_fov_mask"],
                    "non_occluded_mask": flow_data[f"{direction}_{view_name}_non_occluded_mask"],
                    "occlusion_supervision_mask": flow_data[f"{direction}_{view_name}_occlusion_supervision_mask"],
                    "dataset": "monkaa",
                    "label": f"{metadata['seq']}_{direction}_{img_dstype}",
                    "instance": f"{idx}_{view_name}",
                    "is_widebaseline": False,
                    "is_synthetic": True,
                    "suitable_for_refinement": False,
                }
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/uniflowmatch/data/monkaa_processed", type=str)

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

    dataset = Monkaa(
        ROOT=args.root_dir,
        split="train",
        dstype="clean",
        resolution=(224, 224),
        # mask_bg=False,
        aug_crop=False,
        aug_monocular=None,
        # monkaa specific options
        aug_left_right=True,
        aug_clean_final=True,
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
            filename="/uniflowmatch/outputs/datasets_profile/monkaa_profile_results",
        )

    if args.viz:
        rr.script_setup(args, f"Monkaa_Dataloader")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=12, shuffle=False)

    for idx, views in enumerate(dataloader):
        if idx > 200:
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
