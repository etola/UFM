import json
import os
from functools import lru_cache
from glob import glob
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tvf
from tqdm import tqdm

from uniflowmatch.datasets.base.base_optical_flow_dataset import BaseOpticalFlowDataset
from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from uniflowmatch.datasets.base.flow_postprocessing import (
    apply_flow_postprocessing_and_merge_batch,
    collate_fn_with_delayed_flow_postprocessing,
)
from uniflowmatch.datasets.utils.distributed_h5 import DistributedH5Reader
from uniflowmatch.datasets.utils.kubric_fileio import read_png, read_tiff

# Worker: 1. read rgb, depth, segmentation, object_index, transforms, intrinsics, extrinsics
#         2. collor jitter(must be done on CPU) to use torch
#         3. compute flow, occlusion, cropping(augmentation),
# batched GPU: compute flow, occlusion,


class Kubric4D(BaseStereoViewDataset):
    # in kubric, flow, occlusion all need to be computed on the GPU.

    def __init__(
        self,
        ROOT: str,
        processed_root,
        split: str,
        pairs_npz_path: str,
        alternative_root: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.root = ROOT
        self.alternative_root = alternative_root

        super().__init__(split=split, *args, **kwargs)

        # seq_num, vp0, vp1, frame0, frame1
        self.pairs = np.load(pairs_npz_path)["train_pairs" if split == "train" else "test_pairs"]

        # cache metadata
        self.metadata = {}
        for seq_id in tqdm(np.unique(self.pairs[:, 0])):
            data = np.load(os.path.join(processed_root, "metadata", f"scn{seq_id:05d}.npz"))

            intrinsics = data["intrinsics"]
            intrinsics[0, 2] -= 0.5
            intrinsics[1, 2] -= 0.5

            self.metadata[seq_id] = {
                "intrinsics": data["intrinsics"],
                "extrinsics": data["extrinsics"],
                "positions": data["positions"],
                "orientations": data["orientations"],
                "instances_position_scene": data["instances_position_scene"],
                "instances_orientation_scene": data["instances_orientation_scene"],
            }

        self.color_jitter = tvf.ColorJitter(0.5, 0.5, 0.5, 0.1)

    def __len__(self):
        return len(self.pairs)

    def _get_item(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)

        ### kubric sepecific: read information, and prepare the view dict

        seq_num, vp0, vp1, frame0, frame1 = self.pairs[idx]

        # print(f"seq_num: {seq_num}, vp0: {vp0}, vp1: {vp1}, frame0: {frame0}, frame1: {frame1}")

        seq_folder = os.path.join(self.root, f"scn{seq_num:05d}")
        if self.alternative_root is not None and (not os.path.exists(seq_folder)):
            seq_folder = os.path.join(self.alternative_root, f"scn{seq_num:05d}")

        data0 = self.read_information(seq_folder, vp0, frame0)
        data1 = self.read_information(seq_folder, vp1, frame1)

        view0 = {
            "img": torch.from_numpy(data0["rgb"]),
            "norm_depthmap": torch.from_numpy(data0["depth"]),
            "segmentation": torch.from_numpy(data0["segmentation"]),
            "is_widebaseline": True,
            "is_synthetic": True,
            "suitable_for_refinement": False,
        }

        view1 = {
            "img": torch.from_numpy(data1["rgb"]),
            "norm_depthmap": torch.from_numpy(data1["depth"]),
            "segmentation": torch.from_numpy(data1["segmentation"]),
            "is_widebaseline": True,
            "is_synthetic": True,
            "suitable_for_refinement": False,
        }

        # special coordinate transform for kubric on camera coordinate definition
        views = (view0, view1)
        for view, data, vp in zip(views, [data0, data1], [vp0, vp1]):
            R = np.diag([1, -1, -1])  # transform both intrinsics and extrinsics to make K positive

            H, W = view["img"].shape[:2]

            K = self.metadata[seq_num]["intrinsics"][vp].copy() * (np.array([W, H, 1])[:, None])
            K = torch.from_numpy(K @ R)

            T = self.metadata[seq_num]["extrinsics"][vp].copy()
            T[:3, :3] = T[:3, :3] @ R

            view["intrinsics"] = K
            view["extrinsics"] = torch.from_numpy(T)

        assert len(views) == self.num_views

        for view, data, vp_idx, frame_idx in zip(views, [data0, data1], [vp0, vp1], [frame0, frame1]):
            obj_pos = self.metadata[seq_num]["instances_position_scene"][vp_idx, :, frame_idx]
            obj_ori = self.metadata[seq_num]["instances_orientation_scene"][vp_idx, :, frame_idx]  # wxyz

            # prepend identity position & quaternion for background(seg id 0)
            obj_pos = np.concatenate([np.zeros((1, 3)), obj_pos], axis=0)
            obj_ori = np.concatenate([np.array([[1, 0, 0, 0]]), obj_ori], axis=0)

            # pad to 100 objects
            obj_pos_padded = np.zeros((100, 3))
            obj_ori_padded = np.zeros((100, 4))
            obj_ori_padded[:, 0] = 1.0

            obj_pos_padded[: obj_pos.shape[0]] = obj_pos
            obj_ori_padded[: obj_ori.shape[0]] = obj_ori

            # object_position_img = obj_pos[view['segmentation']]
            # object_orientation_img = obj_ori[view['segmentation']]

            view["object_position"] = torch.from_numpy(obj_pos_padded)
            view["object_orientation"] = torch.from_numpy(obj_ori_padded)  # wxyz

            view["dataset"] = "kubric4d"
            view["label"] = f"scn{seq_num:05d}"
            view["instance"] = f"{seq_num:05d}_{vp_idx:02d}_{frame_idx:05d}"
            view["data_norm_type"] = self.data_norm_type

            # additional augmentation config to be performed on the GPU
            view["resolution"] = resolution
            view["aug_crop"] = self.aug_crop

        # normalize image to [0, 1]
        view0["img"] = view0["img"].permute(2, 0, 1) / 255.0
        view1["img"] = view1["img"].permute(2, 0, 1) / 255.0

        # apply color jitter on CPU if required
        if self.transform == "colorjitter":
            view0["img"] = self.color_jitter(view0["img"])
            view1["img"] = self.color_jitter(view1["img"])

        for view in views:
            view["data_partition"] = self.get_data_partition(idx)

        return (view0, view1)

    def read_information(self, root: str, viewpoint: int, frame_id: int):
        frames_folder = os.path.join(root, f"frames_p0_v{viewpoint:d}")

        # data_ranges = os.path.join(frames_folder, f"data_ranges.json")
        # with open(data_ranges) as f:
        #     data_ranges = json.load(f)

        # frame level quantity
        depth_file = os.path.join(frames_folder, f"depth_{frame_id:05d}.tiff")
        # forward_flow_file = os.path.join(frames_folder, f"forward_flow_{frame_id:05d}.png")
        # normal_file = os.path.join(frames_folder, f"normal_{frame_id:05d}.png")
        # object_coordinates_file = os.path.join(frames_folder, f"object_coordinates_{frame_id:05d}.png")
        rgba_file = os.path.join(frames_folder, f"rgba_{frame_id:05d}.png")
        segmentation_file = os.path.join(frames_folder, f"segmentation_{frame_id:05d}.png")

        # read the files
        rgb = read_png(rgba_file)[..., :3]
        depth = read_tiff(depth_file)

        # https://github.com/basilevh/gcd/blob/c3c517f2a53e49d2afdff01cd2b86123e40c62c7/data-gen/kubric/kubric/file_io.py#L293
        # data = (data - min_value) * 65535 / (max_value - min_value)
        # flow_fwd = read_png(forward_flow_file)[..., :2]
        # flow_fwd_max = data_ranges["forward_flow"]["max"]
        # flow_fwd_min = data_ranges["forward_flow"]["min"]

        # flow_fwd = (flow_fwd - flow_fwd_min) * 65535 / (flow_fwd_max - flow_fwd_min)

        # https://github.com/basilevh/gcd/blob/c3c517f2a53e49d2afdff01cd2b86123e40c62c7/data-gen/kubric/kubric/renderer/blender_utils.py#L449
        # normal saved as: (exr_layers["normal"].clip(-1.0, 1.0) + 1) * 65535 / 2).astype(np.uint16)
        # normal = read_png(normal_file) / 65535.0 * 2 - 1

        # https://github.com/basilevh/gcd/blob/c3c517f2a53e49d2afdff01cd2b86123e40c62c7/data-gen/kubric/kubric/renderer/blender_utils.py#L455
        # (exr_layers["object_coordinates"].clip(0.0, 1.0) * 65535).astype(np.uint16)
        # object_coordinates = read_png(object_coordinates_file) / 65535.0

        # 0 = bg, others = index of asset in scene
        segmentation = read_png(segmentation_file)

        return {
            "rgb": rgb,
            "depth": depth.squeeze(-1),
            # "flow_fwd": flow_fwd,
            # "normal": normal,
            # "object_coordinates": object_coordinates,
            "segmentation": segmentation.squeeze(-1),
        }


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="/uniflowmatch/data/flyingthings3d_processed", type=str
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

    dataset = Kubric4D(
        ROOT="/uniflowmatch/data/kubric4d/kubric4d_raw",
        processed_root="/uniflowmatch/data/kubric4d/kubric4d_metadata",
        split="train",
        pairs_npz_path="/uniflowmatch/data/kubric4d/kubric4d_metadata/sampled_pairs.npz",
        resolution=(224, 224),
        data_norm_type="dust3r",
        transform="imgnorm",
        aug_crop=True,
    )

    if args.viz:
        rr.script_setup(args, f"Kubric_Dataloader")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn_with_delayed_flow_postprocessing
    )
    for idx, views in enumerate(dataloader):
        views = apply_flow_postprocessing_and_merge_batch(views)

        if idx > 100:
            break

        assert len(views) == 2
        # print(idx, view_name(views[0]), view_name(views[1]))
        print(idx, views[0]["instance"], views[1]["instance"])
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

            # fov_mask = views[view_idx]["fov_mask"][0].float()
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

                # rr.log(
                #     f"{base_name}/pinhole/fov_mask",
                #     rr.Image(fov_mask),
                # )

                rr.log(
                    f"{base_name}/pinhole/occlusion_supervision_mask",
                    rr.Image(occlusion_supervision_mask),
                )
