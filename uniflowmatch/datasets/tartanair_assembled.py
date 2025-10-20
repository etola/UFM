#!/usr/bin/env python3
# --------------------------------------------------------
# Dataloader for Assembled TartanAir dataset stored in distributed h5 format
# --------------------------------------------------------
import os.path as osp

import numpy as np

from uniflowmatch.datasets.base.base_optical_flow_dataset import BaseOpticalFlowDataset
from uniflowmatch.datasets.base.flow_postprocessing import (
    apply_flow_postprocessing_and_merge_batch,
    collate_fn_with_delayed_flow_postprocessing,
)
from uniflowmatch.datasets.utils.distributed_h5 import DistributedH5Reader


class TartanairAssembled(BaseOpticalFlowDataset):  # BaseOpticalFlowDataset
    """
    BlendeMVS dataset containing object-centric and birds-eye-view images.
    """

    def __init__(
        self,
        *args,
        ROOT: str,
        split: str = None,
        overfit_num_pairs: int = None,
        **kwargs,
    ):
        self.ROOT = ROOT

        super().__init__(
            *args,
            **kwargs,
        )

        self.split = split
        self._load_data(split, overfit_num_pairs)  # load index map of selected split
        self.is_metric_scale = False

    def _load_data(self, split, overfit_num_pairs):
        if split is None:
            raise ValueError(f"Unknown split {split}, must be train, val or overfit")
        elif split in ["train", "overfit", "val"]:
            self.h5_reader = DistributedH5Reader(
                osp.join(self.ROOT, "train" if split in ["train", "overfit"] else "val")
            )

            if split == "overfit":
                raise NotImplementedError("Overfitting is not supported for TartanAir Assembled dataset")
        else:
            self.h5_reader = DistributedH5Reader(osp.join(self.ROOT, split))

    def __len__(self):
        return len(self.h5_reader)

    def get_stats(self):
        return f"{len(self)} pairs from pregenerated TartanAir Pairs"

    def _get_raw_views(self, idx, rng: int):
        data = self.h5_reader.read(
            idx,
            [
                "img0",
                "img1",
                "flow_fwd",
                "flow_bwd",
                "non_occluded_fwd",
                "non_occluded_bwd",
                "occlusion_supervision_mask_fwd",
                "occlusion_supervision_mask_bwd",
                "fov_mask_fwd",
                "fov_mask_bwd",
            ],
        )

        imgs = [data["img0"], data["img1"]]
        flows = [data["flow_fwd"], data["flow_bwd"]]
        non_occluded_masks = [data["non_occluded_fwd"], data["non_occluded_bwd"]]
        occlusion_supervision_masks = [data["occlusion_supervision_mask_fwd"], data["occlusion_supervision_mask_bwd"]]
        fov_masks = [data["fov_mask_fwd"], data["fov_mask_bwd"]]

        views = []
        for view_id, (img, flow, non_occluded_mask, occlusion_supervision_mask, fov_mask) in enumerate(
            zip(imgs, flows, non_occluded_masks, occlusion_supervision_masks, fov_masks)
        ):
            views.append(
                {
                    "img": img,
                    "flow": flow.float(),
                    "non_occluded_mask": non_occluded_mask,
                    "occlusion_supervision_mask": occlusion_supervision_mask,
                    "fov_mask": fov_mask,
                    "dataset": "tartanair_assembled",
                    "label": f"{idx}",
                    "instance": f"{idx}_{view_id}",
                    "is_widebaseline": True,
                    "is_synthetic": True,
                    "suitable_for_refinement": False,
                }
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd",
        "--root_dir",
        default="/jet/home/yzhang25/match_anything/data/TartanAir/assembled/tartanair_640_mega_training_0203_good",
        type=str,
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

    dataset = TartanairAssembled(
        split="train",
        ROOT=args.root_dir,
        # resolution=(512, 384),
        # resolution=(224, 224),
        resolution=(224, 224),
        aug_crop=True,
        aug_monocular=None,
        transform="imgnorm",
        data_norm_type="dust3r",
        seed=777,
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_with_delayed_flow_postprocessing
    )

    if args.profile:
        import cProfile

        def profile_dataset(num_samples=1000):
            sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

            for i, idx in enumerate(sampled_indices):
                dataset[idx]
                print(f"Processed {i+1}/{num_samples} samples", end="\r")

        # save the profile results
        cProfile.run(
            "profile_dataset()", filename="/home/inf/uniflowmatch/outputs/datasets_profile/blendedmvs_profile_results"
        )

    if args.viz:
        rr.script_setup(args, f"TartanAir_assermbled_Dataloader")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    for idx, views in enumerate(dataloader):
        if idx > 1000:
            break

        views = apply_flow_postprocessing_and_merge_batch(
            views,
            depth_error_threshold=0.1,
            depth_error_temperature=0.1,
            relative_depth_error_threshold=0.005,
            opt_iters=0,
        )

        assert len(views) == 2
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


## 795, 60,
