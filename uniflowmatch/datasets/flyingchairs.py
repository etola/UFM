import os
from functools import lru_cache

import numpy as np
import torch
from torchvision.datasets import FlyingChairs as TorchFlyingChairs

from uniflowmatch.datasets.base.base_optical_flow_dataset import BaseOpticalFlowDataset


class FlyingChairs(BaseOpticalFlowDataset):
    def __init__(self, ROOT: str, split: str, *args, **kwargs):
        self.root = ROOT
        super().__init__(split=split, *args, **kwargs)

        self.chairs_dataset = TorchFlyingChairs(root=os.path.dirname(ROOT), split=split)

    def __len__(self):
        return len(self.chairs_dataset)

    def _get_raw_views(self, idx, rng: int):
        img0, img1, flow = self.chairs_dataset[idx]

        img0 = torch.from_numpy(np.asarray(img0).copy())  # from PIL to numpy
        img1 = torch.from_numpy(np.asarray(img1).copy())  # from PIL to numpy
        flow = torch.from_numpy(flow)

        magnitude_valid = (torch.abs(flow[0]) < 1000) & (torch.abs(flow[1]) < 1000)
        torch_fov_mask = self.compute_fov_mask(img0, img1, magnitude_valid, flow)

        view0 = {
            "img": img0,
            "flow": flow,
            "fov_mask": torch_fov_mask,
            "non_occluded_mask": torch_fov_mask.clone(),
            "occlusion_supervision_mask": torch.zeros_like(
                torch_fov_mask
            ),  # no occlusion supervision, this is fov mask
            "dataset": "flyingchairs",
            "label": f"flyingchairs_{idx}",
            "instance": f"{idx}_0",
            "is_widebaseline": False,
            "is_synthetic": True,
            "suitable_for_refinement": False,
        }

        view1 = {
            "img": img1,
            "flow": torch.zeros_like(view0["flow"]),
            "fov_mask": torch.zeros_like(torch_fov_mask),
            "non_occluded_mask": torch.zeros_like(torch_fov_mask),
            "occlusion_supervision_mask": torch.zeros_like(torch_fov_mask),
            "dataset": "flyingchairs",
            "label": f"flyingchairs_{idx}",
            "instance": f"{idx}_1",
            "is_widebaseline": False,
            "is_synthetic": True,
            "suitable_for_refinement": False,
        }

        views = [view0, view1]

        return views

    def compute_fov_mask(self, base_image, other_image, valid, flow):
        assert base_image.shape == other_image.shape
        assert len(base_image.shape) == 3

        fov_mask = valid.clone()
        H, W, C = base_image.shape
        flow_base_coords = self.get_meshgrid(H, W, base_image.device)

        flow_target = flow_base_coords + flow
        fov_mask[(flow_target[0] < 0) | (flow_target[1] < 0) | (flow_target[0] >= W) | (flow_target[1] >= H)] = False

        return fov_mask

    @lru_cache(maxsize=8)
    def get_meshgrid(self, H, W, device):
        flow_base_coords = torch.stack(
            torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"), dim=-1
        ).float()

        flow_base_coords = torch.stack((flow_base_coords[..., -1], flow_base_coords[..., 0]), dim=-1)

        return flow_base_coords.permute(2, 0, 1)


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/uniflowmatch/data/FlyingChairs", type=str)

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

    dataset = FlyingChairs(
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

    if args.viz:
        rr.script_setup(args, f"FlyingChairs_Dataloader")
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
