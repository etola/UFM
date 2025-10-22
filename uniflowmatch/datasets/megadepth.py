#!/usr/bin/env python3
# --------------------------------------------------------
# Dataloader for preprocessed MegaDepth
# dataset at https://www.cs.cornell.edu/projects/megadepth/
# See preprocess_datasets/preprocess_megadepth.py
# Adopted from DUSt3R (Naver Corporation, CC BY-NC-SA 4.0 (non-commercial use only))
# Adopted from AnyMap
# --------------------------------------------------------
import os.path as osp

import numpy as np
from torch.utils.data import DataLoader

from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from uniflowmatch.datasets.base.flow_postprocessing import flow_occlusion_post_processing
from uniflowmatch.utils.image import imread_cv2


class MegaDepth(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, pair_metadata_path, **kwargs):
        self.ROOT = ROOT
        self.pair_metadata_path = pair_metadata_path
        super().__init__(
            *args,
            **kwargs,
        )
        self.loaded_data = self._load_data(self.split, self.pair_metadata_path)
        self.is_metric_scale = False

        if self.split is None:
            pass
        elif self.split == "train":
            self.select_scene(("0015", "0022"), opposite=True)
        elif self.split == "val":
            self.select_scene(("0015", "0022"))
        else:
            raise ValueError(f"bad {self.split=}")

    def _load_data(self, split, pair_metadata_path):
        with np.load(pair_metadata_path, allow_pickle=True) as data:
            self.all_scenes = data["scenes"]
            self.all_images = data["images"]
            self.pairs = data["pairs"]

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f"{len(self)} pairs from {len(self.all_scenes)} scenes"

    def select_scene(self, scene, *instances, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = [s.startswith(scenes) for s in self.all_scenes]
        assert any(scene_id), "no scene found"

        valid = np.in1d(self.pairs["scene_id"], np.nonzero(scene_id)[0])
        if instances:
            image_id = [i.startswith(instances) for i in self.all_images]
            image_id = np.nonzero(image_id)[0]
            assert len(image_id), "no instance found"
            # both together?
            if len(instances) == 2:
                valid &= np.in1d(self.pairs["im1_id"], image_id) & np.in1d(self.pairs["im2_id"], image_id)
            else:
                valid &= np.in1d(self.pairs["im1_id"], image_id) | np.in1d(self.pairs["im2_id"], image_id)

        if opposite:
            valid = ~valid
        assert valid.any()
        self.pairs = self.pairs[valid]

    def _get_views(self, pair_idx, resolution, rng):
        scene_id, im1_id, im2_id, score = self.pairs[pair_idx]

        scene, subscene = self.all_scenes[scene_id].split()
        seq_path = osp.join(self.ROOT, scene, subscene)

        views = []

        for im_id in [im1_id, im2_id]:
            img = self.all_images[im_id]
            try:
                image = imread_cv2(osp.join(seq_path, img + ".jpg"))
                depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
                camera_params = np.load(osp.join(seq_path, img + ".npz"))
            except Exception as e:
                raise OSError(f"cannot load {img}, got exception {e}")

            intrinsics = np.float32(camera_params["intrinsics"])
            camera_pose = np.float32(camera_params["cam2world"])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img)
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="MegaDepth",
                    label=osp.relpath(seq_path, self.ROOT),
                    instance=img,
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
        "-rd", "--root_dir", default="/uniflowmatch/data/megadepth_processed", type=str
    )
    parser.add_argument(
        "-pmp",
        "--pair_metadata_path",
        default="/uniflowmatch/data/dust3r_data_pairs/megadepth_pairs.npz",
        type=str,
    )
    parser.add_argument("--viz", action="store_true")

    return parser


if __name__ == "__main__":
    import rerun as rr

    from uniflowmatch.datasets.base.base_stereo_view_dataset import view_name
    from uniflowmatch.utils.image import rgb
    from uniflowmatch.utils.viz import script_add_rerun_args, warp_image_with_flow

    parser = get_parser()
    script_add_rerun_args(parser)  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    # dataset = MegaDepth(split='train', ROOT=args.root_dir, pair_metadata_path=args.pair_metadata_path,
    #                     resolution=(512, 336), transform='colorjitter', data_norm_type='dust3r',
    #                     aug_crop='auto', aug_monocular=0.005)
    dataset = MegaDepth(
        split="val",
        ROOT=args.root_dir,
        pair_metadata_path=args.pair_metadata_path,
        resolution=(512, 336),
        seed=777,
        transform="imgnorm",
        data_norm_type="dust3r",
    )

    if args.viz:
        rr.script_setup(args, f"MegaDepth_Dataloader")
        rr.set_time_seconds("stable_time", 0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

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
