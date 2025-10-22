import itertools
import os
import os.path as osp
import sys

import cv2
import numpy as np

from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from uniflowmatch.datasets.base.flow_postprocessing import flow_occlusion_post_processing
from uniflowmatch.utils.image import imread_cv2


class HyperSim(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        super().__init__(*args, **kwargs)

        self.split = split

        self.loaded_data = self._load_data()

    def _load_data(self):
        self.all_scenes = sorted([f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))])
        subscenes = []
        for scene in self.all_scenes:
            # not empty
            subscenes.extend(
                [
                    osp.join(scene, f)
                    for f in os.listdir(osp.join(self.ROOT, scene))
                    if os.path.isdir(osp.join(self.ROOT, scene, f))
                    and len(os.listdir(osp.join(self.ROOT, scene, f))) > 0
                ]
            )

        subscene_id = [int(x.split("/")[0].split("_")[1]) for x in subscenes]

        subscenes_filtered = []
        if self.split == "train":
            for subscene, subscene_id_ in zip(subscenes, subscene_id):
                if subscene_id_ < 50:
                    subscenes_filtered.append(subscene)
        elif self.split == "val":
            for subscene, subscene_id_ in zip(subscenes, subscene_id):
                if subscene_id_ >= 50:
                    subscenes_filtered.append(subscene)
        else:
            raise ValueError(f"Unknown split {self.split}")

        offset = 0
        scenes = []
        sceneids = []
        images = []
        start_img_ids = []
        scene_img_list = []
        j = 0
        for scene_idx, scene in enumerate(subscenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_paths = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])
            assert len(rgb_paths) > 0, f"{scene_dir} is empty."
            num_imgs = len(rgb_paths)
            cut_off = self.num_views

            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        self.start_img_ids = start_img_ids

    def __len__(self):
        return len(self.start_img_ids) // 10

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng):

        idx = idx // 10

        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]

        pos, ordered_video = self.get_seq_from_start_id(
            self.num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            block_shuffle=16,
        )

        image_idxs = np.array(all_image_ids)[pos]
        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            rgb_path = self.images[view_idx]
            depth_path = rgb_path.replace("rgb.png", "depth.npy")
            cam_path = rgb_path.replace("rgb.png", "cam.npz")

            rgb_image = imread_cv2(osp.join(scene_dir, rgb_path), cv2.IMREAD_COLOR)
            depthmap = np.load(osp.join(scene_dir, depth_path)).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(osp.join(scene_dir, cam_path))
            intrinsics = cam_file["intrinsics"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="hypersim",
                    label=self.scenes[scene_id] + "_" + rgb_path,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_widebaseline=True,
                    is_synthetic=True,
                    covisible_rendering_parameters=np.array([0.1, 0.1, 0.005], dtype=np.float32),
                    suitable_for_refinement=True,
                )
            )

        assert len(views) == self.num_views
        return views

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [rng.choice(range(min_interval, max_interval + 1)) for _ in range(num_views - 1)]

            # if video or collection
            if rng.random() < video_prob:
                # if fixed interval or random
                if rng.random() < fix_interval_prob:
                    # regular interval
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = pos + rng.choice(pos_candidates, num_views - len(pos), replace=False).tolist()

            pos = sorted(pos) if is_video else self.blockwise_shuffle(pos, rng, block_shuffle)
        else:
            # assert self.allow_repeat
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = self.blockwise_shuffle(pos, rng, block_shuffle) if not is_video else pos
                num_full_repeat = num_views // uniq_num
                pos = pos * num_full_repeat + pos[: num_views - len(pos) * num_full_repeat]
            elif revisit_random < 0.9:  # random
                pos = rng.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="/match_anything/data/hypersim_processed", type=str
    )
    parser.add_argument(
        "-pmp",
        "--pair_metadata_path",
        default="/uniflowmatch/data/dust3r_data_pairs/blendedmvs_pairs.npy",
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

    dataset = HyperSim(
        split="train",
        ROOT=args.root_dir,
        # resolution=(512, 384),
        resolution=(224, 224),
        # resolution=(512, 384),
        # resolution=(560, 560),
        aug_crop=False,  # "auto_crop_asis"
        aug_monocular=None,
        transform="imgnorm",
        data_norm_type="dust3r",
        seed=777,
    )

    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    if args.profile:
        import cProfile

        def profile_dataset(num_samples=1000):
            sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

            for i, idx in enumerate(sampled_indices):
                dataset[idx]
                print(f"Processed {i+1}/{num_samples} samples", end="\r")

        # save the profile results
        cProfile.run(
            "profile_dataset()",
            filename="/uniflowmatch/outputs/datasets_profile/blendedmvs_profile_results",
        )

    if args.viz:
        rr.script_setup(args, f"HyperSim_Dataloader")
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
