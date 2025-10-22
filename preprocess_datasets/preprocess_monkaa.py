"""
This file is for preprocessing the Monkaa dataset.

In particular, we will aim:
1. compute dynamic covisible mask based on disparity & disparity change maps
2. archive all the small files of the dataset into distributed h5 files

Known issues: covisible mask for some of the chairs have false negatives(covisible pixels are marked as not covisible)
"""

import json
import os
import random
from glob import glob
from typing import Any, List

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate
from tqdm import tqdm

from uniflowmatch.datasets.base.flow_postprocessing import (
    compute_reprojection_error,
    flow_occlusion_post_processing,
    get_meshgrid_torch,
    query_projected_mask,
    z_depthmap_to_norm_depthmap_batched,
)
from uniflowmatch.datasets.utils.distributed_h5 import DistributedH5Reader
from uniflowmatch.datasets.utils.frame_utils import parse_pose_file, read_gen
from uniflowmatch.utils.parallel import IndependentParallelProcessing, ParallelProcessing


class ProcessMonkaaSingleQuantity(IndependentParallelProcessing):
    def __init__(self, root: str, output_path: str, batch_size=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.batch_size = batch_size

        self.output_path = os.path.join(output_path, "individual_quantities")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def get_total_job_args(self) -> List[Any]:
        # return a list of arguments to be passed to the final thread executor

        # scan for all the image folders in the dataset
        image_dirs = sorted(glob(os.path.join(self.root, "frames_cleanpass", "*")))

        job_args = []
        for image_dir in image_dirs:
            images = glob(os.path.join(image_dir, "left", "*.png"))
            image_ids = sorted([int(os.path.basename(image).split(".")[0]) for image in images])

            path_parts = image_dir.split("/")

            seq = path_parts[-1]

            for image_id in image_ids:
                job_args.append({"seq": seq, "idx": image_id})

        job_df = pd.DataFrame(job_args)

        # save the final index file
        index_df = job_df.copy()

        # split the job into batches
        job_batches = []
        filenames = []
        indexes = []
        for i in range(0, len(job_df), self.batch_size):
            small_job_df = job_df.iloc[i : i + self.batch_size]
            job_batches.append(small_job_df)

            filename = f"{small_job_df.index[0]}_{small_job_df.index[-1]}.h5"
            for j in range(len(small_job_df)):
                filenames.append(filename)
                indexes.append(j)

        index_json = {}
        for i, ((filename, index), (_, row)) in enumerate(zip(job_df.iterrows(), index_df.iterrows())):
            index_json[i] = {
                "file_name": filenames[i],
                "file_index": indexes[i],
                "seq": row["seq"],
                "idx": row["idx"],
            }

        with open(os.path.join(self.output_path, "index.json"), "w") as f:
            json.dump(index_json, f)

        return job_batches

    def thread_worker(self, job_arg: pd.DataFrame):
        # this function will access information that is defined from a single image
        # and process them into h5 files.
        h5_filepath = os.path.join(self.output_path, f"{job_arg.index[0]}_{job_arg.index[-1]}.h5")

        if os.path.exists(h5_filepath):
            # check if the file is valid
            try:
                depth_average = []
                with h5py.File(h5_filepath, "r") as h5_file:
                    depth = h5_file["depth_left"]
                    for i in range(len(depth)):
                        depth_average.append(depth[i].mean())

                        if depth[i].mean() == 0:
                            print(f"Corrupted file at {i}")
                            raise Exception("Corrupted file at {i}")

                print(f"Skipping {h5_filepath} as it is valid")
                return  # skip the processing if the file is valid
            except:
                print(f"Corrupted file at {h5_filepath}, recreating")

        with h5py.File(h5_filepath, "w") as h5_file:
            for i, (_, row) in tqdm(enumerate(job_arg.iterrows())):
                data = {}
                for direction in ["left", "right"]:
                    # frames_cleanpass
                    frame_cleanpass = np.array(
                        read_gen(
                            os.path.join(
                                self.root,
                                "frames_cleanpass",
                                row["seq"],
                                direction,
                                f"{row['idx']:04d}.png",
                            )
                        )
                    )[..., :3]

                    frame_finalpass = np.array(
                        read_gen(
                            os.path.join(
                                self.root,
                                "frames_finalpass",
                                row["seq"],
                                direction,
                                f"{row['idx']:04d}.png",
                            )
                        )
                    )[..., :3]

                    # disparity
                    disparity = np.array(
                        read_gen(
                            os.path.join(
                                self.root,
                                "disparity",
                                row["seq"],
                                direction,
                                f"{row['idx']:04d}.pfm",
                            )
                        )
                    )

                    # intrinsics
                    intrinsics = np.array([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]]).astype(
                        np.float32
                    )

                    fx = intrinsics[0, 0]

                    # compute derived data - depth
                    depth = fx * 1.0 / disparity

                    # pose
                    pose_file = os.path.join(self.root, "camera_data", row["seq"], "camera_data.txt")

                    # some pose are missing!!! how can they miss this?
                    try:
                        pose = parse_pose_file(pose_file)
                        pose = pose[row["idx"]][direction].astype(np.float32)
                    except:
                        pose = np.nan * np.ones((4, 4))
                        print(f"Missing pose for {row['seq']} {direction} {row['idx']:04d}")

                    data[direction] = {
                        "frame_cleanpass": frame_cleanpass,
                        "frame_finalpass": frame_finalpass,
                        "intrinsics": intrinsics,
                        "depth": depth,
                        "pose": pose,
                    }

                    final_save_data = {}
                    final_save_data.update({k + f"_{direction}": v for k, v in data[direction].items()})

                    # save the data
                    for key, value in final_save_data.items():
                        if key not in h5_file:
                            shape = (len(job_arg),) + value.shape
                            dtype = value.dtype
                            chunking = (1,) + value.shape
                            h5_file.create_dataset(
                                key,
                                shape=shape,
                                dtype=dtype,
                                chunks=chunking,
                                compression="gzip",
                                compression_opts=9,
                            )
                        h5_file[key][i] = value
                    h5_file.flush()


class ProcessMonkaaFlow(IndependentParallelProcessing):

    def __init__(self, root: str, processed_root: str, batch_size=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.processed_root = processed_root
        self.batch_size = batch_size
        self.minibatch = 4

        self.output_path = os.path.join(processed_root, "flow")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def get_total_job_args(self) -> List[Any]:
        # return a list of arguments to be passed to the final thread executor

        # scan for all the image folders in the dataset
        image_dirs = sorted(glob(os.path.join(self.root, "frames_cleanpass", "*")))

        job_args = []
        for image_dir in image_dirs:
            images = glob(os.path.join(image_dir, "left", "*.png"))
            image_ids = sorted([int(os.path.basename(image).split(".")[0]) for image in images])

            path_parts = image_dir.split("/")

            seq = path_parts[-1]

            for image_id, next_image_id in zip(image_ids[:-1], image_ids[1:]):
                job_args.append({"seq": seq, "pair0_id": image_id, "pair1_id": next_image_id})

        job_df = pd.DataFrame(job_args)
        job_df.to_csv(os.path.join(self.output_path, "job_df.csv"))

        filenames = []
        file_indexes = []

        # split the job into batches
        job_batches = []
        for i in range(0, len(job_df), self.batch_size):
            current_batches = job_df.iloc[i : i + self.batch_size]
            job_batches.append(current_batches)

            lower = current_batches.index[0]
            upper = current_batches.index[-1]

            for j, _ in enumerate(range(lower, upper + 1)):
                filenames.append(f"{lower}_{upper}.h5")
                file_indexes.append(j)

        job_df["file_name"] = filenames
        job_df["file_index"] = file_indexes

        job_df.to_json(os.path.join(self.output_path, "index.json"), orient="index")
        # exit()

        return job_batches

    def on_before_process_processing(self, node_job_args, manager):
        # cache the reader to access our packed data
        self.indiv_quantity_reader = DistributedH5Reader(os.path.join(self.processed_root, "individual_quantities"))

        index_file = os.path.join(self.processed_root, "individual_quantities", "index.json")
        with open(index_file, "r") as f:
            self.index_to_file_map = json.load(f)

        self.index_to_file_map = pd.DataFrame.from_dict(self.index_to_file_map, orient="index")
        self.index_to_file_map["original_index"] = self.index_to_file_map.index
        self.index_to_file_map.set_index(["seq", "idx"], inplace=True)

    def thread_worker(self, job_arg: Any):
        # split the job into minibatches
        minibatches = []
        for i in range(0, len(job_arg), self.minibatch):
            minibatches.append(job_arg.iloc[i : i + self.minibatch])

        h5_filename = f"{job_arg.index[0]}_{job_arg.index[-1]}.h5"
        h5_filepath = os.path.join(self.output_path, h5_filename)

        with h5py.File(h5_filepath, "w") as h5_file:

            for direction in ["left", "right"]:
                minibatch_index = 0

                for minibatch in tqdm(minibatches):
                    all_data = []
                    for _, row in minibatch.iterrows():
                        # we need the following quantities for a view pair:
                        # "depthmap", "camera_pose", camera_intrinsics", "flow", "depth_in_other_view", "depth_validity"

                        view0_individual_idx = int(
                            self.index_to_file_map.loc[(row["seq"], row["pair0_id"]), "original_index"]
                        )
                        view1_individual_idx = int(
                            self.index_to_file_map.loc[(row["seq"], row["pair1_id"]), "original_index"]
                        )

                        # read the individual quantities
                        data_view0 = self.indiv_quantity_reader.read(
                            view0_individual_idx,
                            keys=[f"{k}_{direction}" for k in ["frame_cleanpass", "depth", "pose", "intrinsics"]],
                        )
                        data_view1 = self.indiv_quantity_reader.read(
                            view1_individual_idx,
                            keys=[f"{k}_{direction}" for k in ["frame_cleanpass", "depth", "pose", "intrinsics"]],
                        )

                        # read the optical flow data
                        flow_seq_folder = os.path.join(self.root, "optical_flow", row["seq"])
                        flow_0_to_1 = read_gen(
                            os.path.join(
                                flow_seq_folder,
                                f"into_future/{direction}/OpticalFlowIntoFuture_{row['pair0_id']:04d}_{direction[0].upper()}.pfm",
                            )
                        )
                        flow_1_to_0 = read_gen(
                            os.path.join(
                                flow_seq_folder,
                                f"into_past/{direction}/OpticalFlowIntoPast_{row['pair1_id']:04d}_{direction[0].upper()}.pfm",
                            )
                        )

                        # read the disparity data
                        disparity_seq_folder = os.path.join(self.root, "disparity", row["seq"])
                        disparity_0 = read_gen(
                            os.path.join(disparity_seq_folder, f"{direction}/{row['pair0_id']:04d}.pfm")
                        )
                        disparity_1 = read_gen(
                            os.path.join(disparity_seq_folder, f"{direction}/{row['pair1_id']:04d}.pfm")
                        )

                        # read the disparity change data
                        disparity_change_seq_folder = os.path.join(self.root, "disparity_change", row["seq"])
                        disparity_change_0_to_1 = read_gen(
                            os.path.join(
                                disparity_change_seq_folder, f"into_future/{direction}/{row['pair0_id']:04d}.pfm"
                            )
                        )
                        disparity_change_1_to_0 = read_gen(
                            os.path.join(
                                disparity_change_seq_folder, f"into_past/{direction}/{row['pair1_id']:04d}.pfm"
                            )
                        )

                        # compute the derived data
                        depth_0_in_1 = 1050.0 * 1.0 / (disparity_0 + disparity_change_0_to_1)
                        depth_1_in_0 = 1050.0 * 1.0 / (disparity_1 + disparity_change_1_to_0)

                        depth_validity_0 = (data_view0[f"depth_{direction}"] > 0) & ~torch.isnan(
                            data_view0[f"depth_{direction}"]
                        )
                        depth_validity_1 = (data_view1[f"depth_{direction}"] > 0) & ~torch.isnan(
                            data_view1[f"depth_{direction}"]
                        )

                        view0 = {
                            "img": data_view0[f"frame_cleanpass_{direction}"],
                            "depthmap": data_view0[f"depth_{direction}"],
                            "flow": flow_0_to_1,
                            "depth_in_other_view": depth_0_in_1,
                            "depth_validity": depth_validity_0,
                        }

                        view1 = {
                            "img": data_view1[f"frame_cleanpass_{direction}"],
                            "depthmap": data_view1[f"depth_{direction}"],
                            "flow": flow_1_to_0,
                            "depth_in_other_view": depth_1_in_0,
                            "depth_validity": depth_validity_1,
                        }

                        all_data.append((view0, view1))

                    views = default_collate(all_data)

                    # move to GPU if available
                    selected_device = self.get_random_device()

                    for view in views:
                        for key, value in view.items():
                            if isinstance(value, torch.Tensor):
                                view[key] = value.to(selected_device, non_blocking=True)

                    self.occlusion_calculation_sceneflow(
                        views,
                        depth_error_threshold=0.01,
                        depth_error_temperature=0.01,
                        relative_depth_error_threshold=0.001,
                        opt_iters=1,
                    )

                    for view in views:
                        for key, value in view.items():
                            if isinstance(value, torch.Tensor):
                                view[key] = value.to("cpu", non_blocking=True)

                    keep_field = ["flow", "fov_mask", "non_occluded_mask", "occlusion_supervision_mask"]
                    for view in views:
                        for key in list(view.keys()):
                            if key not in keep_field:
                                del view[key]

                    # save the data to a h5 file
                    view_dict = {"view0": views[0], "view1": views[1]}

                    minibatch_index_end = minibatch_index + len(minibatch)

                    for view_name, view in view_dict.items():
                        for key, value in view.items():
                            if isinstance(value, torch.Tensor):
                                value = value.cpu().numpy()

                            h5_key = f"{direction}_{view_name}_{key}"

                            if h5_key not in h5_file:
                                shape = (len(job_arg),) + value[0].shape
                                dtype = value.dtype
                                chunking = (1,) + value[0].shape
                                h5_file.create_dataset(
                                    h5_key,
                                    shape=shape,
                                    dtype=dtype,
                                    chunks=chunking,
                                    compression="gzip",
                                    compression_opts=9,
                                )
                                print("Creating", h5_key, "with shape", shape)
                            h5_file[h5_key][minibatch_index:minibatch_index_end] = value
                            print("Writing to", h5_key, "from", minibatch_index, "to", minibatch_index_end)

                        h5_file.flush()
                    minibatch_index = minibatch_index_end

    def get_random_device(self):
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            selected_device = random.randint(0, gpu_count - 1)
            return torch.device(f"cuda:{selected_device}")
        else:
            return torch.device("cpu")

    def occlusion_calculation_sceneflow(
        self,
        views,
        depth_error_threshold=0.1,
        depth_error_temperature=0.1,
        relative_depth_error_threshold=0.005,
        opt_iters=5,
    ):
        """
        Generate flow supervision from pointmap, depthmap, intrinsics, depth_validity, and extrinsics

        Args:
        - views (list[dict]): list of views, already batched by the dataloader
        """

        assert len(views) == 2, f"Expected 2 views, to compute flow to other view, got {len(views)} views"

        for view, other_view in zip(views, reversed(views)):
            # required fields: intrinsics, depthmap, depth_in_other_view, flow
            B, H, W = view["depthmap"].shape

            uv = get_meshgrid_torch(W, H, device=view["depthmap"].device) + view["flow"]

            valid_mask = (
                view["depth_validity"] & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
            )

            # compute occlusion based on depth reprojection error thresholding
            # expected_norm_depth = z_depthmap_to_norm_depthmap_batched(view["depth_in_other_view"], view["camera_intrinsics"])[valid_mask]
            # norm_depth_in_other_view = z_depthmap_to_norm_depthmap_batched(
            #     other_view["depthmap"], other_view["camera_intrinsics"]
            # )

            expected_norm_depth = view["depth_in_other_view"][valid_mask]
            norm_depth_in_other_view = other_view["depthmap"]

            error_threshold = (
                depth_error_threshold
                + relative_depth_error_threshold * expected_norm_depth
                - np.log(0.5) * depth_error_temperature
            )

            # to determine occlusion, we will threshold the error between the distance of projected point to the other camera center
            # v.s. the norm-depth value recorded in the otherview's depthmap at the projected pixel location. If they met, then the point
            # is the rendered point in the other view, and is not occluded. Otherwise, it is occluded.

            valid_uv = uv[valid_mask]
            if (
                opt_iters > 0
            ):  # if opt_iters is 0, we will not optimize the uv_residual, and there are no need to create the optimizer and the residual tensor
                uv_residual = torch.zeros_like(
                    valid_uv, requires_grad=True
                )  # we optimize uv_residual to estimate the lower bound of the depth error
                opt = torch.optim.Adam([uv_residual], lr=1e-1, weight_decay=1e-1)
                valid_uv = valid_uv + uv_residual
                opt.zero_grad()

            # select the possibly occluded pixels to check for non-occlusion
            possibly_occluded_mask = valid_mask.clone()
            possible_occlusion_in_valid_pixels = torch.ones(
                size=(valid_mask.sum(),), dtype=torch.bool, device=valid_mask.device
            )
            checked_uv = valid_uv  # [possible_occlusion_in_valid_pixels]
            checked_expected_norm_depth = expected_norm_depth  # [possible_occlusion_in_valid_pixels]
            checked_threshold = error_threshold  # [possible_occlusion_in_valid_pixels]

            opt_iteration = 0
            while True:
                # compute the reprojection error of the selected pixels and check if they are non-occluded
                reprojection_error = compute_reprojection_error(
                    checked_uv, checked_expected_norm_depth, norm_depth_in_other_view, possibly_occluded_mask
                )

                occluded_selected_uv = reprojection_error >= checked_threshold

                # update the occlusion mask, uv_combined, and expected_norm_depth with the non_occluded_selected_uv
                possibly_occluded_mask_new = possibly_occluded_mask.clone()
                possibly_occluded_mask_new[possibly_occluded_mask] = occluded_selected_uv

                possible_occlusion_in_valid_pixels_new = possible_occlusion_in_valid_pixels.clone()
                possible_occlusion_in_valid_pixels_new[possible_occlusion_in_valid_pixels] = occluded_selected_uv

                possibly_occluded_mask = possibly_occluded_mask_new
                possible_occlusion_in_valid_pixels = possible_occlusion_in_valid_pixels_new

                if opt_iters == 0 or opt_iteration >= opt_iters:
                    break

                # optimize the uv_residual
                loss = torch.sum(reprojection_error)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    uv_residual.clamp_(-0.707, 0.707)
                opt.zero_grad()

                opt_iteration += 1

                checked_uv = valid_uv[possible_occlusion_in_valid_pixels]
                checked_expected_norm_depth = expected_norm_depth[possible_occlusion_in_valid_pixels]
                checked_threshold = error_threshold[possible_occlusion_in_valid_pixels]

            # the non-occlsion mask is the invert of the possibly occluded mask
            non_occluded_mask = ~possibly_occluded_mask

            view["non_occluded_mask"] = non_occluded_mask & valid_mask
            view["fov_mask"] = valid_mask

            # compute correspondence validity
            view["depth_validity"] = view["depthmap"] > 0

        # finally, account for depth invalidity in the other view
        for view, other_view in zip(views, reversed(views)):
            other_view_depth_validity = query_projected_mask(
                valid_uv.detach(), other_view["depth_validity"], valid_mask
            )
            view["other_view_depth_validity"] = other_view_depth_validity

            # occlusion should be supervised at
            # 1. self depth is valid, once projected will land out of bound in the other view
            # OR
            # 2. self depth is valid, once projected will land in the bound of other view, landing position shows valid depth

            view["occlusion_supervision_mask"] = (view["depth_validity"] & (~valid_mask)) | (
                valid_mask & other_view_depth_validity
            )


if __name__ == "__main__":
    parser = ProcessMonkaaSingleQuantity.get_default_parser()

    parser.add_argument("--root", type=str, default="/uniflowmatch/data/raw_data/Monkaa")
    parser.add_argument("--output_path", type=str, default="/uniflowmatch/data/monkaa_processed")

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process = ProcessMonkaaSingleQuantity(**vars(args))
    process.run()

    process = ProcessMonkaaFlow(**vars(args), processed_root=args.output_path)
    process.run()
