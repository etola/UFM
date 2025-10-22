import argparse
import json
import os
import shutil
from glob import glob
from typing import Any, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from uniflowmatch.datasets.base.flow_postprocessing import (
    compute_reprojection_error,
    get_meshgrid_torch,
    query_projected_mask,
)
from uniflowmatch.datasets.utils.spring_flowio import readDsp5Disp, readFlo5Flow
from uniflowmatch.utils.parallel import IndependentParallelProcessing


class SpringImagePreprocessor(IndependentParallelProcessing):
    def __init__(self, spring_root: str, output_dir: str, **kwargs):
        self.spring_root = spring_root
        self.output_dir = output_dir

        training_folder = os.path.join(self.spring_root, "train")
        self.seq_folders = sorted(glob(os.path.join(training_folder, "*")))

        super().__init__(**kwargs)

    def get_total_job_args(self) -> List[Any]:
        # return a list of arguments to be passed to the final thread executor
        image_files = []
        for base_folder in self.seq_folders:
            frame_left = os.path.join(base_folder, "frame_left")
            frame_right = os.path.join(base_folder, "frame_right")

            # flow_fw_left = os.path.join(base_folder, "flow_FW_left")
            # flow_fw_right = os.path.join(base_folder, "flow_FW_right")
            # flow_bw_left = os.path.join(base_folder, "flow_BW_left")
            # flow_bw_right = os.path.join(base_folder, "flow_BW_right")

            left_frames = sorted(glob(os.path.join(frame_left, "*")))
            right_frames = sorted(glob(os.path.join(frame_right, "*")))

            for left_frame, right_frame in zip(left_frames, right_frames):
                image_files.append(left_frame)
                image_files.append(right_frame)

        return image_files

    def thread_worker(self, job_arg: Any):
        path_parts = job_arg.split("/")
        filename = path_parts[-1]
        direction = path_parts[-2]
        seq_folder = path_parts[-3]

        direction = "left" if direction == "frame_left" else "right"
        filename = f"{filename.split('_')[-1]}"

        split = "train" if int(seq_folder) < 38 else "test"  # reserve some sequences for testing

        output_folder = os.path.join(self.output_dir, split, seq_folder, direction, "images")
        os.makedirs(output_folder, exist_ok=True)

        # no any resize, just copy the image
        img = Image.open(job_arg)

        img.resize((926, 560), Image.Resampling.LANCZOS).save(os.path.join(output_folder, filename))


class SpringPairedFlowPreprocessor(IndependentParallelProcessing):
    def __init__(self, spring_root: str, output_dir: str, **kwargs):
        self.spring_root = spring_root
        self.output_dir = output_dir

        training_folder = os.path.join(self.spring_root, "train")
        self.seq_folders = sorted(glob(os.path.join(training_folder, "*")))

        super().__init__(**kwargs)

    def get_total_job_args(self) -> List[Any]:
        # return a list of arguments to be passed to the final thread executor
        seq_pairs = []
        for base_folder in self.seq_folders:
            frame_left = os.path.join(base_folder, "frame_left")
            left_frames = sorted(glob(os.path.join(frame_left, "*")))

            frame_ids = [int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in left_frames]

            for frame_id, next_frame_id in zip(frame_ids[:-1], frame_ids[1:]):
                seq = base_folder.split("/")[-1]
                seq_pairs.append((seq, frame_id, next_frame_id))

        return seq_pairs

    def thread_worker(self, job_arg: Any):

        seq, frame_id0, frame_id1 = job_arg
        split = "train" if int(seq) < 38 else "test"  # reserve some sequences for testing

        # collect flow information and assemble into a npz file
        for direction in ["left", "right"]:
            # flow data
            flow_fwd_filepath = os.path.join(
                self.spring_root, "train", seq, f"flow_FW_{direction}", f"flow_FW_{direction}_{frame_id0:04d}.flo5"
            )
            flow_bwd_filepath = os.path.join(
                self.spring_root, "train", seq, f"flow_BW_{direction}", f"flow_BW_{direction}_{frame_id1:04d}.flo5"
            )

            # occlusion and fov data
            occ_fov_fwd_filepath = os.path.join(
                self.spring_root,
                "train",
                seq,
                "maps",
                f"matchmap_flow_FW_{direction}",
                f"matchmap_flow_FW_{direction}_{frame_id0:04d}.png",
            )
            occ_fov_bwd_filepath = os.path.join(
                self.spring_root,
                "train",
                seq,
                "maps",
                f"matchmap_flow_BW_{direction}",
                f"matchmap_flow_BW_{direction}_{frame_id1:04d}.png",
            )

            # intrinsics data(to convert disparity into depth)
            intrinsics_filepath = os.path.join(self.spring_root, "train", seq, "cam_data", "intrinsics.txt")

            # read them into pytorch tensors
            flow_fwd = readFlo5Flow(flow_fwd_filepath)
            flow_bwd = readFlo5Flow(flow_bwd_filepath)

            occ_fov_fwd = cv2.imread(occ_fov_fwd_filepath)
            occ_fov_fwd = cv2.cvtColor(occ_fov_fwd, cv2.COLOR_BGR2RGB)

            occ_fov_bwd = cv2.imread(occ_fov_bwd_filepath)
            occ_fov_bwd = cv2.cvtColor(occ_fov_bwd, cv2.COLOR_BGR2RGB)

            covisible_fwd = occ_fov_fwd[..., 0] == 0
            fov_fwd = occ_fov_fwd[..., 1] == 0

            covisible_bwd = occ_fov_bwd[..., 0] == 0
            fov_bwd = occ_fov_bwd[..., 1] == 0

            occlusion_supervision_mask_fwd = np.ones_like(covisible_fwd)
            occlusion_supervision_mask_bwd = np.ones_like(covisible_bwd)

            view0 = {
                "flow": torch.from_numpy(flow_fwd.astype(np.float32)),
                "fov_mask": torch.from_numpy(fov_fwd.astype(np.bool_)),
                "non_occluded_mask": torch.from_numpy(covisible_fwd.astype(np.bool_)),
                "occlusion_supervision_mask": torch.from_numpy(occlusion_supervision_mask_fwd.astype(np.bool_)),
            }

            view1 = {
                "flow": torch.from_numpy(flow_bwd.astype(np.float32)),
                "fov_mask": torch.from_numpy(fov_bwd.astype(np.bool_)),
                "non_occluded_mask": torch.from_numpy(covisible_bwd.astype(np.bool_)),
                "occlusion_supervision_mask": torch.from_numpy(occlusion_supervision_mask_bwd.astype(np.bool_)),
            }

            views = [view0, view1]

            keep_field = ["flow", "fov_mask", "non_occluded_mask", "occlusion_supervision_mask"]
            for view in views:
                for key in list(view.keys()):
                    if key not in keep_field:
                        del view[key]
                    else:
                        view[key] = view[key]

            # apply resizing, and move to cpu and numpy
            for view in views:
                for k, v in view.items():
                    if k == "flow":
                        # need to resize image-wise, and adjust the flow according to X and Y scale
                        view[k] = (
                            torch.nn.functional.interpolate(
                                v.unsqueeze(0).permute(0, 3, 1, 2), size=(560, 926), mode="nearest-exact"
                            )[0]
                            .cpu()
                            .numpy()
                        )

                        view[k][0] *= 926 / v.shape[1] * 2  # flow have double the resolution then the image
                        view[k][1] *= 560 / v.shape[0] * 2
                    else:
                        # resize the image-wise, and squeeze the channel.
                        view[k] = (
                            (
                                torch.nn.functional.interpolate(
                                    v.float().unsqueeze(0).unsqueeze(1),
                                    size=(560, 926),
                                    mode="bilinear",
                                    align_corners=False,
                                    antialias=True,
                                )[0, 0]
                                > 0.5
                            )
                            .cpu()
                            .numpy()
                        )

            # deal with nan values in the flow
            for view in views:
                invalid_flow = np.isnan(views[0]["flow"]).any(axis=0)
                for key in ["fov_mask", "non_occluded_mask", "occlusion_supervision_mask"]:
                    view[key][invalid_flow] = False

            # save the processed data into npz
            output_folder = os.path.join(self.output_dir, split, seq, direction, "flow")
            os.makedirs(output_folder, exist_ok=True)

            print(output_folder)

            np.savez(
                os.path.join(output_folder, f"{frame_id0:04d}_{frame_id1:04d}.npz"),
                **{f"{k}_{i}": v for i, view in enumerate(views) for k, v in view.items()},
            )


if __name__ == "__main__":
    parser = SpringImagePreprocessor.get_default_parser()

    parser.add_argument(
        "--spring_root",
        type=str,
        default="/uniflowmatch/data/raw_data/Spring/spring",
        help="Path to the Spring dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/uniflowmatch/data/spring_processed",
        help="Output directory",
    )

    args = parser.parse_args()

    processor = SpringImagePreprocessor(**vars(args))
    processor.run()

    processor2 = SpringPairedFlowPreprocessor(**vars(args))
    processor2.run()
