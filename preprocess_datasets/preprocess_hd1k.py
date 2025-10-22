import argparse
import json
import os
from glob import glob
from typing import Any, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import HD1K

from uniflowmatch.datasets.utils.frame_utils import read_gen
from uniflowmatch.utils.parallel import IndependentParallelProcessing


class HD1KPreprocessor(IndependentParallelProcessing):

    def __init__(self, hd1k_root: str, output_dir: str, target_shape=(560, 1330), **kwargs):
        self.hd1k_root = hd1k_root
        self.output_dir = output_dir

        self.target_H = target_shape[0]
        self.target_W = target_shape[1]
        self.dataset = HD1K(self.hd1k_root, split="train")

        super().__init__(**kwargs)

    def get_total_job_args(self) -> List[Any]:
        # return a list of arguments to be passed to the final thread executor
        length = len(self.dataset)
        assert length == 1047, f"Expected 1047 samples, got {length}"

        return list(range(length))

    def thread_worker(self, job_arg: Any):
        # process a single job
        img0, img1, flow, valid = self.dataset[job_arg]

        # use lanzcos interpolation to resize the PIL images
        img0 = img0.resize((self.target_W, self.target_H), Image.LANCZOS)
        img1 = img1.resize((self.target_W, self.target_H), Image.LANCZOS)

        # interpolate flow and valid with NEAREST interpolation
        flow_tensor = torch.from_numpy(flow).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid).unsqueeze(0).unsqueeze(1).float()

        original_H, original_W = flow_tensor.shape[-2:]

        flow_tensor = torch.nn.functional.interpolate(flow_tensor, size=(self.target_H, self.target_W), mode="nearest")

        flow_tensor *= torch.tensor([self.target_W / original_W, self.target_H / original_H]).view(1, 2, 1, 1)

        valid_tensor = torch.nn.functional.interpolate(
            valid_tensor, size=(self.target_H, self.target_W), mode="nearest"
        )

        img0 = np.asarray(img0)
        img1 = np.asarray(img1)
        flow = flow_tensor.squeeze(0).numpy()
        valid = valid_tensor.squeeze(0).squeeze(0).bool().numpy()

        np.savez(os.path.join(self.output_dir, f"{job_arg:04d}.npz"), img0=img0, img1=img1, flow=flow, valid=valid)


if __name__ == "__main__":
    parser = HD1KPreprocessor.get_default_parser()

    parser.add_argument(
        "--hd1k_root",
        type=str,
        default="/uniflowmatch/ma_data/HD1K",
        help="Path to the HD1k dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/uniflowmatch/ma_data/hd1k_processed",
        help="Output directory",
    )

    args = parser.parse_args()

    processor = HD1KPreprocessor(**vars(args))
    processor.run()
