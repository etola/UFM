"""
This file implements standard optical flow benchmarks.
"""

import os
import os.path as osp
from glob import glob
from typing import Any, List

import cv2
import numpy as np
import torch
from ufm_benchmarks.base import UFMDenseBenchmarkIteration, UFMDenseBenchmarkIterationResult
from ufm_benchmarks.benchmarks.dense_correspondence.base import UFMDenseBenchmark, UFMDenseCorrespondenceDataset


def readFlowSintel(file_path: str) -> np.ndarray:
    """Read optical flow data from a .flo file."""
    with open(file_path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: {file_path}")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
        flow = np.resize(data, (h, w, 2))
    return flow


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


class DenseFlowDatasetWrapper(UFMDenseCorrespondenceDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        raise NotImplementedError("Length method not implemented.")

    def _getitem(self, idx) -> Any:
        raise NotImplementedError("_getitem method not implemented.")

    def __getitem__(self, idx) -> UFMDenseBenchmarkIteration:
        image0, image1, flow, valid, covisible = self._getitem(idx)

        dense_benchmark_input = UFMDenseBenchmarkIteration()
        dense_benchmark_input.input_img0 = image0
        dense_benchmark_input.input_img1 = image1
        dense_benchmark_input.flow_gt = flow
        dense_benchmark_input.valid_mask = valid
        dense_benchmark_input.covisible_mask = covisible

        return dense_benchmark_input


class Sintel(DenseFlowDatasetWrapper):
    def __init__(self, root: str, split="training", dstype="clean", *args, **kwargs):
        self.img_shape = (436, 1024)

        self.name = "sintel_" + dstype
        self.dstype = dstype

        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)
        occlusion_root = osp.join(root, split, "occlusions")

        self.img_root = image_root
        self.flow_root = flow_root
        self.occlusion_root = occlusion_root

        self.image_list = []
        self.flow_list = []
        self.occ_list = []
        self.extra_info = []

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))

            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                flow_paths = sorted(glob(osp.join(flow_root, scene, "*.flo")))
                self.flow_list += flow_paths

                occlusion_paths = sorted(glob(osp.join(occlusion_root, scene, "*.png")))
                self.occ_list += occlusion_paths

    def __len__(self) -> int:
        return len(self.flow_list)

    def _getitem(self, idx: int):
        """Given an index, return a tuple of (image0, image1, flow, valid, covisible_mask)."""

        # Read image0 and image1
        image0 = cv2.imread(self.image_list[idx][0], cv2.IMREAD_COLOR)
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image0 = torch.from_numpy(image0).to(torch.uint8)  # (H, W, C), torch.uint8

        image1 = cv2.imread(self.image_list[idx][1], cv2.IMREAD_COLOR)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = torch.from_numpy(image1).to(torch.uint8)  # (H, W, C), torch.uint8

        # Read flow
        flow = readFlowSintel(self.flow_list[idx])  # (H, W, 2)
        flow = torch.from_numpy(flow).permute(2, 0, 1).to(torch.float32)  # (2, H, W), torch.float32

        # Read occlusion map
        occlusion = cv2.imread(self.occ_list[idx], cv2.IMREAD_GRAYSCALE)  # (H, W), uint8
        occlusion = torch.from_numpy(occlusion).to(torch.uint8)  # Keep as uint8 for correct comparison

        # Compute non-occluded mask (mask=1 when pixel is not occluded)
        # According to the Sintel dataset, occlusion maps have 0 for non-occluded pixels, and 255 for occluded pixels
        covisible_mask = occlusion == 0  # True where pixel is not occluded
        covisible_mask = covisible_mask.to(torch.bool)  # (H, W), torch.bool

        # Compute validity mask
        # Valid pixels are those that are not occluded
        valid = torch.ones_like(covisible_mask)  # (H, W), torch.bool

        # Ensure the flow contains finite values (optional but recommended)
        finite_flow = torch.isfinite(flow).all(dim=0)  # (H, W), torch.bool
        valid &= finite_flow  # Update valid mask to exclude non-finite flow values

        return image0, image1, flow, valid, covisible_mask


class KITTI(DenseFlowDatasetWrapper):
    def __init__(self, root: str, split="training", *args, **kwargs):
        self.img_shape = (376, 1242)  # KITTI fixed image dimensions
        self.name = "kitti"

        flow_root = osp.join(root, split, "flow_occ")  # Path to flow with valid mask
        covisible_flow_root = osp.join(root, split, "flow_noc")  # Path to flow with only covisible mask

        image_root = osp.join(root, split, "image_2")  # Path to image files

        self.img_root = image_root
        self.flow_root = flow_root

        self.image_list = []
        self.flow_list = []
        self.extra_info = []

        # Gather paths for image and flow files
        images1 = sorted(glob(osp.join(image_root, "*_10.png")))
        images2 = sorted(glob(osp.join(image_root, "*_11.png")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1].replace("_10.png", "")
            self.extra_info.append([frame_id])
            self.image_list.append([img1, img2])

        if split == "training":
            self.flow_list = sorted(glob(osp.join(flow_root, "*_10.png")))
            self.covisible_flow_list = sorted(glob(osp.join(covisible_flow_root, "*_10.png")))

    def __len__(self):
        return len(self.flow_list)

    def _getitem(self, idx):
        """Given an index, return a tuple of (image0, image1, flow, valid, covisible_mask)."""

        # Read image0 and image1
        image0 = cv2.imread(self.image_list[idx][0], cv2.IMREAD_COLOR)
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image0 = torch.from_numpy(image0).to(torch.uint8)

        image1 = cv2.imread(self.image_list[idx][1], cv2.IMREAD_COLOR)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = torch.from_numpy(image1).to(torch.uint8)

        flow, valid = readFlowKITTI(self.flow_list[idx])
        _, covisible = readFlowKITTI(self.covisible_flow_list[idx])

        # Convert flow and valid mask to torch tensors
        flow = torch.from_numpy(flow).permute(2, 0, 1).to(torch.float32)
        valid = torch.from_numpy(valid).to(torch.bool)  # Valid is a 2D mask
        covisible = torch.from_numpy(covisible).to(torch.bool)  # Covisible is a 2D mask

        return image0, image1, flow, valid, covisible


class SintelDenseBenchmark(UFMDenseBenchmark):

    def __init__(self, sintel_root: str, dstype: str = "clean", *args, **kwargs):
        dataset = Sintel(sintel_root, split="training", dstype=dstype)
        super().__init__(image_dataset=dataset, *args, **kwargs)
        self.dstype = dstype

    def analyze_results(self, results: List[UFMDenseBenchmarkIterationResult]):
        epe_all = []
        epe_covis = []

        for result in results:
            epe = (result.input_data.flow_gt - result.flow_pred).norm(dim=0)
            epe_all.append(epe[result.input_data.valid_mask])
            epe_covis.append(epe[result.input_data.valid_mask & result.input_data.covisible_mask])

        epe_all = torch.cat(epe_all)
        epe_covis = torch.cat(epe_covis)

        epe_covis = epe_covis.mean()
        epe_all_mean = epe_all.mean()
        epe_1px = (epe_all > 1).float().mean() * 100
        epe_3px = (epe_all > 3).float().mean() * 100
        epe_5px = (epe_all > 5).float().mean() * 100

        method_time_avg = torch.tensor(self.method_times).mean().item()
        method_time_std = torch.tensor(self.method_times).std().item()

        print(f"epe[covis]: {epe_covis:.3f}")
        print(f"epe[all]: {epe_all_mean:.3f}")
        print(f"epe[all]_1px: {epe_1px:.1f}")
        print(f"epe[all]_3px: {epe_3px:.1f}")
        print(f"epe[all]_5px: {epe_5px:.1f}")

        print(f"method_time_avg: {method_time_avg:.3f}")
        print(f"method_time_std: {method_time_std:.3f}")

        self.output_folder = os.path.join(self.output_root, f"sintel_{self.dstype}")
        os.makedirs(self.output_folder, exist_ok=True)

        with open(os.path.join(self.output_folder, f"{self.tested_solution.identifier}.txt"), "w") as f:
            f.write(f"sintel_{self.dstype} benchmark results for {self.tested_solution.identifier}\n")
            f.write(f"epe[covis]: {epe_covis:.3f}\n")
            f.write(f"epe[all]: {epe_all_mean:.3f}\n")
            f.write(f"epe[all]_1px: {epe_1px:.1f}\n")
            f.write(f"epe[all]_3px: {epe_3px:.1f}\n")
            f.write(f"epe[all]_5px: {epe_5px:.1f}\n")
            f.write(f"method_time_avg: {method_time_avg:.3f}\n")
            f.write(f"method_time_std: {method_time_std:.3f}\n")


class KITTIDenseBenchmark(UFMDenseBenchmark):

    def __init__(self, kitti_root: str, *args, **kwargs):
        dataset = KITTI(kitti_root, split="training")
        super().__init__(image_dataset=dataset, *args, **kwargs)

    def analyze_results(self, results: List[UFMDenseBenchmarkIterationResult]):

        epe_all = []
        epe_covis = []
        gt_norm_all = []

        for result in results:
            epe = (result.input_data.flow_gt - result.flow_pred).norm(dim=0)
            gt_norm = (result.input_data.flow_gt).norm(dim=0)[result.input_data.valid_mask]

            epe_all.append(epe[result.input_data.valid_mask])
            epe_covis.append(epe[result.input_data.valid_mask & result.input_data.covisible_mask])
            gt_norm_all.append(gt_norm)

        # compute F1 EPE, F1 EPE[covis], and F1 EPE percentage.
        F1_epe = torch.tensor([x.mean() for x in epe_all]).mean()
        F1_epe_covis = torch.tensor([x.mean() for x in epe_covis]).mean()

        all_epe = torch.cat(epe_all)
        all_gt_norm = torch.cat(gt_norm_all)

        is_f1_outlier = (all_epe > 3) & (all_epe > 0.05 * all_gt_norm)
        F1_epe_percentage = is_f1_outlier.float().mean() * 100

        # method times
        method_time_avg = torch.tensor(self.method_times).mean().item()
        method_time_std = torch.tensor(self.method_times).std().item()

        print(f"F1 EPE[covis]: {F1_epe_covis:.3f}")
        print(f"F1 EPE: {F1_epe:.3f}")
        print(f"F1 EPE percentage: {F1_epe_percentage:.3f}")

        print(f"method_time_avg: {method_time_avg:.3f}")
        print(f"method_time_std: {method_time_std:.3f}")

        self.output_folder = os.path.join(self.output_root, "kitti")
        os.makedirs(self.output_folder, exist_ok=True)

        with open(os.path.join(self.output_folder, f"{self.tested_solution.identifier}.txt"), "w") as f:
            f.write(f"kitti benchmark results for {self.tested_solution.identifier}\n")
            f.write(f"F1 EPE[covis]: {F1_epe_covis:.3f}\n")
            f.write(f"F1 EPE: {F1_epe:.3f}\n")
            f.write(f"F1 EPE percentage: {F1_epe_percentage:.3f}\n")
            f.write(f"method_time_avg: {method_time_avg:.3f}\n")
            f.write(f"method_time_std: {method_time_std:.3f}\n")
