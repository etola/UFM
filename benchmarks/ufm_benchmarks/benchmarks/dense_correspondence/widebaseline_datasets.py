import os
from typing import Dict, List

import torch
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from ufm_benchmarks.base import UFMDenseBenchmarkIteration, UFMDenseBenchmarkIterationResult
from ufm_benchmarks.benchmarks.dense_correspondence.base import UFMDenseBenchmark, UFMDenseCorrespondenceDataset
from ufm_benchmarks.benchmarks.dense_correspondence.opticalflow_datasets import DenseFlowDatasetWrapper

from uniflowmatch.datasets import DTU, ETH3D, TartanairAssembled
from uniflowmatch.datasets.base.flow_postprocessing import static_flow_postprocessing_pathway


class StaticFlowDatasetWrapper(UFMDenseCorrespondenceDataset):

    def __init__(self, dataset: Dataset, covisible_kwargs: Dict[str, float] = {}, device="cuda:0"):
        self.dataset = dataset
        self.covisible_kwargs = covisible_kwargs
        self.device = device

    def transfer_batch_to_device_and_process(self, batch):
        batch_device = tree_map(lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch)

        batch_device = static_flow_postprocessing_pathway(batch_device, **self.covisible_kwargs)

        return batch_device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> UFMDenseBenchmarkIteration:
        batch = default_collate([self.dataset[idx]])
        batch = self.transfer_batch_to_device_and_process(batch)

        prepared_input = UFMDenseBenchmarkIteration()
        prepared_input.input_img0 = (batch[0]["img"].squeeze(0) * 255.0).to(torch.uint8).cpu()
        prepared_input.input_img1 = (batch[1]["img"].squeeze(0) * 255.0).to(torch.uint8).cpu()
        prepared_input.flow_gt = batch[0]["flow"].squeeze(0).cpu()
        prepared_input.valid_mask = batch[0]["fov_mask"].squeeze(0).cpu()
        prepared_input.covisible_mask = batch[0]["non_occluded_mask"].squeeze(0).cpu()

        return prepared_input


class WideBaselineDenseBenchmark(UFMDenseBenchmark):

    def analyze_results(self, results: List[UFMDenseBenchmarkIterationResult]):

        epe_all = []
        epe_covis = []
        epe_occluded = []

        epe_covis_instance = []
        epe_1px_instance = []
        epe_2px_instance = []
        epe_5px_instance = []

        epe_occluded_instance = []

        for result in results:
            epe = (result.input_data.flow_gt - result.flow_pred).norm(dim=0)
            epe_all.append(epe[result.input_data.valid_mask])

            epe_covis.append(epe[result.input_data.valid_mask & result.input_data.covisible_mask & result.pred_mask])
            epe_occluded.append(
                epe[result.input_data.valid_mask & (~result.input_data.covisible_mask) & result.pred_mask]
            )
            if not result.pred_mask.all():
                print(f"pred_mask: {result.pred_mask.sum()}/{result.pred_mask.numel()}")

            eval_mask = result.input_data.valid_mask & result.input_data.covisible_mask & result.pred_mask
            occluded_eval_mask = (result.input_data.valid_mask & (~result.input_data.covisible_mask)) & result.pred_mask

            if eval_mask.sum() > 50:  # filter out instances with too few covisible pixels
                epe_covis_instance.append(epe[eval_mask].mean())
                epe_1px_instance.append((epe[eval_mask] > 1).float().mean() * 100)
                epe_2px_instance.append((epe[eval_mask] > 2).float().mean() * 100)
                epe_5px_instance.append((epe[eval_mask] > 5).float().mean() * 100)

                epe_occluded_instance.append(epe[occluded_eval_mask].mean())

        epe_all = torch.cat(epe_all)
        epe_covis = torch.cat(epe_covis)
        epe_occluded = torch.cat(epe_occluded)

        self.epe_covis_mean = epe_covis.mean()
        self.epe_occluded_mean = epe_occluded.mean()
        # epe_all_mean = epe_all.mean()
        self.epe_1px = (epe_covis > 1).float().mean() * 100
        self.epe_2px = (epe_covis > 2).float().mean() * 100
        self.epe_5px = (epe_covis > 5).float().mean() * 100

        self.method_time_avg = torch.tensor(self.method_times).mean().item()
        self.method_time_std = torch.tensor(self.method_times).std().item()

        print(f"epe_covis: {self.epe_covis_mean:.3f}")
        print(f"epe_covis_1px: {self.epe_1px:.1f}")
        print(f"epe_covis_2px: {self.epe_2px:.1f}")
        print(f"epe_covis_5px: {self.epe_5px:.1f}")

        print(f"method_time_avg: {self.method_time_avg:.3f}")
        print(f"EPE occluded: {self.epe_occluded_mean:.3f}")

        self.print_stats()

    def print_stats(self):
        pass


class ETH3DDenseBenchmark(WideBaselineDenseBenchmark):

    def __init__(self, eth3d_root: str, eth3d_pairs_root: str, *args, **kwargs):
        dataset = ETH3D(
            ROOT=eth3d_root,
            pair_metadata_path=eth3d_pairs_root,
            split="val",
            seed=777,
            transform="imgnorm",
            data_norm_type="dummy",
            resolution=(560, 420),
            aug_portrait_or_landscape=False,
        )[::3]

        dataset_wrapper = StaticFlowDatasetWrapper(dataset)

        super().__init__(image_dataset=dataset_wrapper, *args, **kwargs)

    def print_stats(self):
        self.output_folder = os.path.join(self.output_root, f"eth3d_dense")
        os.makedirs(self.output_folder, exist_ok=True)

        with open(os.path.join(self.output_folder, f"{self.tested_solution.identifier}.txt"), "w") as f:
            f.write(f"ETH3D Dense Correspondence Benchmark Results\n")
            f.write(f"--------------------------------------------\n")
            f.write(f"EPE Covisible: {self.epe_covis_mean:.3f}\n")
            f.write(f"EPE Covisible > 1px (%): {self.epe_1px:.1f}\n")
            f.write(f"EPE Covisible > 2px (%): {self.epe_2px:.1f}\n")
            f.write(f"EPE Covisible > 5px (%): {self.epe_5px:.1f}\n")
            f.write(f"EPE Occluded: {self.epe_occluded_mean:.3f}\n")
            f.write(f"Method Time Avg (s): {self.method_time_avg:.3f}\n")
            f.write(f"--------------------------------------------\n")


class DTUDenseBenchmark(WideBaselineDenseBenchmark):

    def __init__(self, dtu_root: str, dtu_pairs_root: str, *args, **kwargs):
        dataset = DTU(
            ROOT=dtu_root,
            pair_metadata_path=dtu_pairs_root,
            split="val",
            seed=777,
            transform="imgnorm",
            data_norm_type="dummy",
            resolution=(560, 420),
            aug_portrait_or_landscape=False,
        )[::50]

        dataset_wrapper = StaticFlowDatasetWrapper(
            dataset,
            covisible_kwargs={
                "depth_error_threshold": 12.0,
                "depth_error_temperature": 12.0,
                "relative_depth_error_threshold": 0.01,
                "opt_iters": 0,
            },
        )

        super().__init__(image_dataset=dataset_wrapper, *args, **kwargs)

    def print_stats(self):
        self.output_folder = os.path.join(self.output_root, f"dtu_dense")
        os.makedirs(self.output_folder, exist_ok=True)

        with open(os.path.join(self.output_folder, f"{self.tested_solution.identifier}.txt"), "w") as f:
            f.write(f"DTU Dense Correspondence Benchmark Results\n")
            f.write(f"--------------------------------------------\n")
            f.write(f"EPE Covisible: {self.epe_covis_mean:.3f}\n")
            f.write(f"EPE Covisible > 1px (%): {self.epe_1px:.1f}\n")
            f.write(f"EPE Covisible > 2px (%): {self.epe_2px:.1f}\n")
            f.write(f"EPE Covisible > 5px (%): {self.epe_5px:.1f}\n")
            f.write(f"EPE Occluded: {self.epe_occluded_mean:.3f}\n")
            f.write(f"Method Time Avg (s): {self.method_time_avg:.3f}\n")
            f.write(f"--------------------------------------------\n")


class TAWBImageDataset(UFMDenseCorrespondenceDataset):

    def __init__(self, dataset_root: str, device="cuda:0"):
        self.dataset = TartanairAssembled(
            ROOT=dataset_root,
            split="data",
            resolution=[(560, 420)],
            seed=777,
            transform="imgnorm",
            data_norm_type="dummy",
        )
        self.device = device

    def __len__(self):
        return 1500  # 1500 samples for benchmarking is enough

    def __getitem__(self, idx) -> UFMDenseBenchmarkIteration:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        views = self.dataset[idx]

        image0 = (views[0]["img"] * 255.0).permute(1, 2, 0).to("cpu", torch.uint8)
        image1 = (views[1]["img"] * 255.0).permute(1, 2, 0).to("cpu", torch.uint8)
        flow = views[0]["flow"].to("cpu", torch.float32)
        valid_mask = views[0]["fov_mask"].to("cpu", torch.bool)
        covisible_mask = views[0]["non_occluded_mask"].to("cpu", torch.bool)

        dense_benchmark_input = UFMDenseBenchmarkIteration()
        dense_benchmark_input.input_img0 = image0
        dense_benchmark_input.input_img1 = image1

        dense_benchmark_input.flow_gt = flow
        dense_benchmark_input.valid_mask = valid_mask
        dense_benchmark_input.covisible_mask = covisible_mask

        return dense_benchmark_input


class TAWBDenseBenchmark(WideBaselineDenseBenchmark):

    def __init__(self, dataset_root, *args, **kwargs):
        dataset = TAWBImageDataset(dataset_root)
        super().__init__(image_dataset=dataset, *args, **kwargs)

    def print_stats(self):
        self.output_folder = os.path.join(self.output_root, f"ta_wb_dense")
        os.makedirs(self.output_folder, exist_ok=True)

        with open(os.path.join(self.output_folder, f"{self.tested_solution.identifier}.txt"), "w") as f:
            f.write(f"TA-WB Dense Correspondence Benchmark Results\n")
            f.write(f"--------------------------------------------\n")
            f.write(f"EPE Covisible: {self.epe_covis_mean:.3f}\n")
            f.write(f"EPE Covisible > 1px (%): {self.epe_1px:.1f}\n")
            f.write(f"EPE Covisible > 2px (%): {self.epe_2px:.1f}\n")
            f.write(f"EPE Covisible > 5px (%): {self.epe_5px:.1f}\n")
            f.write(f"EPE Occluded: {self.epe_occluded_mean:.3f}\n")
            f.write(f"Method Time Avg (s): {self.method_time_avg:.3f}\n")
            f.write(f"--------------------------------------------\n")
