from typing import Any, List, Sized

import torch
from torch.utils.data import Dataset, Subset
from ufm_benchmarks.base import (
    UFMBenchmarkBase,
    UFMDenseBenchmarkIteration,
    UFMDenseBenchmarkIterationResult,
    UFMSolutionBase,
)


class UFMDenseCorrespondenceDataset(Dataset):

    def __len__(self) -> int:
        raise NotImplementedError("Length method not implemented.")

    def __getitem__(self, idx: int) -> UFMDenseBenchmarkIteration:
        raise NotImplementedError("Get item method not implemented.")


class UFMDenseBenchmark(UFMBenchmarkBase):

    def __init__(
        self,
        solution: UFMSolutionBase,
        output_root: str,
        image_dataset: UFMDenseCorrespondenceDataset,
    ):

        super().__init__(solution=solution, output_root=output_root)

        self.image_dataset = image_dataset

        # create a random, deterministic ordering of the dataset
        random_index = torch.randperm(len(self.image_dataset), generator=torch.Generator().manual_seed(42))
        self.image_dataset = Subset(self.image_dataset, random_index)  # type: ignore

    def benchmark_iteration(
        self, solution: UFMSolutionBase, batch: UFMDenseBenchmarkIteration
    ) -> UFMDenseBenchmarkIterationResult:

        result = solution.solve_correspondence(batch)

        # remove input images from result to save memory
        result.input_data.input_img0 = None
        result.input_data.input_img1 = None

        self.method_times.append(result.method_time_ms)
        return result

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx: int) -> UFMDenseBenchmarkIterationResult:
        input_data: UFMDenseBenchmarkIteration = self.image_dataset[idx]
        return self.benchmark_iteration(self.tested_solution, input_data)

    def analyze_results(self, results: List[UFMDenseBenchmarkIterationResult]):
        raise NotImplementedError("Analyze the results of the benchmark.")
