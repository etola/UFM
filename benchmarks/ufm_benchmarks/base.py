import os
import time
from typing import Any, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


# Basic inputs/outputs for each benchmark
class UFMDenseBenchmarkIteration:
    # input images
    input_img0: Optional[torch.Tensor]  # torch.Tensor, HWC, uint8, cpu
    input_img1: Optional[torch.Tensor]  # torch.Tensor, HWC, uint8, cpu

    # ground-truth quantities
    flow_gt: torch.Tensor  # torch.Tensor, CHW, cpu
    covisible_mask: torch.Tensor  # torch.Tensor, HW, cpu
    valid_mask: torch.Tensor  # torch.Tensor, HW, cpu


class UFMDenseBenchmarkIterationResult:
    input_data: UFMDenseBenchmarkIteration

    # predicted quantities
    flow_pred: torch.Tensor  # torch.Tensor, CHW, cpu
    pred_mask: torch.Tensor  # torch.Tensor, HW, cpu

    method_time_ms: Optional[float] = None  # time taken by the core method (excluding data transfer) in milliseconds


# Base class for solutions
class UFMSolutionBase:

    identifier: str
    device: str = "cuda:0"

    def __init__(self, identifier: str, device: str = "cuda:0"):
        self.identifier = identifier
        self.device = device
        self.last_solution_ms = None
        self.start_time = None

    def start_method_timer(self):
        # we put timer inside methods to exclude data transfer between CPU and GPU
        torch.cuda.synchronize()
        self.start_time = time.time()

    def stop_method_timer(self):
        torch.cuda.synchronize()
        self.last_solution_ms = (time.time() - self.start_time) * 1000
        # print(f"Solution time: {self.last_solution_ms:.2f} ms")

    def solve_correspondence(self, input_data: UFMDenseBenchmarkIteration) -> UFMDenseBenchmarkIterationResult:
        raise NotImplementedError("Solve the correspondence problem.")

    # def solve_fundamental(
    #     self, input_data: UFMFundamentalBenchmarkIterationInput
    # ) -> UFMFundamentalBenchmarkIterationResult:
    #     raise NotImplementedError("Solve the fundamental matrix problem.")


# Base class for benchmarks
class UFMBenchmarkBase(Dataset):
    def __init__(
        self,
        solution: UFMSolutionBase,
        output_root: str,
    ):

        # model to be tested
        self.tested_solution = solution

        # output root folder
        self.output_root = output_root

        # timing functionality
        self.start_time = None
        self.method_times = []

    def __len__(self) -> int:
        raise NotImplementedError("Return the number of testing examples of the benchmark.")

    def __getitem__(self, idx):
        raise NotImplementedError("Return the result of the benchmark at the given index.")

    def analyze_results(self, results: List[Any]):
        raise NotImplementedError("Analyze the results of the benchmark.")

    def run_all(self):

        results = []
        for result_id in tqdm(range(len(self))):
            result = self[result_id]
            results.append(result)

        self.analyze_results(results)

    def stop_method_timer(self):
        torch.cuda.synchronize()
        self.last_solution_ms = (time.time() - self.start_time) * 1000
        # print(f"Solution time: {self.last_solution_ms:.2f} ms")
        self.last_solution_ms = None
        self.start_time = None

    def start_method_timer(self):
        # we put timer inside methods to exclude data transfer between CPU and GPU
        torch.cuda.synchronize()
        self.start_time = time.time()
