from typing import List, Optional, Tuple

import torch
from ufm_benchmarks.base import UFMDenseBenchmarkIteration, UFMDenseBenchmarkIterationResult, UFMSolutionBase

from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence, UniFlowMatchModelsBase


class UFMSolution(UFMSolutionBase):

    def __init__(
        self,
        hf_repo,
    ):
        identifier = hf_repo.split("/")[-1]
        super().__init__(identifier=f"{identifier}")

        if "Refine" in hf_repo:
            self.model = UniFlowMatchClassificationRefinement.from_pretrained(hf_repo)
        else:
            self.model = UniFlowMatchConfidence.from_pretrained(hf_repo)
        self.model: UniFlowMatchModelsBase

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def solve_correspondence(self, input_data: UFMDenseBenchmarkIteration) -> UFMDenseBenchmarkIterationResult:

        assert input_data.input_img0 is not None and input_data.input_img1 is not None, "Input images must be provided"

        # Move data to device
        input_data.input_img0 = input_data.input_img0.to(self.device)
        input_data.input_img1 = input_data.input_img1.to(self.device)

        self.start_method_timer()

        ufm_output = self.model.predict_correspondences_batched(
            input_data.input_img0, input_data.input_img1, data_norm_type=None
        )

        self.stop_method_timer()

        result = UFMDenseBenchmarkIterationResult()
        result.input_data = input_data
        result.flow_pred = ufm_output.flow.flow_output[0].cpu()
        result.pred_mask = torch.ones_like(input_data.valid_mask, dtype=torch.bool, device="cpu")
        result.method_time_ms = self.last_solution_ms

        return result
