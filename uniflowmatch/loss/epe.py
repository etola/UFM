from functools import lru_cache, partial
from typing import Any, Dict, List, Tuple

import torch

from uniflowmatch.loss.base import SupervisionBase
from uniflowmatch.models.base import UFMOutputInterface


class FlowEPELoss(SupervisionBase):
    def __init__(
        self,
        multiplier: float = 1,
        supervision_range: str = "occlusion",
        minimum_mask_threshold: float = 0.1,
        fov_threshold: float = 0.0,
    ) -> None:
        """
        Initialize the FlowEPE class.

        Args:
            multiplier (float, optional): The multiplier for the loss. Defaults to 1.
            supervision_range (str, optional): Which mask is used in computing the loss. Defaults to "occlusion", alternative is "valid".
            - occlusion: The loss is computed only on the non-occluded pixels.
            - valid: The loss is computed on non-occluded pixels, and occluded but within fov pixels.
            minimum_mask_threshold (float): Minimum mask threshold.
            fov_threshold (float): Field of view threshold.
        """

        super().__init__(multiplier)

        self.enabled = True
        self.name = "flow_epe"

        self.supervision_range = supervision_range
        self.fov_threshold = fov_threshold

        assert self.supervision_range in ["occlusion", "fov_mask", "all", "occlusion_for_wb_all_pixel_for_of"]

        if (self.supervision_range == "occlusion") or (self.supervision_range == "occlusion_for_wb_all_pixel_for_of"):
            self.expected_mask_key = "non_occluded_mask"
        elif self.supervision_range == "fov_mask":
            self.expected_mask_key = "fov_mask"
        else:
            self.expected_mask_key = None

        self.minimum_mask_threshold = minimum_mask_threshold

    def get_supervision_signal(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Method to get the supervision signal.

        Args:
            batch (List[Dict[str, torch.Tensor]]): The input batch dictionary.

        Returns:
            Dict[str, torch.Tensor]: The supervision signal.
        """

        direction_idx = 0 # only forward direction
        if self.expected_mask_key is None:
            # compute a "fov" mask with tolerance to the out of bound pixels
            mask = self.compute_fov_mask_batched(
                batch[direction_idx]["img"],
                batch[1 - direction_idx]["img"],
                torch.ones_like(batch[direction_idx]["flow"][:, 0], dtype=torch.bool),
                batch[direction_idx]["flow"],
                self.fov_threshold,
            )
        else:
            mask = batch[direction_idx][self.expected_mask_key]

        return {"flow": batch[direction_idx]["flow"], "mask": mask}

    def compute_fov_mask_batched(self, base_image: torch.Tensor, other_image: torch.Tensor, valid: torch.Tensor, flow: torch.Tensor, tolerance: float=0.0):
        """
        Compute a field-of-view (FOV) mask for a batch of images based on the flow vectors.

        Args:
            base_image (torch.Tensor): The base image tensor of shape (B, C, H, W).
            other_image (torch.Tensor): The other image tensor of shape (B, C, H, W).
            valid (torch.Tensor): A boolean tensor of shape (B, H, W) indicating valid pixels in the base image.
            flow (torch.Tensor): The flow tensor of shape (B, 2, H, W).
            tolerance (float): Tolerance value for out of bound pixels. Defaults to 0.0.

        Returns:
            torch.Tensor: A boolean tensor of shape (B, H, W) indicating the FOV mask.
        """
        assert base_image.shape == other_image.shape

        fov_mask = valid.clone()
        B, C, H, W = base_image.shape
        flow_base_coords = self.get_meshgrid(H, W, base_image.device)

        flow_base_coords = torch.stack((flow_base_coords[..., -1], flow_base_coords[..., 0]), dim=-1)  # H,W -> X, Y
        flow_target = flow_base_coords.permute(2, 0, 1).unsqueeze(0) + flow

        fov_mask[
            (flow_target[:, 0] < -tolerance * W)
            | (flow_target[:, 1] < -tolerance * H)
            | (flow_target[:, 0] >= (1 + tolerance) * W)
            | (flow_target[:, 1] >= (1 + tolerance) * H)
        ] = False

        return fov_mask

    @lru_cache(maxsize=5)
    def get_meshgrid(self, H: int, W: int, device: str) -> torch.Tensor:
        """
        Get a meshgrid of coordinates.
        
        Args:
            H: Height dimension.    
            W: Width dimension.
            device: Device for tensor placement.
            
        Returns:
            torch.Tensor: Meshgrid tensor.
        """
        return torch.stack(
            torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"),
            dim=-1,
        ).float()

    def get_model_result(
        self, batch: List[Dict[str, torch.Tensor]], model_result: UFMOutputInterface
    ) -> Dict[str, torch.Tensor]:
        """
        Method to get the result from the model.

        Args:
            batch (List[Dict[str, torch.Tensor]]): The input batch dictionary.
            model_result (UFMOutputInterface): The model result interface.

        Returns:
            Dict[str, torch.Tensor]: The model result.
        """
        B = batch[0]["flow"].shape[0]

        assert model_result.flow is not None and model_result.flow.flow_output is not None
        return {"flow_output": model_result.flow.flow_output[:B]}

    def compute_loss_with_gathered_info(self, result_dict: Dict[str, torch.Tensor], supervision_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Method to compute the loss with gathered information.

        Args:
            result_dict (Dict[str, torch.Tensor]): The result dictionary.
            supervision_dict (Dict[str, torch.Tensor]): The supervision dictionary.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
            - loss (torch.Tensor): The computed loss.
            - log_dict (Dict[str, Any]): The additional information.
        """

        reference_flow = supervision_dict["flow"]
        reference_mask = supervision_dict["mask"]
        reference_mask_float = reference_mask.float()
        flow_output = result_dict["flow_output"]

        reference_mask = reference_mask & (~reference_flow.isnan().any(dim=1))
        reference_mask = reference_mask & (torch.abs(reference_flow).sum(dim=1) < 2048)

        reference_flow[reference_flow.isnan()] = (
            0  # set nan to 0, will be masked out later by reference_mask or reference_mask_float
        )

        if torch.isnan(flow_output).any():
            print("flow_output contains nan", torch.isnan(flow_output).sum())

        flow_output[torch.isnan(flow_output)] = 0  # set nan to 0, the canonical guess for somehow invalid flow

        reference_flow[torch.abs(reference_flow) > 2048] = 0  # for kitti training only to transmit invalidity

        # compute EPE loss where mask is true
        flow_epe = torch.linalg.norm(flow_output - reference_flow, dim=1)

        instance_validity_ratio = torch.mean(reference_mask_float, dim=(-1, -2))
        instance_validity_sum = torch.sum(reference_mask_float, dim=(-1, -2))
        instance_validity = instance_validity_ratio > self.minimum_mask_threshold

        px_mean_epe = flow_epe[reference_mask].mean()

        # mask out the occluded pixels
        mean_epe = (flow_epe * reference_mask_float).sum() / (torch.sum(reference_mask_float) + 1e-4)

        display_name = "flow_epe@" + self.expected_mask_key if self.expected_mask_key is not None else "flow_epe@all"

        pixel_outlier_threshold = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        pixel_outlier_count = []
        outlier_dict = {}

        for i, threshold in enumerate(pixel_outlier_threshold):
            pixel_outlier_count.append((flow_epe[reference_mask] > threshold).float().sum().item())

            if reference_mask.sum() > 0:
                pixel_outlier_count[-1] /= reference_mask.sum().item()

            outlier_dict[f"pixel_outlier/lv{i+1}"] = pixel_outlier_count[-1]

        information_dict = {
            display_name: mean_epe.detach().cpu().item(),
            "px_" + display_name: px_mean_epe.detach().cpu().item(),
        }

        information_dict.update(outlier_dict)

        return mean_epe, information_dict

    def __repr__(self):
        return f"{self.multiplier}x({self.name}@{self.supervision_range})"


class RobustRegressionLoss(FlowEPELoss):

    def __init__(self, c: float = 0.03, alpha: float = 0.5, target="flow_output", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.alpha = alpha

        self.name = "robust_regression_loss"
        self.target = target

    def get_model_result(
        self, batch: List[Dict[str, torch.Tensor]], model_result: UFMOutputInterface
    ) -> Dict[str, torch.Tensor]:
        """
        Method to get the result from the model.

        Args:
            batch (List[Dict[str, torch.Tensor]]): The input batch dictionary.
            model_result (UFMOutputInterface): The model result interface.

        Returns:
            Dict[str, torch.Tensor]: The model result.
        """
        B = batch[0]["flow"].shape[0]

        if self.target == "flow_output":
            assert model_result.flow is not None and model_result.flow.flow_output is not None
            flow_batch = model_result.flow.flow_output
        elif self.target == "regression_flow_output":
            assert model_result.classification_refinement is not None and model_result.classification_refinement.regression_flow_output is not None
            flow_batch = model_result.classification_refinement.regression_flow_output
        else:
            raise ValueError("Invalid target selection")

        assert flow_batch is not None
        return {"flow_output": flow_batch[:B]}

    def compute_loss_with_gathered_info(self, result_dict: Dict[str, torch.Tensor], supervision_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Method to compute the loss with gathered information.

        Args:
        result_dict (Dict[str, torch.Tensor]): The result dictionary.
        supervision_dict (Dict[str, Any]): The supervision dictionary.

        Returns:
        Tuple[torch.Tensor, Dict[str, Any]]:
        - loss (torch.Tensor): The computed loss.
        - log_dict (Dict[str, Any]): The additional information.
        """

        reference_flow = supervision_dict["flow"]
        reference_mask = supervision_dict["mask"]
        reference_mask_float = reference_mask.float()
        flow_output = result_dict["flow_output"]

        reference_mask = reference_mask & (~reference_flow.isnan().any(dim=1))
        reference_mask = reference_mask & (torch.abs(reference_flow).sum(dim=1) < 2048)

        reference_flow[reference_flow.isnan()] = (
            0  # set nan to 0, will be masked out later by reference_mask or reference_mask_float
        )

        if torch.isnan(flow_output).any():
            print("flow_output contains nan", torch.isnan(flow_output).sum())

        flow_output[torch.isnan(flow_output)] = 0  # set nan to 0, the canonical guess for somehow invalid flow
        reference_flow[torch.abs(reference_flow) > 2048] = 0

        # compute EPE loss where mask is true
        flow_epe = torch.linalg.norm(flow_output - reference_flow, dim=1)

        # compute robust regression loss
        error_scaled = torch.sum(((flow_output - reference_flow) / self.c) ** 2, dim=1)
        robust_loss = (
            abs(self.alpha - 2) / self.alpha * (torch.pow(error_scaled / abs(self.alpha - 2) + 1, self.alpha / 2) - 1)
        )

        mean_robust_loss = (robust_loss * reference_mask_float).sum() / (torch.sum(reference_mask_float) + 1e-4)

        px_mean_epe = flow_epe[reference_mask].mean()

        # mask out the occluded pixels
        mean_epe = (flow_epe * reference_mask_float).sum() / (torch.sum(reference_mask_float) + 1e-4)

        display_name = (
            "robust_epe@" + self.expected_mask_key if self.expected_mask_key is not None else "robust_epe@all"
        )
        epe_display_name = (
            f"flow_epe[{self.target}]@" + self.expected_mask_key
            if self.expected_mask_key is not None
            else f"flow_epe[{self.target}]@all"
        )

        pixel_outlier_threshold = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        pixel_outlier_count = []
        outlier_dict = {}

        for i, threshold in enumerate(pixel_outlier_threshold):
            pixel_outlier_count.append((flow_epe[reference_mask] > threshold).float().sum().item())

            if reference_mask.sum() > 0:
                pixel_outlier_count[-1] /= reference_mask.sum().item()

            outlier_dict[f"{self.target}_pixel_outlier/lv{i+1}"] = pixel_outlier_count[-1]

        information_dict = {
            display_name: mean_robust_loss.detach().cpu().item(),
            epe_display_name: mean_epe.detach().cpu().item(),
            "px_" + display_name: px_mean_epe.detach().cpu().item(),
        }

        information_dict.update(outlier_dict)

        return mean_robust_loss, information_dict