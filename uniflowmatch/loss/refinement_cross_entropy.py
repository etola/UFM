from typing import Dict, List
import torch

from uniflowmatch.loss.base import SupervisionBase
from uniflowmatch.models.base import UFMOutputInterface
from uniflowmatch.utils.geometry import get_meshgrid, get_meshgrid_torch


class RefinementCrossEntropyLoss(SupervisionBase):
    def __init__(
        self,
        multiplier: float = 1,
        supervision_range: str = "occlusion",
        supervision_strategy: str = "4_point_average",
    ) -> None:
        """
        Initialize the RefinementCrossEntropyLoss class.

        Args:
        multiplier (float, optional): The multiplier for the loss. Defaults to 1.
        supervision_range (str, optional): Which mask is used in computing the loss. Defaults to "occlusion", alternative is "valid".
        - occlusion: The loss is computed only on the non-occluded pixels.
        - valid: The loss is computed on non-occluded pixels, and occluded but within fov pixels.
        supervise_backward (str): Policy on supervising the predicted backward flow. Defaults to "optional".
        - optional: The backward flow is supervised if the supervision signal exists in batch.
        - always: The backward flow is always supervised.
        - never: The backward flow is never supervised.
        """

        super().__init__(multiplier)

        self.enabled = True
        self.name = "refinement_cross_entropy"

        self.supervision_range = supervision_range
        self.expected_mask_key = "non_occluded_mask" if self.supervision_range == "occlusion" else "fov_mask"

        self.supervision_strategy = supervision_strategy

    def get_supervision_signal(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Method to get the supervision signal given direction.

        Args:
        batch (List[Dict[str, torch.Tensor]]): The input batch dictionary.

        Returns:
        Dict[str, torch.Tensor]: The supervision signal.
        """

        direction_idx = 0 # forward direction

        return {"flow": batch[direction_idx]["flow"], "mask": batch[direction_idx][self.expected_mask_key]}

    def get_model_result(
        self, batch: List[Dict[str, torch.Tensor]], model_result: UFMOutputInterface
    ) -> Dict[str, torch.Tensor]:
        """
        Method to get the result from the model given direction.

        Args:
        batch (List[Dict[str, torch.Tensor]]): The input batch dictionary.
        model_result (UFMOutputInterface): The model result object.

        Returns:
        Dict[str, torch.Tensor]: The model result.
        """
        B = batch[0]["flow"].shape[0]

        assert model_result.classification_refinement is not None

        return {
            "regression_flow_output": model_result.classification_refinement.regression_flow_output[:B],
            "residual": model_result.classification_refinement.residual[:B],
            "log_softmax": model_result.classification_refinement.log_softmax[:B],
        }


    def compute_loss_with_gathered_info(self, result_dict, supervision_dict):
        """
        Method to compute the loss with gathered information.

        Args:
        result_dict (Dict[str, Any]): The result dictionary.
        supervision_dict (Dict[str, Any]): The supervision dictionary.

        Returns:
        Tuple[torch.Tensor, Dict[str, Any]]:
        - loss (torch.Tensor): The computed loss.
        - log_dict (Dict[str, Any]): The additional information.
        """

        reference_flow = supervision_dict["flow"]
        reference_mask = supervision_dict["mask"]
        reference_mask_float = reference_mask.float()

        # handle NaN in the reference flow
        reference_mask = reference_mask & (~reference_flow.isnan().any(dim=1))
        reference_mask = reference_mask & (torch.abs(reference_flow).sum(dim=1) < 2048)

        reference_flow[reference_flow.isnan()] = (
            0  # set nan to 0, will be masked out later by reference_mask or reference_mask_float
        )

        # get the features from the model result
        regression_flow_output = result_dict["regression_flow_output"]
        residual = result_dict["residual"]
        log_softmax = result_dict["log_softmax"]

        # get the dimensions
        P = log_softmax.shape[-1]
        R = (P - 1) // 2
        B, C, H, W = regression_flow_output.shape

        # compute the indices of the fetch
        base_grid_xy = get_meshgrid_torch(W=W, H=H, device=reference_flow.device).permute(2, 0, 1).reshape(1, 2, H, W)

        est_target_coordinate_xy_float = regression_flow_output + base_grid_xy
        est_target_coordinate_xy = est_target_coordinate_xy_float.view(B, 2, H, W)

        gt_target_coordinate_xy = reference_flow + base_grid_xy
        gt_in_local_coordinate_xy = gt_target_coordinate_xy - est_target_coordinate_xy
        gt_in_local_coordinate_ij = gt_in_local_coordinate_xy[:, [1, 0]]

        # compute the expected weights from the local gt coordinate. We use the simplist version for now.
        gt_weights = torch.zeros(B, H, W, P * P, device=reference_flow.device)

        if self.supervision_strategy == "single": # use a nearest pixel as the GT and supervise all the weights to it
            gt_in_coordinate_ij = (gt_in_local_coordinate_ij + P / 2).int()
            gt_in_refinement_range = ((gt_in_coordinate_ij >= 0) & (gt_in_coordinate_ij < P)).all(dim=1)

            valid_coordinates = gt_in_coordinate_ij.permute(0, 2, 3, 1)[gt_in_refinement_range]
            valid_coordinates = valid_coordinates[..., 0] * P + valid_coordinates[..., 1]

            gt_weights[gt_in_refinement_range] = torch.scatter(
                gt_weights[gt_in_refinement_range], -1, valid_coordinates.unsqueeze(-1).long(), 1.0
            )
        elif self.supervision_strategy == "4_point_average": # use adjacent 4 pixels and do bilinear interpolation
            gt_in_coordinate_ij = gt_in_local_coordinate_ij + P / 2
            gt_in_refinement_range = ((gt_in_coordinate_ij >= 0.5) & (gt_in_coordinate_ij < P - 0.5)).all(dim=1)

            valid_coordinates = gt_in_coordinate_ij.permute(0, 2, 3, 1)[gt_in_refinement_range]

            # the start coordinate is the bottom left corner of the 2x2 square
            valid_coordinate_start = (valid_coordinates - 0.5).int() + 0.5
            didj = valid_coordinates - valid_coordinate_start
            di = didj[..., 0]
            dj = didj[..., 1]

            w0 = (1 - di) * (1 - dj)  # for i, j
            w1 = di * (1 - dj)  # for i+1, j
            w2 = (1 - di) * dj  # for i, j+1
            w3 = di * dj  # for i+1, j+1

            ij_base_coordinate = (valid_coordinates - 0.5).int()
            ij_base_coordinate = ij_base_coordinate[..., 0] * P + ij_base_coordinate[..., 1]

            weights = [w0, w1, w2, w3]
            offsets = [0, P, 1, P + 1]

            for w, offset in zip(weights, offsets):
                gt_weights[gt_in_refinement_range] = torch.scatter(
                    gt_weights[gt_in_refinement_range],
                    -1,
                    (ij_base_coordinate + offset).unsqueeze(-1).long(),
                    w.view(-1, 1),
                )
        else:
            raise ValueError(f"Unknown supervision strategy: {self.supervision_strategy}")

        log_softmax = log_softmax.reshape(B, H, W, P * P)

        gt_in_refinement_range = gt_in_refinement_range & reference_mask

        loss_CE = -torch.sum(gt_weights[gt_in_refinement_range] * log_softmax[gt_in_refinement_range], dim=-1)
        loss_CE = torch.sum(loss_CE) / (1e-4 + torch.sum(gt_in_refinement_range))

        ratio_in_refinement_range = torch.sum(gt_in_refinement_range) / (1e-4 + torch.sum(reference_mask_float))

        display_name = "refinement_cross_entropy@" + self.expected_mask_key

        return self.multiplier * loss_CE, {
            display_name: loss_CE.detach().cpu().item(),
            "ratio_in_refinement_range": ratio_in_refinement_range.cpu().item(),
        }

    def __repr__(self):
        return f"{self.multiplier}x({self.name}@{self.supervision_range})"
