import math
from typing import Any, Dict, List, Tuple

import flow_vis
import torch
import wandb

from uniflowmatch.loss.base import SupervisionBase
from uniflowmatch.models.base import UFMOutputInterface
from uniflowmatch.utils.geometry import get_meshgrid, get_meshgrid_torch


def obtain_neighboirhood_features_batched(
    features: torch.Tensor, target_xy: torch.Tensor, validity: torch.Tensor, patch_size: int = 7
) -> torch.Tensor:
    """
    Given a feature field and a set of XY location with validity, obtain their neighborhood
    feature map for those valid queries(by bilinear interpolation).

    Args:
    - features: BCHW torch.Tensor containing features to be interpolated
    - target_xy: BHW2 torch.tensor with center of pixel have coordinate 0.5 specifying location
    - validity: BHW torch.Tensor denoting which target is a valid position to retrieve its feature
    - patch_size: int, size of the local neighborhood

    Return:
    - sampled_features: NPPC tensor containing local patch features only at the valid position
    """

    B, C, H, W = features.shape
    P = patch_size

    # Obtain XY location in the source for all valid target locations
    source_location = torch.argwhere(validity)
    sel_target_xy = target_xy[validity].view(-1, 1, 2) # shape: L, 1, 2

    # a local P x P grid with (0,0) at the center, shape: 1, P*P, 2
    local_grid = get_meshgrid_torch(W=(P + 1), H=(P + 1), device=target_xy.device).view(1, -1, 2) - (P // 2)

    # add local grid to each target location.
    all_query_xy_int = ((local_grid - 0.5) + sel_target_xy).long() # shape: L, P*P, 2
    all_query_b_int = source_location[:, 0:1].repeat(1, (P + 1) * (P + 1))

    # prepare output feature tensor
    features_output = torch.zeros(
        (source_location.shape[0], (P + 1) * (P + 1), C), device=features.device, dtype=torch.bfloat16
    )

    # Filter out-of-boundary query
    query_valid = (
        (all_query_xy_int[..., 0] >= 0)
        & (all_query_xy_int[..., 0] < W)
        & (all_query_xy_int[..., 1] >= 0)
        & (all_query_xy_int[..., 1] < H)
    )

    query_valid_b = all_query_b_int[query_valid]
    query_valid_xy = all_query_xy_int[query_valid]

    # index and fetch features
    features_valid = features[query_valid_b, :, query_valid_xy[:, 1], query_valid_xy[:, 0]]

    # construct output feature tensor and assign to valid locations
    features_output[query_valid] = features_valid.bfloat16()
    features_output = features_output.view(-1, P + 1, P + 1, C)

    # bilinear interpolation
    sel_target_xy = sel_target_xy.view(-1, 2)
    remainder = sel_target_xy - 0.5 - (sel_target_xy - 0.5).floor()
    remainder = remainder.view(-1, 1, 1, 2)

    rx_1m = 1 - remainder[..., 0]
    ry_1m = 1 - remainder[..., 1]

    interp_feat = (
        (rx_1m * ry_1m).unsqueeze(-1) * features_output[:, :-1, :-1, :]
        + (rx_1m * remainder[..., 1]).unsqueeze(-1) * features_output[:, 1:, :-1, :]
        + (remainder[..., 0] * ry_1m).unsqueeze(-1) * features_output[:, :-1, 1:, :]
        + (remainder[..., 0] * remainder[..., 1]).unsqueeze(-1) * features_output[:, 1:, 1:, :]
    )

    return interp_feat


class RefinementCrossEntropyLossEfficient(SupervisionBase):
    def __init__(
        self,
        multiplier: float = 1,
        supervision_range: str = "occlusion",
        supervision_strategy: str = "4_point_average",
    ) -> None:
        """
        Initialize the RefinementCrossEntropyLossEfficient class.

        This class differs from the original by only computing refinement on location that should be optimized
        (in supervision_range and flow prediction within error bound). This avoids unnecessary refinement result
        at places we do not intend to optimize at the first place.

        Args:
        multiplier (float, optional): The multiplier for the loss. Defaults to 1.
        supervision_range (str, optional): Which mask is used in computing the loss. Defaults to "occlusion", alternative is "valid".
        - occlusion: The loss is computed only on the non-occluded pixels.
        - valid: The loss is computed on non-occluded pixels, and occluded but within fov pixels.
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
        batch (List[Dict[str, torch.Tensor]): The input batch dictionary.

        Returns:
        Dict[str, torch.Tensor]: The supervision signal.
        """

        direction_idx = 0 # forward only

        return {
            "flow": batch[direction_idx]["flow"],
            "mask": batch[direction_idx][self.expected_mask_key],
            "suitable_for_refinement": batch[direction_idx]["suitable_for_refinement"],
        }

    def get_model_result(
        self, batch: List[Dict[str, torch.Tensor]], model_result: UFMOutputInterface
    ) -> Dict[str, torch.Tensor]:
        """
        Method to get the result from the model given direction.

        Args:
        batch (Dict[str, Any]): The input batch dictionary.
        model_result (Dict[str, Any]): The model result dictionary.

        Returns:
        Dict[str, torch.Tensor]: The model result.
        """
        B = batch[0]["flow"].shape[0]

        assert model_result.classification_refinement is not None
        assert model_result.classification_refinement.temperature is not None
        assert model_result.classification_refinement.attention_bias is not None

        return {
            "regression_flow_output": model_result.classification_refinement.regression_flow_output[:B],
            "feature_map_0": model_result.classification_refinement.feature_map_0,
            "feature_map_1": model_result.classification_refinement.feature_map_1,
            "attention_bias": model_result.classification_refinement.attention_bias,
            "temperature": model_result.classification_refinement.temperature,
        }
     

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

        # handle NaN in the reference flow
        reference_mask = reference_mask & (~reference_flow.isnan().any(dim=1))
        reference_mask = reference_mask & (torch.abs(reference_flow).sum(dim=1) < 2048)

        reference_flow[reference_flow.isnan()] = (
            0  # set nan to 0, will be masked out later by reference_mask or reference_mask_float
        )

        # can this dataset be used for refinement training? 
        suitable_for_refinement = supervision_dict["suitable_for_refinement"]

        # get the features from the model result
        regression_flow_output = result_dict["regression_flow_output"]

        feature_map_0 = result_dict["feature_map_0"]
        feature_map_1 = result_dict["feature_map_1"]
        attention_bias = result_dict["attention_bias"]
        temperature = result_dict["temperature"]

        # get the dimensions
        P = int(math.sqrt(attention_bias.shape[0] + 1e-6))
        R = (P - 1) // 2
        B, _, H, W = regression_flow_output.shape

        # compute the indices of the fetch
        base_grid_xy = get_meshgrid_torch(W=W, H=H, device=reference_flow.device).view(1, H, W, 2)
        est_target_coordinate_xy = regression_flow_output.detach().permute(0, 2, 3, 1) + base_grid_xy

        flow_residual = (reference_flow - regression_flow_output).detach()
        optimizable_flow_residual = (flow_residual >= -P / 2 + 0.5 + 1e-4).all(dim=1) & (
            flow_residual < P / 2 - 0.5 - 1e-4
        ).all(dim=1)
        optimizable_flow_residual &= reference_mask

        # Apply dataset-sepecific setting to train only on dataset that are accurate enough for refinement training
        optimizable_flow_residual[~suitable_for_refinement] = False

        # print("Number of pixels in refinement range:", optimizable_flow_residual.sum().item())
        optimizable_flow_residual = self.cap_mask_entry(optimizable_flow_residual, max_true=400000)

        display_name = "refinement_cross_entropy@" + self.expected_mask_key
        if optimizable_flow_residual.sum() == 0:
            return 0 * self.multiplier * torch.sum(feature_map_0), {
                display_name: 0.0,
                "ratio_in_refinement_range": 0.0,
            } # dummy loss to avoid error

        # Select the features according to the optimizable_flow_residual mask
        source_feature_opt = (
            feature_map_0.permute(0, 2, 3, 1).reshape(B, H * W, -1)[optimizable_flow_residual.view(B, H * W)].float()
        )

        target_feature_opt = obtain_neighboirhood_features_batched(
            features=feature_map_1,
            target_xy=est_target_coordinate_xy + 0.5,
            validity=optimizable_flow_residual,
            patch_size=P,
        )

        L = source_feature_opt.shape[0]

        # compute log-softmax attention with the same protocol of the model
        attention_score = (
            torch.matmul(target_feature_opt.view(L, P * P, -1), source_feature_opt.unsqueeze(-1)).squeeze(-1)
            / temperature
        ) + attention_bias

        log_softmax = torch.nn.functional.log_softmax(attention_score, dim=-1)

        flow_residual_valid = flow_residual.permute(0, 2, 3, 1)[optimizable_flow_residual]

        # compute the expected weights from the local gt coordinate. We use the simplist version for now.
        gt_weights = torch.zeros(L, P * P, device=reference_flow.device)

        if self.supervision_strategy == "4_point_average":
            valid_coordinates = flow_residual_valid + P / 2

            # the start coordinate is the bottom left corner of the 2x2 square
            valid_coordinate_start = (valid_coordinates - 0.5).int() + 0.5
            dxdy = valid_coordinates - valid_coordinate_start
            di = dxdy[..., 1]
            dj = dxdy[..., 0]

            w0 = (1 - di) * (1 - dj)  # for i, j
            w1 = di * (1 - dj)  # for i+1, j
            w2 = (1 - di) * dj  # for i, j+1
            w3 = di * dj  # for i+1, j+1

            xy_base_coordinate = (valid_coordinates - 0.5).int()
            ij_base_coordinate = xy_base_coordinate[..., 1] * P + xy_base_coordinate[..., 0]

            weights = [w0, w1, w2, w3]
            offsets = [0, P, 1, P + 1]

            for w, offset in zip(weights, offsets):
                indexed_position = (ij_base_coordinate + offset).unsqueeze(-1).long()

                if not (indexed_position.min().cpu().item() >= 0 and indexed_position.max().cpu().item() < P * P):
                    print("something is wrong in indexing range!")

                gt_weights = torch.scatter(
                    gt_weights,
                    -1,
                    indexed_position,
                    w.view(-1, 1),
                )
        else:
            raise ValueError(f"Unknown supervision strategy: {self.supervision_strategy}")

        log_softmax = log_softmax.reshape(-1, P * P)

        loss_CE = -torch.sum(gt_weights * log_softmax, dim=-1)
        loss_CE = torch.sum(loss_CE) / (1e-4 + L)

        ratio_in_refinement_range = L / (1e-4 + torch.sum(reference_mask))

        return self.multiplier * loss_CE, {
            display_name: loss_CE.detach().cpu().item(),
            "ratio_in_refinement_range": ratio_in_refinement_range.cpu().item(),
        }

    def cap_mask_entry(self, mask: torch.Tensor, max_true: int = 100) -> torch.Tensor:
        """
        Cap the number of True entries in the mask to a maximum of `max_true`.
        """
        num_true = mask.sum().item()

        if num_true > max_true:
            keep_ratio = max_true / num_true
            positive_keep_mask = torch.rand(num_true, device=mask.device) < keep_ratio
            mask[mask.clone()] = positive_keep_mask
        return mask

    def __repr__(self):
        return f"{self.multiplier}x({self.name}@{self.supervision_range})"