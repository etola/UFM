from typing import Any, Dict, List, Optional, Tuple

import torch

from uniflowmatch.loss.base import SupervisionBase
from uniflowmatch.models.base import UFMOutputInterface


class CrossEntropyLoss(SupervisionBase):
    def __init__(
        self,
        multiplier: float = 1,
        target: str = "occlusion",
        target_supervision_mask: Optional[str] = None,
    ) -> None:
        """
        Initialize the CrossEntropy class.

        Args:
            multiplier (float, optional): The multiplier for the loss. Defaults to 1.
            target (str, optional): The target key in the batch dictionary. Defaults to "occlusion".
            target_supervision_mask (Optional[str], optional): The key for the supervision mask. Defaults to None.
        """

        super().__init__(multiplier)

        self.enabled = True
        self.name = target + "_ce"
        self.target = target

        self.batch_mask_key = "non_occluded_mask" if self.target == "occlusion" else "fov_mask"
        self.model_mask_key = "occlusion" if self.target == "occlusion" else "fov"

        self.target_supervision_mask = target_supervision_mask

    def get_supervision_signal(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Method to get the supervision signal.

        Args:
            batch (List[Dict[str, torch.Tensor]]): The input batch dictionary.

        Returns:
            Dict[str, torch.Tensor]: The supervision signal.
        """

        view_idx = 0  # forward direction

        if self.target_supervision_mask is not None:
            return {
                "mask": batch[view_idx][self.batch_mask_key],
                "valid": batch[view_idx][self.target_supervision_mask],
            }
        else:
            return {"mask": batch[view_idx][self.batch_mask_key]}

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

        B = batch[0][self.batch_mask_key].shape[0]
        assert self.batch_mask_key == "non_occluded_mask"
        assert model_result.covisibility is not None
        return {"logits": model_result.covisibility.logits[:B]}

    def compute_loss_with_gathered_info(self, result_dict: Dict[str, torch.Tensor], supervision_dict: Dict[str, torch.Tensor]):
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

        reference_mask = supervision_dict["mask"]
        logits_output = result_dict["logits"].squeeze(1)  # squeeze the channel dimension

        # compute the cross entropy loss
        if self.target_supervision_mask is not None:
            ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_output, reference_mask.float(), reduction="none"
            )
            ce_loss = torch.sum(ce_loss[supervision_dict["valid"]]) / (supervision_dict["valid"].sum() + 1e-6)
        else:
            ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_output, reference_mask.float())

        display_name = f"{self.target}_ce"
        return self.multiplier * ce_loss, {display_name: ce_loss.detach()}