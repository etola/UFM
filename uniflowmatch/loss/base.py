from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from uniflowmatch.datasets import PATHWAY_REQUIREMENTS, TEXT_QUANTITIES
from uniflowmatch.models.base import (
    UFMClassificationRefinementOutput,
    UFMFlowFieldOutput,
    UFMMaskFieldOutput,
    UFMOutputInterface,
)


class SupervisionBase:
    def __init__(self, multiplier: float = 1) -> None:
        """
        Base class for all supervision/loss functions.
        """

        self.enabled = True
        self.name = "base"
        self.multiplier = multiplier

    def compute_loss(
        self, batch: Dict[str, Any], model_result: UFMOutputInterface
    ) -> Tuple[bool, torch.Tensor, Dict[str, Any]]:
        """
        Call method to check required information in the batch and model_result.

        Args:
            batch (Dict[str, Any]): The input batch dictionary.
            model_result (UFMOutputInterface): The model result.

        Returns:
            enabled (bool): Whether the required information is present.
            loss (torch.Tensor): The computed loss.
            info (Dict[str, Any]): The additional information.
        """

        # Check if the required information is present to compute the loss
        if self._check_required_result(batch, model_result):
            loss, log_dict = self._compute_loss(batch, model_result) # compute the loss and logs
            return True, loss, log_dict
        else:
            return False, torch.tensor(0.0), {}

    def _check_required_result(self, batch: Dict[str, Any], model_result: UFMOutputInterface) -> bool:
        try:
            self.supervision_fwd = self.get_supervision_signal(batch)
            self.result_fwd = self.get_model_result(batch, model_result)
            return True
        except Exception as e:
            print(e)
            return False

    def _compute_loss(self, batch: Dict[str, Any], model_result: Dict[str, Any]):
        loss, log_dict = self.compute_loss_with_gathered_info(self.result_fwd, self.supervision_fwd)
        del self.supervision_fwd
        del self.result_fwd

        return loss, log_dict

    def get_supervision_signal(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Method to get the supervision signal.

        Args:
        batch (Dict[str, Any]): The input batch dictionary.

        Returns:
        Dict[str, torch.Tensor]: The supervision signal.
        """

        raise NotImplementedError("Subclasses must implement this method")

    def get_model_result(
        self, batch: List[Dict[str, torch.Tensor]], model_result: UFMOutputInterface
    ) -> Dict[str, torch.Tensor]:
        """
        Method to get the result from the model.

        Args:
        batch (Dict[str, Any]): The input batch dictionary.
        model_result (UFMOutputInterface): The model result dictionary.

        Returns:
        Dict[str, torch.Tensor]: The model result.
        """

        raise NotImplementedError("Subclasses must implement this method")

    def compute_loss_with_gathered_info(self, result_dict: Dict[str, torch.Tensor], supervision_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
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

        raise NotImplementedError("Subclasses must implement this method")

    def __repr__(self):
        return f"{self.multiplier}x({self.name})"