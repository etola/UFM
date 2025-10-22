from uniflowmatch.loss.base import SupervisionBase
from uniflowmatch.loss.cross_entropy import CrossEntropyLoss
from uniflowmatch.loss.epe import (
    FlowEPELoss,
    RobustRegressionLoss,
)

from uniflowmatch.loss.refinement_cross_entropy import RefinementCrossEntropyLoss
from uniflowmatch.loss.refinement_cross_entropy_efficient import RefinementCrossEntropyLossEfficient


SUPERVISION_CLASS = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "FlowEPELoss": FlowEPELoss,
    "RobustRegressionLoss": RobustRegressionLoss,
    "RefinementCrossEntropyLoss": RefinementCrossEntropyLoss,
    "RefinementCrossEntropyLossEfficient": RefinementCrossEntropyLossEfficient,
}


def get_loss(loss_class: str, **kwargs) -> SupervisionBase:
    """
    Get the loss object based on the loss class and kwargs.

    Args:
        loss_class (str): The loss class.
        **kwargs: The keyword arguments to pass to the loss object.

    Returns:
        SupervisionBase: The loss object.
    """

    if loss_class not in SUPERVISION_CLASS:
        raise ValueError(f"Loss name {loss_class} not found")

    return SUPERVISION_CLASS[loss_class](**kwargs)
