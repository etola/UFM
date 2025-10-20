from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from cv2 import add


class FlowManipulationBase:
    def __init__(self):
        pass

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        raise NotImplementedError

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        flow_bwd: Optional[torch.Tensor],
        valid_bwd: Optional[torch.Tensor],
        additional_masks_fwd: List[torch.Tensor] = [],
        additional_masks_bwd: List[torch.Tensor] = [],
        **kwargs,
    ):
        """
        Apply the flow manipulation to the input correspondence pairs.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - flow: Tensor of shape (B, 2, H, W), dtype float32 representing the flow.
            - valid: Tensor of shape (B, H, W), dtype bool representing the valid mask.
            - flow_bwd: Tensor of shape (B, 2, H, W), dtype float32 representing the backward flow.
            - valid_bwd: Tensor of shape (B, H, W), dtype bool representing the backward valid mask.
            - additional_masks_fwd: List of additional forward masks to manipulate.
            - additional_masks_bwd: List of additional backward masks to manipulate.
        Returns:
            The input arguments, manipulated, in the same shapes and dtypes.
        """
        raise NotImplementedError

    def check_input(self, H: int, W: int) -> bool:
        """
        Check whether the input shapes are correct for the current manipulation.

        Args:
            - H: Height of the input images.
            - W: Width of the input images.

        Returns:
            The input arguments, manipulated, in the same shapes and dtypes.
        """
        raise NotImplementedError


class FlowManipulationComposite(FlowManipulationBase):
    def __init__(self, *manipulations: List[FlowManipulationBase]):
        self.manipulations = manipulations

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        output_shape = (H, W)
        for manipulation in self.manipulations:
            output_shape = manipulation.output_shape(*output_shape)

        return output_shape

    def check_input(self, H: int, W: int) -> bool:
        current_shape = (H, W)
        for manipulation in self.manipulations:
            if not manipulation.check_input(*current_shape):
                return False

            current_shape = manipulation.output_shape(*current_shape)

        return True

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        flow_bwd: Optional[torch.Tensor],
        valid_bwd: Optional[torch.Tensor],
        additional_masks_fwd: List[torch.Tensor] = [],
        additional_masks_bwd: List[torch.Tensor] = [],
        **kwargs,
    ):
        for manipulation in self.manipulations:
            img0, img1, flow, valid, flow_bwd, valid_bwd, additional_masks_fwd, additional_masks_bwd = manipulation(
                img0, img1, flow, valid, flow_bwd, valid_bwd, additional_masks_fwd, additional_masks_bwd, **kwargs
            )

        return img0, img1, flow, valid, flow_bwd, valid_bwd, additional_masks_fwd, additional_masks_bwd


class PadManipulation(FlowManipulationBase):
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        return self.target_size

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        flow_bwd: Optional[torch.Tensor],
        valid_bwd: Optional[torch.Tensor],
        additional_masks_fwd: List[torch.Tensor] = [],
        additional_masks_bwd: List[torch.Tensor] = [],
        **kwargs,
    ):
        """
        Apply the flow manipulation to the input correspondence pairs.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - flow: Tensor of shape (B, 2, H, W), dtype float32 representing the flow.
            - valid: Tensor of shape (B, H, W), dtype bool representing the valid mask.
            - flow_bwd: Tensor of shape (B, 2, H, W), dtype float32 representing the backward flow.
            - valid_bwd: Tensor of shape (B, H, W), dtype bool representing the backward valid mask.
            - additional_masks_fwd: List of additional forward masks to manipulate.
            - additional_masks_bwd: List of additional backward masks to manipulate.
        Returns:
            The input arguments, manipulated, in the same shapes and dtypes.
        """

        assert img0.shape == img1.shape, "Image shapes must match"

        _, h, w, _ = img0.shape
        target_h, target_w = self.target_size

        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = (pad_left, pad_right, pad_top, pad_bottom)

        img0_padded = F.pad(img0.permute(0, 3, 1, 2), padding).permute(0, 2, 3, 1)
        img1_padded = F.pad(img1.permute(0, 3, 1, 2), padding).permute(0, 2, 3, 1)
        flow_padded = F.pad(flow, (pad_left, pad_right, pad_top, pad_bottom))
        valid_padded = F.pad(valid.unsqueeze(1), padding).squeeze(1)
        flow_bwd_padded = F.pad(flow_bwd, (pad_left, pad_right, pad_top, pad_bottom)) if flow_bwd is not None else None
        valid_bwd_padded = F.pad(valid_bwd.unsqueeze(1), padding).squeeze(1) if valid_bwd is not None else None

        additional_padded_fwd = []
        for mask in additional_masks_fwd:
            additional_padded_fwd.append(F.pad(mask.unsqueeze(1), padding).squeeze(1))

        additional_padded_bwd = []
        for mask in additional_masks_bwd:
            additional_padded_bwd.append(F.pad(mask.unsqueeze(1), padding).squeeze(1))

        # Mark the padded regions as invalid
        if pad_top > 0:
            valid_padded[:, :pad_top, :] = 0
        if pad_bottom > 0:
            valid_padded[:, -pad_bottom:, :] = 0
        if pad_left > 0:
            valid_padded[:, :, :pad_left] = 0
        if pad_right > 0:
            valid_padded[:, :, -pad_right:] = 0

        # also for the backward flow
        if valid_bwd_padded is not None:
            if pad_top > 0:
                valid_bwd_padded[:, :pad_top, :] = 0
            if pad_bottom > 0:
                valid_bwd_padded[:, -pad_bottom:, :] = 0
            if pad_left > 0:
                valid_bwd_padded[:, :, :pad_left] = 0
            if pad_right > 0:
                valid_bwd_padded[:, :, -pad_right:] = 0

        return (
            img0_padded,
            img1_padded,
            flow_padded,
            valid_padded,
            flow_bwd_padded,
            valid_bwd_padded,
            additional_padded_fwd,
            additional_padded_bwd,
        )

    def check_input(self, H: int, W: int) -> bool:
        """
        Check whether the input shapes are correct for the current manipulation.

        Args:
            - H: Height of the input images.
            - W: Width of the input images.

        Returns:
            The input arguments, manipulated, in the same shapes and dtypes.
        """

        return W <= self.target_size[1] and H <= self.target_size[0]


class CenterCropManipulation(FlowManipulationBase):
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        return self.target_size

    def check_input(self, H: int, W: int) -> bool:
        return H >= self.target_size[0] and W >= self.target_size[1]

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        flow_bwd: Optional[torch.Tensor],
        valid_bwd: Optional[torch.Tensor],
        additional_masks_fwd: List[torch.Tensor] = [],
        additional_masks_bwd: List[torch.Tensor] = [],
        **kwargs,
    ):
        """
        Apply the flow manipulation to the input correspondence pairs.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - flow: Tensor of shape (B, 2, H, W), dtype float32 representing the flow.
            - valid: Tensor of shape (B, H, W), dtype bool representing the valid mask.
            - flow_bwd: Tensor of shape (B, 2, H, W), dtype float32 representing the backward flow.
            - valid_bwd: Tensor of shape (B, H, W), dtype bool representing the backward valid mask.
            - additional_masks_fwd: List of additional forward masks to manipulate.
            - additional_masks_bwd: List of additional backward masks to manipulate.
        Returns:
            The input arguments, manipulated, in the same shapes and dtypes.
        """

        assert img0.shape == img1.shape, "Image shapes must match"

        _, h, w, _ = img0.shape
        target_h, target_w = self.target_size

        crop_top = (h - target_h) // 2
        crop_bottom = h - target_h - crop_top
        crop_left = (w - target_w) // 2
        crop_right = w - target_w - crop_left

        crop = (crop_left, crop_right, crop_top, crop_bottom)

        img0_cropped = img0[:, crop_top : h - crop_bottom, crop_left : w - crop_right, :]
        img1_cropped = img1[:, crop_top : h - crop_bottom, crop_left : w - crop_right, :]
        flow_cropped = flow[
            :, :, crop_top : h - crop_bottom, crop_left : w - crop_right
        ]  # equal crop start in both images result in no change in flow values
        valid_cropped = valid[:, crop_top : h - crop_bottom, crop_left : w - crop_right]

        flow_bwd_cropped = (
            flow_bwd[:, :, crop_top : h - crop_bottom, crop_left : w - crop_right] if flow_bwd is not None else None
        )
        valid_bwd_cropped = (
            valid_bwd[:, crop_top : h - crop_bottom, crop_left : w - crop_right] if valid_bwd is not None else None
        )

        # in the crop operation, we may render some correspondence invalid because we may crop away its destination.
        # therefore, we need to recalculate the fov mask and update the valid mask accordingly.

        fov_mask_fwd = self.compute_fov_mask_batched(img0_cropped, img1_cropped, valid_cropped, flow_cropped)
        valid_cropped_fwd = valid_cropped & fov_mask_fwd

        fov_mask_bwd = (
            self.compute_fov_mask_batched(img1_cropped, img0_cropped, valid_bwd_cropped, flow_bwd_cropped)
            if flow_bwd_cropped is not None
            else None
        )
        valid_bwd_cropped = valid_bwd_cropped & fov_mask_bwd if valid_bwd_cropped is not None else None

        additional_cropped = []
        for mask in additional_masks_fwd:
            additional_cropped.append(mask[:, crop_top : h - crop_bottom, crop_left : w - crop_right] & valid_cropped)

        additional_bwd_cropped = []
        for mask in additional_masks_bwd:
            additional_bwd_cropped.append(
                mask[:, crop_top : h - crop_bottom, crop_left : w - crop_right] & valid_bwd_cropped
            )

        return (
            img0_cropped,
            img1_cropped,
            flow_cropped,
            valid_cropped_fwd,
            flow_bwd_cropped,
            valid_bwd_cropped,
            additional_cropped,
            additional_bwd_cropped,
        )

    def compute_fov_mask_batched(self, base_image, other_image, valid, flow):
        assert base_image.shape == other_image.shape

        fov_mask = valid.clone()
        B, H, W, C = base_image.shape
        flow_base_coords = self.get_meshgrid(H, W, base_image.device)

        flow_base_coords = torch.stack((flow_base_coords[..., -1], flow_base_coords[..., 0]), dim=-1)  # H,W -> X, Y
        flow_target = flow_base_coords.permute(2, 0, 1).unsqueeze(0) + flow

        fov_mask[
            (flow_target[:, 0] < 0) | (flow_target[:, 1] < 0) | (flow_target[:, 0] >= W) | (flow_target[:, 1] >= H)
        ] = False

        return fov_mask

    @lru_cache(maxsize=5)
    def get_meshgrid(self, H, W, device):
        return torch.stack(
            torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"),
            dim=-1,
        ).float()


class ResizeHorizontalAxisManipulation(FlowManipulationBase):
    def __init__(self, horizontal_axis: int):
        self.horizontal_axis = horizontal_axis

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        resize_ratio = self.horizontal_axis / W

        return (int(H * resize_ratio), self.horizontal_axis)

    def check_input(self, H: int, W: int) -> bool:
        return True

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        flow_bwd: Optional[torch.Tensor],
        valid_bwd: Optional[torch.Tensor],
        additional_masks_fwd: List[torch.Tensor] = [],
        additional_masks_bwd: List[torch.Tensor] = [],
        **kwargs,
    ):
        """
        Apply the flow manipulation to the input correspondence pairs.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - flow: Tensor of shape (B, 2, H, W), dtype float32 representing the flow.
            - valid: Tensor of shape (B, H, W), dtype bool representing the valid mask.
            - flow_bwd: Tensor of shape (B, 2, H, W), dtype float32 representing the backward flow.
            - valid_bwd: Tensor of shape (B, H, W), dtype bool representing the backward valid mask.
            - additional_masks_fwd: List of additional forward masks to manipulate.
            - additional_masks_bwd: List of additional backward masks to manipulate.
        Returns:
            The input arguments, manipulated, in the same shapes and dtypes.
        """

        assert img0.shape == img1.shape, "Image shapes must match"

        _, h, w, _ = img0.shape
        target_h, target_w = self.output_shape(h, w)

        img0_resized = (
            F.interpolate(
                img0.permute(0, 3, 1, 2).float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
        )
        img1_resized = (
            F.interpolate(
                img1.permute(0, 3, 1, 2).float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
        )

        # Resize the flow and adjust its values
        scale_factor = (target_h / h, target_w / w)
        flow_resized_fwd = F.interpolate(flow, size=(target_h, target_w), mode="nearest")
        flow_resized_fwd[:, 0, :, :] *= scale_factor[1]
        flow_resized_fwd[:, 1, :, :] *= scale_factor[0]

        valid_resized_fwd = (
            F.interpolate(valid.unsqueeze(1).half(), size=(target_h, target_w), mode="nearest").squeeze(1).bool()
        )

        additional_resized_fwd = []
        for mask in additional_masks_fwd:
            additional_resized_fwd.append(
                F.interpolate(mask.unsqueeze(1).half(), size=(target_h, target_w), mode="nearest").squeeze(1).bool()
            )

        # Resize the backward flow and adjust its values
        flow_resized_bwd = None
        valid_resized_bwd = None

        if flow_bwd is not None:
            flow_resized_bwd = F.interpolate(flow_bwd, size=(target_h, target_w), mode="nearest")
            flow_resized_bwd[:, 0, :, :] *= scale_factor[1]
            flow_resized_bwd[:, 1, :, :] *= scale_factor[0]

            valid_resized_bwd = (
                F.interpolate(valid_bwd.unsqueeze(1).half(), size=(target_h, target_w), mode="nearest")
                .squeeze(1)
                .bool()
            )

        additional_resized_bwd = []
        for mask in additional_masks_bwd:
            additional_resized_bwd.append(
                F.interpolate(mask.unsqueeze(1).half(), size=(target_h, target_w), mode="nearest").squeeze(1).bool()
            )

        return (
            img0_resized,
            img1_resized,
            flow_resized_fwd,
            valid_resized_fwd,
            flow_resized_bwd,
            valid_resized_bwd,
            additional_resized_fwd,
            additional_resized_bwd,
        )


class ResizeVerticalAxisManipulation(ResizeHorizontalAxisManipulation):
    def __init__(self, vertical_axis: int):
        self.vertical_axis = vertical_axis

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        resize_ratio = self.vertical_axis / H

        return (self.vertical_axis, int(W * resize_ratio))

    def check_input(self, H: int, W: int) -> bool:
        return True


class ResizeShortAxisManipulation(ResizeHorizontalAxisManipulation):
    def __init__(self, minimum_size: Tuple[int, int]):
        self.min_H, self.min_W = minimum_size

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        resize_ratio = max(self.min_H / H + 1e-8, self.min_W / W + 1e-8)

        return (int(H * resize_ratio), int(W * resize_ratio))

    def check_input(self, H: int, W: int) -> bool:
        return True


class CovisibleGuidedCropManipulation(FlowManipulationBase):
    def __init__(self, target_size: Tuple[int, int], N_crop=10):
        self.target_size = target_size
        self.N_crop = N_crop

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        return self.target_size

    def check_input(self, H: int, W: int) -> bool:
        return H >= self.target_size[0] and W >= self.target_size[1]

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        flow_bwd: Optional[torch.Tensor],
        valid_bwd: Optional[torch.Tensor],
        additional_masks_fwd: List[torch.Tensor] = [],
        additional_masks_bwd: List[torch.Tensor] = [],
        num_covisible_sample=2048,
        **kwargs,
    ):
        """
        Apply the flow manipulation to the input correspondence pairs.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - flow: Tensor of shape (B, 2, H, W), dtype float32 representing the flow.
            - valid: Tensor of shape (B, H, W), dtype bool representing the valid mask.
            - flow_bwd: Tensor of shape (B, 2, H, W), dtype float32 representing the backward flow.
            - valid_bwd: Tensor of shape (B, H, W), dtype bool representing the backward valid mask.
            - additional_masks_fwd: List of additional forward masks to manipulate.
            - additional_masks_bwd: List of additional backward masks to manipulate.
        Returns:
            The input arguments, manipulated, in the same shapes and dtypes.
        """
        assert img0.shape == img1.shape, "Image shapes must match"

        B, h, w, _ = img0.shape
        target_h, target_w = self.target_size

        covisible_pairs = []
        batch_ids = []
        for self_valid, self_flow in zip([valid, valid_bwd], [flow, flow_bwd]):

            covisible_sum = torch.sum(self_valid, dim=(1, 2))
            covisible_th = num_covisible_sample / (covisible_sum + 1e-8)

            random_mask = torch.rand_like(self_valid, dtype=torch.float32) < covisible_th.view(-1, 1, 1)
            self_valid = self_valid & random_mask

            batch_id, covisible_i, covisible_j = torch.where(self_valid)

            N_covisible = len(covisible_i)
            covisible_pair = torch.zeros((N_covisible, 5), dtype=torch.int64, device=img0.device)
            covisible_pair[:, 0] = batch_id
            covisible_pair[:, 1] = covisible_j  # x coordinate
            covisible_pair[:, 2] = covisible_i  # y coordinate
            covisible_pair[:, 3] = covisible_j + self_flow[batch_id, 0, covisible_i, covisible_j].round().to(
                torch.int64
            )  # x coordinate
            covisible_pair[:, 4] = covisible_i + self_flow[batch_id, 1, covisible_i, covisible_j].round().to(
                torch.int64
            )  # y coordinate

            covisible_pairs.append(covisible_pair)
            batch_ids.append(batch_id)

        # suggest N_crop candidate crops for each element in the batch, subject to shape constraint between the source image and target size
        crop_i_range = max(0, h - target_h)
        crop_j_range = max(0, w - target_w)
        N_crop = self.N_crop

        crop_view0_xy = torch.zeros((B, N_crop, 2), dtype=torch.int64, device=img0.device)
        crop_view0_xy[:, :, 0] = (
            torch.randint(0, crop_j_range, (B, N_crop), device=img0.device) if crop_j_range > 0 else 0
        )
        crop_view0_xy[:, :, 1] = (
            torch.randint(0, crop_i_range, (B, N_crop), device=img0.device) if crop_i_range > 0 else 0
        )

        crop_view1_xy = torch.zeros((B, N_crop, 2), dtype=torch.int64, device=img0.device)
        crop_view1_xy[:, :, 0] = (
            torch.randint(0, crop_j_range, (B, N_crop), device=img0.device) if crop_j_range > 0 else 0
        )
        crop_view1_xy[:, :, 1] = (
            torch.randint(0, crop_i_range, (B, N_crop), device=img0.device) if crop_i_range > 0 else 0
        )

        target_wh = torch.tensor([target_w, target_h], dtype=torch.int64, device=img0.device)
        # evaluate the quality of pairwise crops by iou
        # evaluate forward covisibility, get a tensor of shape (B, N_crop, N_crop)

        sum_visibilities = []
        for batch_id, self_covisible, self_crop_xy, other_crop_xy in zip(
            batch_ids, covisible_pairs, [crop_view0_xy, crop_view1_xy], [crop_view1_xy, crop_view0_xy]
        ):
            # 1. check if a covisible pair is in the self crop
            source_vis = (self_crop_xy[batch_id] <= self_covisible[:, 1:3].unsqueeze(1)).all(dim=-1) & (
                self_crop_xy[batch_id] + target_wh > self_covisible[:, 1:3].unsqueeze(1)
            ).all(dim=-1)
            # 2. check if a covisible pair target is in the other crop
            target_vis = (other_crop_xy[batch_id] <= self_covisible[:, 3:].unsqueeze(1)).all(dim=-1) & (
                other_crop_xy[batch_id] + target_wh > self_covisible[:, 3:].unsqueeze(1)
            ).all(dim=-1)

            visibility = source_vis.view(-1, N_crop, 1) & target_vis.view(-1, 1, N_crop)

            sum_visibility = torch.zeros((B, N_crop, N_crop), dtype=torch.int32, device=img0.device)
            sum_visibility.scatter_add_(
                dim=0, index=batch_id.view(-1, 1, 1).repeat(1, N_crop, N_crop), src=visibility.int()
            )
            sum_visibility = sum_visibility.float() / (target_h * target_w)

            sum_visibilities.append(sum_visibility)
        del visibility

        # compute iou
        iou = sum_visibilities[0] * sum_visibilities[1] / (sum_visibilities[0] + sum_visibilities[1] + 1e-8) * 2

        iou_flatten = iou.view(-1, N_crop * N_crop)
        best_crops = torch.argmax(iou_flatten, dim=-1)

        best_crops_i = best_crops // N_crop
        best_crops_j = best_crops % N_crop

        # select the crops
        img0_crop_start = crop_view0_xy[torch.arange(B), best_crops_i]
        img1_crop_start = crop_view1_xy[torch.arange(B), best_crops_j]

        # build the index tensor to get the cropped images - forward pass
        ii, jj = torch.meshgrid(
            torch.arange(target_h, device=img0.device), torch.arange(target_w, device=img0.device), indexing="ij"
        )
        batched_ii = ii.unsqueeze(0).repeat(B, 1, 1) + img0_crop_start[:, 1].view(-1, 1, 1)  # ii + y start
        batched_jj = jj.unsqueeze(0).repeat(B, 1, 1) + img0_crop_start[:, 0].view(-1, 1, 1)  # jj + x start

        img0_cropped = img0[torch.arange(B).view(-1, 1, 1), batched_ii, batched_jj]
        flow_cropped = flow[torch.arange(B).view(-1, 1, 1), :, batched_ii, batched_jj] + (
            img0_crop_start - img1_crop_start
        ).view(B, 1, 1, 2)
        flow_cropped = flow_cropped.permute(0, 3, 1, 2)

        # process all the forward masks
        valid_cropped_fwd = valid[torch.arange(B).view(-1, 1, 1), batched_ii, batched_jj]

        additional_cropped = []
        for mask in additional_masks_fwd:
            additional_cropped.append(mask[torch.arange(B, device=img0.device).view(-1, 1, 1), batched_ii, batched_jj])

        # backward pass
        batched_ii = ii.unsqueeze(0).repeat(B, 1, 1) + img1_crop_start[:, 1].view(-1, 1, 1)  # ii + y start
        batched_jj = jj.unsqueeze(0).repeat(B, 1, 1) + img1_crop_start[:, 0].view(-1, 1, 1)  # jj + x start

        img1_cropped = img1[torch.arange(B).view(-1, 1, 1), batched_ii, batched_jj]
        flow_bwd_cropped = (
            flow_bwd[torch.arange(B, device=img0.device).view(-1, 1, 1), :, batched_ii, batched_jj]
            + (img1_crop_start - img0_crop_start).view(B, 1, 1, 2)
            if flow_bwd is not None
            else None
        )
        flow_bwd_cropped = flow_bwd_cropped.permute(0, 3, 1, 2) if flow_bwd_cropped is not None else None

        # # process all the valid masks
        valid_cropped_bwd = (
            valid_bwd[torch.arange(B, device=img0.device).view(-1, 1, 1), batched_ii, batched_jj]
            if valid_bwd is not None
            else None
        )

        fov_mask_fwd_new = self.compute_fov_mask_batched(
            img0_cropped, img1_cropped, torch.ones_like(valid_cropped_fwd), flow_cropped
        )
        valid_cropped_fwd = valid_cropped_fwd & fov_mask_fwd_new

        fov_mask_bwd_new = (
            self.compute_fov_mask_batched(
                img1_cropped, img0_cropped, torch.ones_like(valid_cropped_fwd), flow_bwd_cropped
            )
            if flow_bwd_cropped is not None
            else None
        )
        valid_cropped_bwd = valid_cropped_bwd & fov_mask_bwd_new if valid_cropped_bwd is not None else None

        # compute additional masks
        for mask in additional_cropped:
            mask &= fov_mask_fwd_new

        additional_cropped_bwd = []
        for mask in additional_masks_bwd:
            additional_cropped_bwd.append(
                mask[torch.arange(B, device=img0.device).view(-1, 1, 1), batched_ii, batched_jj] & fov_mask_bwd_new
            )

        return (
            img0_cropped,
            img1_cropped,
            flow_cropped,
            valid_cropped_fwd,
            flow_bwd_cropped,
            valid_cropped_bwd,
            additional_cropped,
            additional_cropped_bwd,
        )

    def compute_fov_mask_batched(self, base_image, other_image, valid, flow):
        assert base_image.shape == other_image.shape

        fov_mask = valid.clone()
        B, H, W, C = base_image.shape
        flow_base_coords = self.get_meshgrid(H, W, base_image.device)

        flow_base_coords = torch.stack((flow_base_coords[..., -1], flow_base_coords[..., 0]), dim=-1)  # H,W -> X, Y
        flow_target = flow_base_coords.permute(2, 0, 1).unsqueeze(0) + flow

        fov_mask[
            (flow_target[:, 0] < 0) | (flow_target[:, 1] < 0) | (flow_target[:, 0] >= W) | (flow_target[:, 1] >= H)
        ] = False

        return fov_mask

    @lru_cache(maxsize=5)
    def get_meshgrid(self, H, W, device):
        return torch.stack(
            torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"),
            dim=-1,
        ).float()


class RandomEqualCropManipulation(FlowManipulationBase):
    def __init__(self, target_size: Tuple[int, int], N_crop=10):
        self.target_size = target_size
        self.N_crop = N_crop

    def output_shape(self, H: int, W: int) -> Tuple[int, int]:
        """
        Compute the output shape of the image after the resize operation.
        """

        return self.target_size

    def check_input(self, H: int, W: int) -> bool:
        return H >= self.target_size[0] and W >= self.target_size[1]

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        flow: torch.Tensor,
        valid: torch.Tensor,
        flow_bwd: Optional[torch.Tensor],
        valid_bwd: Optional[torch.Tensor],
        additional_masks_fwd: List[torch.Tensor] = [],
        additional_masks_bwd: List[torch.Tensor] = [],
        num_covisible_sample=2048,
        **kwargs,
    ):
        """
        Apply the flow manipulation to the input correspondence pairs.

        Args:
            - img0: Tensor of shape (B, H, W, C), dtype uint8 representing the first set of images.
            - img1: Tensor of shape (B, H, W, C), dtype uint8 representing the second set of images.
            - flow: Tensor of shape (B, 2, H, W), dtype float32 representing the flow.
            - valid: Tensor of shape (B, H, W), dtype bool representing the valid mask.
            - flow_bwd: Tensor of shape (B, 2, H, W), dtype float32 representing the backward flow.
            - valid_bwd: Tensor of shape (B, H, W), dtype bool representing the backward valid mask.
            - additional_masks_fwd: List of additional forward masks to manipulate.
            - additional_masks_bwd: List of additional backward masks to manipulate.
        Returns:
            The input arguments, manipulated, in the same shapes and dtypes.
        """
        assert img0.shape == img1.shape, "Image shapes must match"

        B, h, w, _ = img0.shape
        target_h, target_w = self.target_size

        # suggest N_crop candidate crops for each element in the batch, subject to shape constraint between the source image and target size
        crop_i_range = max(0, h - target_h)
        crop_j_range = max(0, w - target_w)

        crop_i = (
            torch.randint(0, crop_i_range, (B,), device=img0.device)
            if crop_i_range > 0
            else torch.zeros(B, device=img0.device, dtype=torch.int64)
        )
        crop_j = (
            torch.randint(0, crop_j_range, (B,), device=img0.device)
            if crop_j_range > 0
            else torch.zeros(B, device=img0.device, dtype=torch.int64)
        )

        # build the index tensor to get the cropped images - forward pass
        ii, jj = torch.meshgrid(
            torch.arange(target_h, device=img0.device), torch.arange(target_w, device=img0.device), indexing="ij"
        )
        batched_ii = ii.unsqueeze(0).repeat(B, 1, 1) + crop_i.view(-1, 1, 1)  # ii + y start
        batched_jj = jj.unsqueeze(0).repeat(B, 1, 1) + crop_j.view(-1, 1, 1)  # jj + x start

        img0_cropped = img0[torch.arange(B).view(-1, 1, 1), batched_ii, batched_jj]
        flow_cropped = flow[torch.arange(B).view(-1, 1, 1), :, batched_ii, batched_jj]
        flow_cropped = flow_cropped.permute(0, 3, 1, 2)

        # process all the forward masks
        valid_cropped_fwd = valid[torch.arange(B).view(-1, 1, 1), batched_ii, batched_jj]

        additional_cropped = []
        for mask in additional_masks_fwd:
            additional_cropped.append(mask[torch.arange(B, device=img0.device).view(-1, 1, 1), batched_ii, batched_jj])

        # backward pass
        batched_ii = ii.unsqueeze(0).repeat(B, 1, 1) + crop_i.view(-1, 1, 1)  # ii + y start
        batched_jj = jj.unsqueeze(0).repeat(B, 1, 1) + crop_j.view(-1, 1, 1)  # jj + x start

        img1_cropped = img1[torch.arange(B).view(-1, 1, 1), batched_ii, batched_jj]
        flow_bwd_cropped = (
            flow_bwd[torch.arange(B, device=img0.device).view(-1, 1, 1), :, batched_ii, batched_jj]
            if flow_bwd is not None
            else None
        )
        flow_bwd_cropped = flow_bwd_cropped.permute(0, 3, 1, 2) if flow_bwd_cropped is not None else None

        # # process all the valid masks
        valid_cropped_bwd = (
            valid_bwd[torch.arange(B, device=img0.device).view(-1, 1, 1), batched_ii, batched_jj]
            if valid_bwd is not None
            else None
        )

        fov_mask_fwd_new = self.compute_fov_mask_batched(
            img0_cropped, img1_cropped, torch.ones_like(valid_cropped_fwd), flow_cropped
        )
        valid_cropped_fwd = valid_cropped_fwd & fov_mask_fwd_new

        fov_mask_bwd_new = (
            self.compute_fov_mask_batched(
                img1_cropped, img0_cropped, torch.ones_like(valid_cropped_fwd), flow_bwd_cropped
            )
            if flow_bwd_cropped is not None
            else None
        )
        valid_cropped_bwd = valid_cropped_bwd & fov_mask_bwd_new if valid_cropped_bwd is not None else None

        # compute additional masks
        for mask in additional_cropped:
            mask &= fov_mask_fwd_new

        additional_cropped_bwd = []
        for mask in additional_masks_bwd:
            additional_cropped_bwd.append(
                mask[torch.arange(B, device=img0.device).view(-1, 1, 1), batched_ii, batched_jj] & fov_mask_bwd_new
            )

        return (
            img0_cropped,
            img1_cropped,
            flow_cropped,
            valid_cropped_fwd,
            flow_bwd_cropped,
            valid_cropped_bwd,
            additional_cropped,
            additional_cropped_bwd,
        )

    def compute_fov_mask_batched(self, base_image, other_image, valid, flow):
        assert base_image.shape == other_image.shape

        fov_mask = valid.clone()
        B, H, W, C = base_image.shape
        flow_base_coords = self.get_meshgrid(H, W, base_image.device)

        flow_base_coords = torch.stack((flow_base_coords[..., -1], flow_base_coords[..., 0]), dim=-1)  # H,W -> X, Y
        flow_target = flow_base_coords.permute(2, 0, 1).unsqueeze(0) + flow

        fov_mask[
            (flow_target[:, 0] < 0) | (flow_target[:, 1] < 0) | (flow_target[:, 0] >= W) | (flow_target[:, 1] >= H)
        ] = False

        return fov_mask

    @lru_cache(maxsize=5)
    def get_meshgrid(self, H, W, device):
        return torch.stack(
            torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"),
            dim=-1,
        ).float()


if __name__ == "__main__":
    pass
