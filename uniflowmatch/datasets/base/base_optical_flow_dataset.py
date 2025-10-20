"""
This class implements the same interface as the BaseStereoViewDataset, with the only difference being
that it skips the flow supervision computation step because all supervision are precomputed
"""

import random

import cv2
import numpy as np
import PIL
import PIL.Image as Image
import torch
import torchvision.transforms as tvf

from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, is_good_type, view_name
from uniflowmatch.datasets.utils.flow_manipulation import (
    CenterCropManipulation,
    CovisibleGuidedCropManipulation,
    FlowManipulationComposite,
    RandomEqualCropManipulation,
    ResizeShortAxisManipulation,
)


class BaseOpticalFlowDataset(BaseStereoViewDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grayscale = tvf.RandomGrayscale(p=0.05)
        self.gaussianblur = tvf.RandomApply([tvf.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05)

    def __len__(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def _get_raw_views(self, idx, rng):
        raise NotImplementedError("This method should be implemented by the subclass")

    def _get_item(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # BatchedRandomSampler Requirements
        resolution = self._resolutions[ar_idx]

        # Load the base information
        views = self._get_raw_views(idx, self._rng)
        assert len(views) == self.num_views

        assert (
            views[0]["img"].shape == views[1]["img"].shape
        ), "Unequal raw image shapes are not implemented for pure flow datasets"

        W, H = resolution
        if not self.aug_crop:
            resize_crop_manipulation = FlowManipulationComposite(
                ResizeShortAxisManipulation((H, W)), CenterCropManipulation((H, W))
            )
        else:
            expansion = np.random.uniform(1.0, 1.2)

            # we do not shift the image to focus on the motion carried in the optical flow datasets
            resize_crop_manipulation = FlowManipulationComposite(
                ResizeShortAxisManipulation((int(H * expansion), int(W * expansion))),
                RandomEqualCropManipulation((H, W)),
            )

        crop_result = resize_crop_manipulation(
            img0=views[0]["img"].unsqueeze(0),
            img1=views[1]["img"].unsqueeze(0),
            flow=views[0]["flow"].unsqueeze(0),
            valid=views[0]["non_occluded_mask"].unsqueeze(0),
            flow_bwd=views[1]["flow"].unsqueeze(0),
            valid_bwd=views[1]["non_occluded_mask"].unsqueeze(0),
            additional_masks_fwd=[
                views[0]["occlusion_supervision_mask"].unsqueeze(0),
                views[0]["fov_mask"].unsqueeze(0),
            ],
            additional_masks_bwd=[
                views[1]["occlusion_supervision_mask"].unsqueeze(0),
                views[1]["fov_mask"].unsqueeze(0),
            ],
        )

        views[0]["img"] = Image.fromarray(crop_result[0].squeeze(0).numpy())  # img0 of output
        views[1]["img"] = Image.fromarray(crop_result[1].squeeze(0).numpy())  # img1 of output
        views[0]["flow"] = crop_result[2].squeeze(0)  # flow of output
        views[1]["flow"] = crop_result[4].squeeze(0)  # flow_bwd of output
        views[0]["non_occluded_mask"] = crop_result[3].squeeze(0)  # additional_masks_fwd of output
        views[1]["non_occluded_mask"] = crop_result[5].squeeze(0)  # additional_masks_bwd of output

        views[0]["occlusion_supervision_mask"] = crop_result[6][0].squeeze(0)  # additional_masks_fwd of output
        views[1]["occlusion_supervision_mask"] = crop_result[7][0].squeeze(0)  # additional_masks_bwd of output

        views[0]["fov_mask"] = crop_result[6][1].squeeze(0)
        views[1]["fov_mask"] = crop_result[7][1].squeeze(0)

        # apply imagenorm and color augmentations
        for v, view in enumerate(views):
            # encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))

            if self.transform == "colorjitter":
                view["img"] = self.color_jitter(view["img"])
                view["img"] = self.grayscale(view["img"])
                view["img"] = self.gaussianblur(view["img"])

            # with a probability, augment the image with super low intensity
            # for augmentation
            if self._rng.random() < 0.01 and self.aug_low_light:
                # apply darkening and quantization
                view["img"] = self.apply_darkening_and_quantization(np.asarray(view["img"], np.float32))
                # print("lowlight_augmentation")

            # to tensor (pixel value still 0-255), and normalize
            view["img"] = torch.from_numpy((view["img"] - self.mean_255) / self.std_255).permute(2, 0, 1)
            view["data_norm_type"] = self.data_norm_type

        # apply swap augmentations
        if self.aug_swap:
            self._swap_view_aug(views)

        # apply monocular augmentations
        if self.aug_monocular:
            raise NotImplementedError("Monocular augmentation is not implemented yet")

        # apply rot90 augmentations
        if self.aug_rot90 is False:
            pass
        elif self.aug_rot90 == "same":
            # TODO: similarly here, we need to manipulate the flow carefully
            # rotate_90(views, k=self._rng.choice(4))
            raise NotImplementedError("Rot90 augmentation is not implemented yet")
        elif self.aug_rot90 == "diff":
            # rotate_90(views[:1], k=self._rng.choice(4))
            # rotate_90(views[1:], k=self._rng.choice(4))
            raise NotImplementedError("Rot90 augmentation is not implemented yet")
        else:
            raise ValueError(f"Bad value for {self.aug_rot90=}")

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            # transpose_to_landscape(view) # FIXME: not implemented, need to implement rot90 in consideration of flow
            height, width = view["true_shape"]
            assert width >= height, "Transpose to landscape is needed in this dataset"
            # this allows to check whether the RNG is is the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

        # add dummy pose and intrinsics for the flow dataset. we are going to set them to NaN for now.
        for view in views:
            view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            view["camera_intrinsics"] = np.full((3, 3), np.nan, dtype=np.float32)

        # Add data partition
        for view in views:
            view["data_partition"] = self.get_data_partition(idx)

        return views
