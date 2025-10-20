#!/usr/bin/env python3
# --------------------------------------------------------
# Base class for datasets
# Adopted from AnyMap
# Adopted from DUSt3R & MASt3R (Naver Corporation, CC BY-NC-SA 4.0 (non-commercial use only))
# --------------------------------------------------------
import copy
from typing import Dict, Self

import numpy as np
import PIL
import PIL.Image as Image
import torch
import torchvision.transforms as tvf
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

import uniflowmatch.datasets.utils.cropping as cropping
from uniflowmatch.datasets.base.easy_dataset import EasyDataset
from uniflowmatch.datasets.utils.cropping import (
    crop_to_homography,
    crop_to_homography_no_rotation,
    extract_correspondences_from_pts3d,
    gen_random_crops,
    in2d_rect,
)
from uniflowmatch.utils.geometry import depthmap_to_absolute_camera_coordinates, depthmap_to_camera_coordinates, geotrf


class BaseStereoViewDataset(EasyDataset):
    """Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        *,  # only keyword arguments
        split=None,
        resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
        transform=None,
        data_norm_type=None,
        aug_crop=False,
        aug_swap=False,
        aug_monocular=False,
        aug_portrait_or_landscape=False,  # automatic choice between landscape/portrait when possible
        aug_rot90=False,
        aug_low_light=False,
        n_tentative_crops=4,
        seed=None,
    ):
        """
        PyTorch Dataset for stereo view pairs.

        Args:
            split (str): 'train', 'val', 'test', 'trainval', 'all', etc.
            resolution (int or tuple or list of tuples): Resolution of the images
            transform (str): Transform to apply to the images. Options:
            - 'colorjitter': tvf.ColorJittter(0.5, 0.5, 0.5, 0.1) after ImgNorm
            - 'imgnorm': ImgNorm only
            data_norm_type (str): Image normalization type.
                                  For options, see UniCeption image normalization dict.
            aug_crop (bool or float or str): Augment crop. If float, it is the scale of the crop. Other option is 'auto'.
            aug_swap (bool): Augment swap. If True, swap the views with 50% probability.
            aug_monocular (bool): Augment monocular. If True, return the same view twice.
            aug_portrait_or_landscape (bool): Augment portrait or landscape. If True, randomly choose between portrait and landscape for square aspect ratio images.
            aug_rot90 (bool or str): Augment by a rotation which is a multiple of 90 deg. If 'same', rotate both views by the same amount. If 'diff', rotate each view by a different amount.
            n_tentative_crops (int): Number of tentative crops to generate for each view.
            seed (int): Seed for the random number generator.
        """
        self.num_views = 2
        self.split = split
        self._set_resolutions(resolution)

        if data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
            self.data_norm_type = data_norm_type
            self.std_255 = 255.0 * np.array(IMAGE_NORMALIZATION_DICT[data_norm_type].std)
            self.mean_255 = 255.0 * np.array(IMAGE_NORMALIZATION_DICT[data_norm_type].mean)
        else:
            raise ValueError(
                f"Unknown data_norm_type: {data_norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
            )

        self.transform = transform
        assert self.transform in ["colorjitter", "imgnorm"], f"Bad value for {self.transform}"

        if self.transform == "colorjitter":
            self.color_jitter = tvf.ColorJitter(0.5, 0.5, 0.5, 0.1)
            self.grayscale = tvf.RandomGrayscale(p=0.05)
            self.gaussianblur = tvf.RandomApply([tvf.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05)

        self.aug_crop = aug_crop
        if self.aug_crop == "False":
            self.aug_crop = False

        self.n_tentative_crops = n_tentative_crops
        self.aug_swap = aug_swap
        self.aug_monocular = aug_monocular
        self.aug_portrait_or_landscape = aug_portrait_or_landscape
        self.aug_rot90 = aug_rot90
        self.aug_low_light = aug_low_light

        self.seed = seed

        self.is_metric_scale = False  # by default a dataset is not metric scale, subclasses can overwrite this

    def partition(self, partition_ratio: Dict[int, float], seed: int = 42) -> Self:
        num_samples = len(self)
        classes = list(partition_ratio.keys())
        ratios = np.array([partition_ratio[c] for c in classes])

        # Normalize ratios in case they don't sum exactly to 1.0
        ratios = ratios / ratios.sum()

        # Determine number of samples per class
        counts = (ratios * num_samples).astype(int)
        counts[-1] += num_samples - counts.sum()  # Adjust for rounding

        # Build partition list
        data_partition = np.concatenate([np.full(count, c, dtype=int) for c, count in zip(classes, counts)])

        # Shuffle with fixed seed
        rng = np.random.default_rng(seed)
        rng.shuffle(data_partition)

        self.data_partition = data_partition
        return self

    def get_data_partition(self, idx: int) -> int:
        if hasattr(self, "data_partition") and self.data_partition is not None:
            return self.data_partition[idx]
        return 0

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(height, int), f"Bad type for {height=} {type(height)=}, should be int"
            assert width >= height
            self._resolutions.append((width, height))

    def _swap_view_aug(self, views):
        if self._rng.random() < 0.5:
            views.reverse()

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
            which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2) and self.aug_portrait_or_landscape:
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        offset_factor = 0.5
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=offset_factor)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        return image, depthmap, intrinsics2

    def generate_crops_from_pair(self, view1, view2, resolution, aug_crop_arg, n_crops=4, rng=np.random):
        views = [view1, view2]

        if aug_crop_arg is False:
            # compatibility
            for i in range(2):
                view = views[i]
                view["img"], view["depthmap"], view["camera_intrinsics"] = self._crop_resize_if_necessary(
                    view["img"], view["depthmap"], view["camera_intrinsics"], resolution, rng=rng
                )
                view["pts3d"], view["valid_mask"] = depthmap_to_absolute_camera_coordinates(
                    view["depthmap"], view["camera_intrinsics"], view["camera_pose"]
                )
            return

        # extract correspondences
        corres = extract_correspondences_from_pts3d(*views, target_n_corres=500, rng=rng)

        # generate 4 random crops in each view
        view_crops = []
        crops_resolution = []
        corres_msks = []
        for i in range(2):
            if aug_crop_arg == "auto" or aug_crop_arg == "auto_crop_asis":
                S = min(views[i]["img"].size)
                R = min(resolution)
                aug_crop = S * (S - R) // R
                aug_crop = max(0.1 * S, aug_crop)  # for cropping: augment scale of at least 10%, and more if possible
            else:
                aug_crop = aug_crop_arg

            # tranpose the target resolution if necessary
            assert resolution[0] >= resolution[1]
            W, H = imsize = views[i]["img"].size
            crop_resolution = resolution
            if H > 1.1 * W:
                # image is portrait mode
                crop_resolution = resolution[::-1]
            elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
                # image is square, so we chose (portrait, landscape) randomly
                if rng.integers(2):
                    crop_resolution = resolution[::-1]

            crops = gen_random_crops(imsize, n_crops, crop_resolution, aug_crop=aug_crop, rng=rng)
            view_crops.append(crops)
            crops_resolution.append(crop_resolution)

            # compute correspondences
            corres_msks.append(in2d_rect(corres[i], crops))

        # compute IoU for each
        intersection = np.float32(corres_msks[0]).T @ np.float32(corres_msks[1])
        # select best pair of crops
        best = np.unravel_index(intersection.argmax(), (n_crops, n_crops))
        crops = [view_crops[i][c] for i, c in enumerate(best)]

        # crop with the homography
        for i in range(2):
            view = views[i]

            if aug_crop_arg == "auto":
                imsize, K_new, R, H = crop_to_homography(view["camera_intrinsics"], crops[i], crops_resolution[i])
            elif aug_crop_arg == "auto_crop_asis":
                imsize, K_new, R, H = crop_to_homography_no_rotation(
                    view["camera_intrinsics"], crops[i], crops_resolution[i]
                )
            else:
                raise ValueError(f"Bad value for {aug_crop_arg=}")
            # imsize, K_new, H = upscale_homography(imsize, resolution, K_new, H)

            # update camera params
            K_old = view["camera_intrinsics"]
            view["camera_intrinsics"] = K_new
            view["camera_pose"] = view["camera_pose"].copy()
            view["camera_pose"][:3, :3] = view["camera_pose"][:3, :3] @ R

            # apply homography to image and depthmap
            homo8 = (H / H[2, 2]).ravel().tolist()[:8]
            view["img"] = view["img"].transform(
                imsize, Image.Transform.PERSPECTIVE, homo8, resample=Image.Resampling.BICUBIC
            )

            depthmap2 = depthmap_to_camera_coordinates(view["depthmap"], K_old)[0] @ R[:, 2]
            view["depthmap"] = np.array(
                Image.fromarray(depthmap2).transform(imsize, Image.Transform.PERSPECTIVE, homo8)
            )

            # recompute 3d points from scratch
            view["pts3d"], view["valid_mask"] = depthmap_to_absolute_camera_coordinates(
                view["depthmap"], view["camera_intrinsics"], view["camera_pose"]
            )

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

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        for v, view in enumerate(views):
            assert (
                "pts3d" not in view
            ), f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view["idx"] = (idx, ar_idx, v)
            view["is_metric_scale"] = self.is_metric_scale

            assert "camera_intrinsics" in view
            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view["camera_pose"]).all(), f"NaN in camera pose for view {view_name(view)}"
            assert "pts3d" not in view
            assert "valid_mask" not in view
            # assert np.isfinite(view["depthmap"]).all(), f"NaN in depthmap for view {view_name(view)}" # comment out for speed

            if self.aug_crop:
                # pts3d is recomputed from depthmap except in the aug_crop pathway. disabling to accelerate
                # dataloading for aug_crop=False
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
                view["pts3d"] = pts3d
                view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

        self.generate_crops_from_pair(
            views[0],
            views[1],
            resolution=resolution,
            aug_crop_arg=self.aug_crop,
            n_crops=self.n_tentative_crops,
            rng=self._rng,
        )
        for v, view in enumerate(views):
            # encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))

            # replace with my transform
            if self.transform == "colorjitter":
                view["img"] = self.color_jitter(view["img"])
                view["img"] = self.grayscale(view["img"])
                view["img"] = self.gaussianblur(view["img"])

            # with a probability, augment the image with super low intensity
            # for augmentation
            randomval = self._rng.random()
            if randomval < 0.01 and self.aug_low_light:
                # apply darkening and quantization
                view["img"] = tvf.RandomGrayscale(p=1.0)(view["img"])
                view["img"] = self.apply_darkening_and_quantization(np.asarray(view["img"], np.float32))
                # print("lowlight_augmentation")

            # print(randomval, self.split)

            # to tensor (pixel value still 0-255), and normalize
            view["img"] = torch.from_numpy((view["img"] - self.mean_255) / self.std_255).permute(2, 0, 1)
            view["data_norm_type"] = self.data_norm_type
            # Pixels for which depth is fundamentally undefined
            view["sky_mask"] = view["depthmap"] < 0

        if self.aug_swap:
            self._swap_view_aug(views)

        if self.aug_monocular:
            if self._rng.random() < self.aug_monocular:
                views = [copy.deepcopy(views[0]) for _ in range(len(views))]

        if self.aug_rot90 is False:
            pass
        elif self.aug_rot90 == "same":
            rotate_90(views, k=self._rng.choice(4))
        elif self.aug_rot90 == "diff":
            rotate_90(views[:1], k=self._rng.choice(4))
            rotate_90(views[1:], k=self._rng.choice(4))
        else:
            raise ValueError(f"Bad value for {self.aug_rot90=}")

        # check data-types metric_scale
        for v, view in enumerate(views):
            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view["camera_intrinsics"]

            # check shapes
            assert view["depthmap"].shape == view["img"].shape[1:]
            assert view["depthmap"].shape == view["pts3d"].shape[:2]
            assert view["depthmap"].shape == view["valid_mask"].shape

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

        for view in views:
            view["depthmap"] = torch.from_numpy(view["depthmap"])
            view["pts3d"] = torch.from_numpy(view["pts3d"])
            view["valid_mask"] = torch.from_numpy(view["valid_mask"])
            view["sky_mask"] = torch.from_numpy(view["sky_mask"])

        for view in views:
            view["data_partition"] = self.get_data_partition(idx)
        # camera pose and camera intrinsics are too small, they don't need to be converted to torch.Tensor

        return views

    def apply_darkening_and_quantization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply darkening, quantization, noise, and bias to a normalized [0, 1] RGB image of shape (3, H, W).

        Args:
            image (np.ndarray): Input image of shape (3, H, W), values in [0, 1].

        Returns:
            np.ndarray: Processed image of shape (3, H, W), values in [0, 1].
        """

        darkening_factor = np.random.uniform(0.1, 0.3)
        darkened_image = np.clip(image * darkening_factor, 0, 255)

        # Quantization: reduce the number of levels per channel
        quantization_levels = 16
        quant_step = 255.0 / quantization_levels
        quantized_image = np.floor(darkened_image / quant_step) * quant_step

        # Add noise (small Gaussian noise in [0, 1] range)
        noise = np.random.normal(loc=0.0, scale=0.02 * 255, size=quantized_image.shape)
        quantized_image += noise

        # Add random bias per pixel and per channel
        bias = np.random.uniform(0, 0.08 * 255, size=quantized_image.shape)
        quantized_image += bias

        # Clip final image to [0, 1] range
        final_image = np.clip(quantized_image, 0, 255)

        return final_image.astype(np.float32)


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view["true_shape"]

    if width < height:
        rotate_90([view], k=1)


def rotate_90(views, k=1):
    from scipy.spatial.transform import Rotation

    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = Rotation.from_euler("z", 90 * k, degrees=True).as_matrix()

    for view in views:
        view["img"] = torch.rot90(view["img"], k=k, dims=(-2, -1))  # WARNING!! dims=(-1,-2) != dims=(-2,-1)
        view["depthmap"] = np.rot90(view["depthmap"], k=k).copy()
        view["camera_pose"] = view["camera_pose"] @ RT

        RT2 = np.eye(3, dtype=np.float32)
        RT2[:2, :2] = RT[:2, :2] * ((1, -1), (-1, 1))
        H, W = view["depthmap"].shape
        if k % 4 == 0:
            pass
        elif k % 4 == 1:
            # top-left (0,0) pixel becomes (0,H-1)
            RT2[:2, 2] = (0, H - 1)
        elif k % 4 == 2:
            # top-left (0,0) pixel becomes (W-1,H-1)
            RT2[:2, 2] = (W - 1, H - 1)
        elif k % 4 == 3:
            # top-left (0,0) pixel becomes (W-1,0)
            RT2[:2, 2] = (W - 1, 0)
        else:
            raise ValueError(f"Bad value for {k=}")

        view["camera_intrinsics"][:2, 2] = geotrf(RT2, view["camera_intrinsics"][:2, 2])
        if k % 2 == 1:
            K = view["camera_intrinsics"]
            np.fill_diagonal(K, K.diagonal()[[1, 0, 2]])

        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
        view["pts3d"] = pts3d
        view["valid_mask"] = np.rot90(view["valid_mask"], k=k).copy()
        view["sky_mask"] = np.rot90(view["sky_mask"], k=k).copy()
        view["true_shape"] = np.int32((H, W))
