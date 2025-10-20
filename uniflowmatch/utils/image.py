#!/usr/bin/env python3
# --------------------------------------------------------
# Utilitary functions for loading & converting images
# Adopted from DUSt3R (Naver Corporation, CC BY-NC-SA 4.0 (non-commercial use only))
# Adopted from Mapanything
# --------------------------------------------------------
import os

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

try:
    from pillow_heif import register_heif_opener  # noqa

    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT


def img_to_arr(img):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, norm_type, true_shape=None):
    """
    Convert to normalized image tensor to RGB image for visualization
    Args:
        ftensor: image tensor or list of image tensors
        norm_type: normalization type, see UniCeption IMAGE_NORMALIZATION_DICT keys
        true_shape: if provided, the image will be cropped to this shape
    """
    if isinstance(ftensor, list):
        return [rgb(x, norm_type, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        if norm_type in IMAGE_NORMALIZATION_DICT.keys():
            img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
            mean = img_norm.mean.numpy()
            std = img_norm.std.numpy()
        else:
            raise ValueError(
                f"Unknown image normalization type: {norm_type}. Available types: {IMAGE_NORMALIZATION_DICT.keys()}"
            )
        img = ftensor * std + mean
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, norm_type, patch_size, square_ok=False, verbose=True, bayer_format=False):
    """
    Open and convert all images in a list or folder to proper input format for model
    Args:
        folder_or_list (str or list): Path to folder or list of image paths.
        size (int): Resize long side to this size.
        norm_type (str): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys.
        patch_size (int, optional): Patch size for image processing.
        square_ok (bool, optional): If True, allow square images. Defaults to False.
        verbose (bool, optional): If True, print progress messages. Defaults to True.
        bayer_format (bool, optional): If True, read images in Bayer format. Defaults to False.
    Returns:
        list: List of dictionaries containing image data and metadata
    """
    if isinstance(folder_or_list, str):
        # If folder_or_list is a string, assume it's a path to a folder
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        # If folder_or_list is a list, assume it's a list of image paths
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list

    else:
        # If folder_or_list is neither a string nor a list, raise an error
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    # Define supported image extensions
    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    # Initialize empty list to store image data
    imgs = []
    for path in folder_content:
        # Check if the file has a supported image extension
        if not path.lower().endswith(supported_images_extensions):
            continue
        if bayer_format:
            # If bayer_format is True, read the image in Bayer format
            color_bayer = cv2.imread(os.path.join(root, path), cv2.IMREAD_UNCHANGED)
            color = cv2.cvtColor(color_bayer, cv2.COLOR_BAYER_RG2BGR)
            img = PIL.Image.fromarray(color)
            img = exif_transpose(img).convert("RGB")
        else:
            # Otherwise, read the image normally
            img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
        W1, H1 = img.size
        if size == 224:
            # If size is 224, resize the short side to 224 and then crop
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # Otherwise, resize the long side to size
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // patch_size) * (patch_size / 2), ((2 * cy) // patch_size) * (patch_size / 2)
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
        if norm_type in IMAGE_NORMALIZATION_DICT.keys():
            img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
            ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)])
        else:
            raise ValueError(
                f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
            )
        imgs.append(
            dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=len(imgs), instance=str(len(imgs)))
        )

    assert imgs, "no images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs
