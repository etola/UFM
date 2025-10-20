#!/usr/bin/env python3
# --------------------------------------------------------
# Base class for datasets
# Adopted from AnyMap
# Adopted from DUSt3R & MASt3R (Naver Corporation, CC BY-NC-SA 4.0 (non-commercial use only))
# Modified to provide flow groundtruth supervision
# --------------------------------------------------------

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import default_collate
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

from uniflowmatch.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from uniflowmatch.datasets.utils.flow_manipulation import (
    CenterCropManipulation,
    CovisibleGuidedCropManipulation,
    FlowManipulationComposite,
    ResizeShortAxisManipulation,
)
from uniflowmatch.utils.geometry import (
    get_meshgrid_torch,
    project_points_to_pixels_batched,
    quaternion_to_rot_matrix,
    z_depthmap_to_norm_depthmap_batched,
)

# Here, we have different pathwats for different types of datasets.
# See paper Appendix Section A for more details.
PATHWAY_REQUIREMENTS = {
    "identity": {
        "img",
        "data_norm_type",
        "instance",
        "flow",
        "non_occluded_mask",
        "occlusion_supervision_mask",
        "fov_mask",
        "is_widebaseline",
        "is_synthetic",
        "camera_pose",
        "camera_intrinsics",
        "data_partition",
        "suitable_for_refinement",
    },
    "static_flow_postprocessing": {
        "img",
        "data_norm_type",
        "instance",
        "pts3d",
        "camera_pose",
        "camera_intrinsics",
        "depthmap",
        "is_widebaseline",
        "is_synthetic",
        "covisible_rendering_parameters",
        "data_partition",
        "suitable_for_refinement",
    },
    "kubric_flow_postprocessing": {
        "img",
        "norm_depthmap",
        "segmentation",
        "intrinsics",
        "extrinsics",
        "object_position",
        "object_orientation",
        "resolution",
        "data_norm_type",
        "instance",
        # augmentation settings
        "aug_crop",
        "is_widebaseline",
        "is_synthetic",
        "data_partition",
        "suitable_for_refinement",
    },
}

# These quantities should not be treated as a tensor
TEXT_QUANTITIES = {"data_norm_type", "instance", "is_widebaseline", "is_synthetic"}


@dataclass
class PathwayCollatedBatch:
    img_shape: Tuple[int, int]  # shape of images in each collection
    pathways: List[str]  # The collections each example belongs to
    pathway_batch: Dict[str, Any]  # The collated batch for each pathway


# Custom collate function to batch examples in a batch according to their pathway,
# enabling efficient transfer to GPU and postprocessing.
def collate_fn_with_delayed_flow_postprocessing(batch: List[Any]):
    """
    StereoViewFlowDataset provides flow supervision directly, while the BaseStereoViewDataset does not.
    When mixing batches, we need to apply different pathways for flow supervision comming from both
    datasets. This function will apply flow postprocessing to the views that do not have it.
    """

    batch_size = len(batch)  # get the batch size from the first view

    pathways = []
    for i in range(batch_size):
        # select a pathway for each example in the batch
        pathway = None
        for candidate_pathway, requirements in PATHWAY_REQUIREMENTS.items():
            if all(requirement in batch[i][0] for requirement in requirements):
                pathway = candidate_pathway
                break
        if pathway is None:
            print(batch[i][0].keys())
        assert pathway is not None, "No pathway found for the example"
        pathways.append(pathway)

    # collect the data for each pathway(only img, flow, non_occluded_mask, occlusion_supervision_mask)
    pathway_batch = {}
    for pathway in PATHWAY_REQUIREMENTS.keys():
        pathway_data = []
        for i in range(batch_size):
            if pathways[i] == pathway:
                pathway_data.append(
                    [{k: view[k] for k in (view.keys() & PATHWAY_REQUIREMENTS[pathway])} for view in batch[i]]
                )
        pathway_batch[pathway] = default_collate(pathway_data) if len(pathway_data) > 0 else None

    # determine image shape from img_shape if batch is not kubric, and resolution if batch is kubric
    example_pathway = set(pathways).pop()
    if example_pathway == "identity" or example_pathway == "static_flow_postprocessing":
        img_shape = pathway_batch[example_pathway][0]["img"][0].shape[-2:]
    elif example_pathway == "kubric_flow_postprocessing":
        W = pathway_batch[example_pathway][0]["resolution"][0][0].item()
        H = pathway_batch[example_pathway][0]["resolution"][1][0].item()

        img_shape = (H, W)

    # return the assignment and the batch
    return PathwayCollatedBatch(img_shape=img_shape, pathways=pathways, pathway_batch=pathway_batch)


# After the custom collated batch is being transferred to GPU, this function will apply flow postprocessing
# for different pathways, and merge the batch.
def apply_flow_postprocessing_and_merge_batch(
    custom_collated_batch: PathwayCollatedBatch,
    depth_error_threshold=0.5,
    depth_error_temperature=0.5,
    relative_depth_error_threshold=0.01,
    opt_iters=0,
):
    """
    After the custom collated batch is being transferred to GPU, this function will apply flow postprocessing
    to the views without it, and merge the batch.
    """

    assert isinstance(
        custom_collated_batch, PathwayCollatedBatch
    ), "Expected custom collated batch, please use collate_fn_with_delayed_flow_postprocessing in the dataloader"

    B = len(custom_collated_batch.pathways)
    H, W = custom_collated_batch.img_shape

    # determine the device
    device = None
    for pathway in set(custom_collated_batch.pathways):
        if len(custom_collated_batch.pathway_batch[pathway]) > 0:
            device = custom_collated_batch.pathway_batch[pathway][0]["img"].device
            break

    assert device is not None, "No device found for the batch"

    # pre-allocate the final batch output
    output_views = []
    for _ in range(2):
        view = {
            "img": torch.zeros(B, 3, H, W, device=device, dtype=torch.float32),
            "flow": torch.zeros(B, 2, H, W, device=device, dtype=torch.float32),
            "non_occluded_mask": torch.zeros(B, H, W, device=device, dtype=torch.bool),
            "occlusion_supervision_mask": torch.zeros(B, H, W, device=device, dtype=torch.bool),
            "fov_mask": torch.zeros(B, H, W, device=device, dtype=torch.bool),
            "camera_pose": torch.zeros(B, 4, 4, device=device, dtype=torch.float32),
            "camera_intrinsics": torch.zeros(B, 3, 3, device=device, dtype=torch.float32),
            "data_partition": torch.zeros(B, device=device, dtype=torch.int64),
            "suitable_for_refinement": torch.zeros(B, device=device, dtype=torch.bool),
        }
        output_views.append(view)

    # apply different pathways to the views
    for pathway in set(custom_collated_batch.pathways):
        current_pathway_mask = torch.tensor(
            [pathway == p for p in custom_collated_batch.pathways], dtype=torch.bool, device=device
        )

        if pathway == "identity":
            result_views = identity_pathway(custom_collated_batch.pathway_batch[pathway])
        elif pathway == "static_flow_postprocessing":
            result_views = static_flow_postprocessing_pathway(
                custom_collated_batch.pathway_batch[pathway],
                depth_error_threshold=depth_error_threshold,
                depth_error_temperature=depth_error_temperature,
                relative_depth_error_threshold=relative_depth_error_threshold,
                opt_iters=opt_iters,
            )
        elif pathway == "kubric_flow_postprocessing":
            result_views = kubric_flow_postprocessing_pathway(custom_collated_batch.pathway_batch[pathway])
        else:
            raise ValueError(f"Unknown pathway {pathway}")

        for i, view in enumerate(result_views):
            for key in PATHWAY_REQUIREMENTS["identity"] - TEXT_QUANTITIES:
                output_views[i][key][current_pathway_mask] = view[key]

    # special case for TEXT_QUANTITIES: data_norm_type, instance, is_widebaseline, is_synthetic
    for i, view in enumerate(output_views):
        for k in TEXT_QUANTITIES:
            text_quantity = [None] * B
            for pathway in set(custom_collated_batch.pathways):
                belongs_to_pathway = np.array([pathway == p for p in custom_collated_batch.pathways])
                indexes_in_batch = np.where(belongs_to_pathway)[0]

                for pathway_index, batch_index in enumerate(indexes_in_batch):
                    text_quantity[batch_index] = custom_collated_batch.pathway_batch[pathway][i][k][pathway_index]

            view[k] = text_quantity

    return output_views


# Different pathways for flow postprocessing
def identity_pathway(batch):
    # copy the data from allocated_views to the batch
    return batch


def static_flow_postprocessing_pathway(
    batch, depth_error_threshold=0.1, depth_error_temperature=0.1, relative_depth_error_threshold=0.005, opt_iters=2
):
    with torch.inference_mode(mode=False):  # else, autograd(for occlusion calculation) will not work
        flow_occlusion_post_processing(
            batch,
            depth_error_threshold=depth_error_threshold,
            depth_error_temperature=depth_error_temperature,
            relative_depth_error_threshold=relative_depth_error_threshold,
            opt_iters=opt_iters,
        )

    return batch


def kubric_flow_postprocessing_pathway(batch):
    # compute flow
    for view, other_view in zip(batch, reversed(batch)):
        # project into camera coordinates
        pts3d = unproject_depth(view["norm_depthmap"], view["intrinsics"].float(), normdepth=True)

        B, H, W = pts3d.shape[:3]

        # camera 0 to world at time 0 transform
        worldt0_T_camt0 = view["extrinsics"].float()

        # object to world at time 0 transform
        worldt0_T_obj = torch.zeros((B, 100, 4, 4), device=pts3d.device)
        worldt0_T_obj[:, :, :3, :3] = quaternion_to_rot_matrix(view["object_orientation"], scalar_first=True)
        worldt0_T_obj[:, :, :3, 3] = view["object_position"]
        worldt0_T_obj[:, :, 3, 3] = 1

        # object to world at time 1 transform
        worldt1_T_obj = torch.zeros((B, 100, 4, 4), device=pts3d.device)
        worldt1_T_obj[:, :, :3, :3] = quaternion_to_rot_matrix(other_view["object_orientation"], scalar_first=True)
        worldt1_T_obj[:, :, :3, 3] = other_view["object_position"]
        worldt1_T_obj[:, :, 3, 3] = 1

        # camera 1 to world at time 1 transform
        worldt1_T_camt1 = other_view["extrinsics"].float()

        # compose all the transforms
        camt0_T_obj = torch.linalg.inv(worldt0_T_camt0).unsqueeze(1) @ worldt0_T_obj
        camt1_T_obj = torch.linalg.inv(worldt1_T_camt1).unsqueeze(1) @ worldt1_T_obj
        cam1_T_camt0 = camt1_T_obj @ torch.linalg.inv(camt0_T_obj)

        # transform points from cam0 to cam1
        # gather the transform for each pixel
        # apply the transform to each pixel
        batch_object_idx = 100 * torch.arange(B, device=pts3d.device).view(B, 1, 1) + view["segmentation"]
        gathered_transform = cam1_T_camt0.view(-1, 4, 4)[batch_object_idx]

        pts3d_cam1 = (gathered_transform[..., :3, :3] @ pts3d.unsqueeze(-1)).squeeze(-1) + gathered_transform[
            ..., :3, 3
        ]

        # project the points to pixels according to the intrinsics of the other view
        uv, valid_mask = project_points_to_pixels_batched(pts3d_cam1, other_view["intrinsics"].float())

        view["fov_mask"] = valid_mask

        view["flow"] = uv - get_meshgrid_torch(W, H, device=uv.device)
        view["flow"][~valid_mask] = 0.0
        view["flow"] = view["flow"].permute(0, 3, 1, 2)

        # compute occlusion based on depth reprojection error thresholding
        # supply "depth_validity" and "expected_normdepth"
        view["depth_validity"] = view["norm_depthmap"] > 0
        view["expected_normdepth"] = torch.zeros_like(view["norm_depthmap"])
        view["expected_normdepth"][view["fov_mask"]] = torch.linalg.norm(pts3d_cam1[valid_mask], dim=-1)

    flow_occlusion_post_processing(
        batch,
        opt_iters=0,
        depth_error_threshold=0.1,
        depth_error_temperature=0.1,
        relative_depth_error_threshold=0.005,
    )

    # apply cropping and augmentations
    # assert target resolution, and cropping augmentation selection are uniform across the batch
    W = torch.unique(batch[0]["resolution"][0])
    assert len(W) == 1, "Expected uniform width resolution"

    H = torch.unique(batch[0]["resolution"][1])
    assert len(H) == 1, "Expected uniform height resolution"

    aug_crop = torch.unique(batch[0]["aug_crop"])
    assert len(aug_crop) == 1, "Expected uniform cropping augmentation"

    W = W.item()
    H = H.item()
    aug_crop = aug_crop.item()

    if not aug_crop:
        manipulations = FlowManipulationComposite(
            ResizeShortAxisManipulation(minimum_size=(H, W)),
            CenterCropManipulation((H, W)),
        )
    else:
        expansion = np.random.uniform(1.0, 1.2)

        manipulations = FlowManipulationComposite(
            ResizeShortAxisManipulation(minimum_size=(int(expansion * H), int(expansion * W))),
            CovisibleGuidedCropManipulation((H, W), N_crop=10),
        )

    # apply the manipulation to the views
    crop_result = manipulations(
        img0=(batch[0]["img"].permute(0, 2, 3, 1) * 255).to(torch.uint8),
        img1=(batch[1]["img"].permute(0, 2, 3, 1) * 255).to(torch.uint8),
        flow=batch[0]["flow"],
        valid=batch[0]["non_occluded_mask"],
        flow_bwd=batch[1]["flow"],
        valid_bwd=batch[1]["non_occluded_mask"],
        additional_masks_fwd=[
            batch[0]["occlusion_supervision_mask"],
            batch[0]["fov_mask"],
        ],
        additional_masks_bwd=[batch[1]["occlusion_supervision_mask"], batch[1]["fov_mask"]],
    )

    batch[0]["img"] = crop_result[0].permute(0, 3, 1, 2)  # img0 of output
    batch[1]["img"] = crop_result[1].permute(0, 3, 1, 2)  # img1 of output
    batch[0]["flow"] = crop_result[2]  # flow of output
    batch[1]["flow"] = crop_result[4]  # flow_bwd of output
    batch[0]["non_occluded_mask"] = crop_result[3]  # additional_masks_fwd of output
    batch[1]["non_occluded_mask"] = crop_result[5]  # additional_masks_bwd of output
    batch[0]["occlusion_supervision_mask"] = crop_result[6][0]  # additional_masks_fwd of output
    batch[1]["occlusion_supervision_mask"] = crop_result[7][0]  # additional_masks_bwd of output
    batch[0]["fov_mask"] = crop_result[6][1]
    batch[1]["fov_mask"] = crop_result[7][1]

    # normalize the image with the required data_norm_type
    for view in batch:
        data_norm_type = set(view["data_norm_type"])
        assert len(data_norm_type) == 1, "Expected uniform data_norm_type"
        data_norm_type = data_norm_type.pop()

        data_norm = IMAGE_NORMALIZATION_DICT[data_norm_type]

        mean_cvt = data_norm.mean.view(1, 3, 1, 1).to(view["img"].device) * 255.0
        std_cvt = data_norm.std.view(1, 3, 1, 1).to(view["img"].device) * 255.0

        view["img"] = (view["img"] - mean_cvt) / std_cvt

    # add dummy intrinsics and pose because they are not well-handled in the augmentation
    # - we should update the augmentation code before the mega-training.
    for view in batch:
        N = view["img"].shape[0]
        view["camera_pose"] = torch.full((N, 4, 4), fill_value=np.nan, device=view["img"].device)
        view["camera_intrinsics"] = torch.full((N, 3, 3), fill_value=np.nan, device=view["img"].device)

    return batch


@lru_cache(maxsize=5)
def get_meshgrid_xxyy(w, h, device):
    x = torch.arange(w, device=device)
    y = torch.arange(h, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    return xx, yy


def unproject_depth(depth, intrinsics, flow=None, normdepth=False):
    B, H, W = depth.shape
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]

    xx, yy = get_meshgrid_xxyy(W, H, device=depth.device)

    xx = xx.view(1, H, W) - cx.view(B, 1, 1)
    yy = yy.view(1, H, W) - cy.view(B, 1, 1)

    if flow is not None:
        xx = xx + flow[:, :, 0]
        yy = yy + flow[:, :, 1]

    if normdepth:
        # depth is the length of the point to camera center, not z coordinate
        x_norm = xx.view(B, H, W) / fx.view(B, 1, 1)
        y_norm = yy.view(B, H, W) / fy.view(B, 1, 1)

        depth = depth / torch.sqrt(x_norm**2 + y_norm**2 + 1)

    X = xx.view(B, H, W) / fx.view(B, 1, 1) * depth
    Y = yy.view(B, H, W) / fy.view(B, 1, 1) * depth
    Z = depth

    return torch.stack([X, Y, Z], axis=-1)


def compute_reprojection_error(uv, expected_depth, actual_depthmap, possibly_occluded_mask):
    """
    Compute reprojection error between the expected depth and the actual depthmap at the projected pixel location.

    Args:
    - uv (torch.Tensor): projected pixel locations
    - expected_depth (torch.Tensor): expected depth values for each uv
    - actual_depthmap (torch.Tensor): actual depthmap (B, H, W)
    - possibly_occluded_mask (torch.Tensor): mask of pixels that are possibly occluded

    Returns:
    - torch.Tensor: reprojection error
    """

    B, H, W = possibly_occluded_mask.shape
    valid_pixels1_opt = uv.permute(1, 0) + 0.5  # since the pixel center is represented as 0.0

    # convert to normalized coordinates to apply grid sample
    shape_normalizer = torch.tensor([W, H], device=valid_pixels1_opt.device, dtype=valid_pixels1_opt.dtype).view(2, 1)
    valid_pixels1_opt_normalized = valid_pixels1_opt / shape_normalizer * 2 - 1

    valid_pixels1_opt_normalized = torch.clip(valid_pixels1_opt_normalized, -1, 1)

    # pad the queries to get uniform length for all images in batch
    pixels_in_each_example = torch.sum(possibly_occluded_mask, dim=[1, 2])
    max_pixels = torch.max(pixels_in_each_example)
    sum_pixels = torch.cumsum(pixels_in_each_example, dim=0)

    padded_queries = torch.zeros(B, max_pixels, 2, device=valid_pixels1_opt.device, dtype=valid_pixels1_opt.dtype)
    valid_padded_mask = torch.arange(max_pixels, device=valid_pixels1_opt.device) < pixels_in_each_example[:, None]

    # fill the queries with the valid pixels
    padded_queries[valid_padded_mask] = valid_pixels1_opt_normalized.permute(1, 0)

    # apply grid sample
    sampled_depth = torch.nn.functional.grid_sample(
        actual_depthmap.view(B, 1, H, W),  # expand to BCHW
        grid=padded_queries.unsqueeze(1),
        # now grid have shape 1, 1, V, 2, in which V is the unrolled pixels
        # at which valid_mask is True. In other words, the results corresponds
        # to pixels True in valid_mask, unrolled row by row.
        mode="bilinear",  # This is a very important design choice. Normally
        # we would not use bilinear interpolation for depth map, because it will
        # create non-existent points when interpolating between motion boundaries.
        # but here we are only using it to validate, which means its effect will not
        # propagate to the regression values. Using bilinear solves aliasing
        # at highly inclined angles.
        padding_mode="zeros",
        align_corners=False,
    )[
        :, 0, 0, :
    ]  # output is BCHW, we only have the unrolled pixels in W dimension

    # select the non-padded values
    sampled_depth = sampled_depth[valid_padded_mask]

    return torch.abs(sampled_depth - expected_depth)


def query_projected_mask(uv, other_mask, uv_source_mask):
    """
    Compute reprojection error between the expected depth and the actual depthmap at the projected pixel location.

    Args:
    - uv (torch.Tensor): projected pixel locations
    - other_mask (torch.Tensor): boolean mask to query (B, H, W)
    - uv_source_mask (torch.Tensor): mask of pixels that corresponds to the uv pixels

    Returns:
    - torch.Tensor: reprojection error
    """

    B, H, W = other_mask.shape
    valid_pixels1_opt = uv.permute(1, 0) + 0.5  # since the pixel center is represented as 0.0

    # convert to normalized coordinates to apply grid sample
    shape_normalizer = torch.tensor([W, H], device=valid_pixels1_opt.device, dtype=valid_pixels1_opt.dtype).view(2, 1)
    valid_pixels1_opt_normalized = valid_pixels1_opt / shape_normalizer * 2 - 1

    valid_pixels1_opt_normalized = torch.clip(valid_pixels1_opt_normalized, -1, 1)

    # pad the queries to get uniform length for all images in batch
    pixels_in_each_example = torch.sum(uv_source_mask, dim=[1, 2])
    max_pixels = torch.max(pixels_in_each_example)
    sum_pixels = torch.cumsum(pixels_in_each_example, dim=0)

    padded_queries = torch.zeros(B, max_pixels, 2, device=valid_pixels1_opt.device, dtype=valid_pixels1_opt.dtype)
    valid_padded_mask = torch.arange(max_pixels, device=valid_pixels1_opt.device) < pixels_in_each_example[:, None]

    # fill the queries with the valid pixels
    padded_queries[valid_padded_mask] = valid_pixels1_opt_normalized.permute(1, 0)

    # apply grid sample
    sampled_mask = torch.nn.functional.grid_sample(
        other_mask.view(B, 1, H, W).float(),  # expand to BCHW
        grid=padded_queries.unsqueeze(1),
        # now grid have shape 1, 1, V, 2, in which V is the unrolled pixels
        # at which valid_mask is True. In other words, the results corresponds
        # to pixels True in valid_mask, unrolled row by row.
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[
        :, 0, 0, :
    ]  # output is BCHW, we only have the unrolled pixels in W dimension

    # select the non-padded values
    sampled_mask = sampled_mask[valid_padded_mask]

    output_mask = torch.zeros_like(uv_source_mask)
    output_mask[uv_source_mask] = sampled_mask.to(torch.bool)

    return output_mask


def flow_occlusion_post_processing(
    views, depth_error_threshold=0.1, depth_error_temperature=0.1, relative_depth_error_threshold=0.005, opt_iters=5
):
    """
    Generate flow supervision from pointmap, depthmap, intrinsics, and extrinsics.

    Args:
    - views (list[dict]): list of views, already batched by the dataloader
    """

    assert len(views) == 2, f"Expected 2 views, to compute flow to other view, got {len(views)} views"

    for view, other_view in zip(views, reversed(views)):
        if "flow" in view:
            assert "fov_mask" in view
            assert "depth_validity" in view
            assert "expected_normdepth" in view
            assert "norm_depthmap" in other_view

            B, H, W = view["fov_mask"].shape

            # print("Warning: flow already present in post processing, doing occlusion only")

            uv = get_meshgrid_torch(W, H, device=view["fov_mask"].device) + view["flow"].permute(0, 2, 3, 1)
            expected_norm_depth = view["expected_normdepth"][view["fov_mask"]]

            valid_mask = view["fov_mask"]
            norm_depth_in_other_view = other_view["norm_depthmap"]  # Cautious: this need to be normalized depth

        else:
            # project points from current view to other view
            # points are in a row-major format, so we need to transpose the last 2 dimensions
            B, H, W, C = view["pts3d"].shape

            world_to_other_camera = torch.linalg.inv(other_view["camera_pose"])

            current_points_in_other = (
                view["pts3d"].view(B, -1, C) @ world_to_other_camera[:, :3, :3].permute(0, 2, 1)
                + world_to_other_camera[:, :3, 3][:, None, :]
            )
            current_points_in_other = current_points_in_other.view(B, H, W, C)

            # project points to pixels
            uv, valid_mask = project_points_to_pixels_batched(current_points_in_other, other_view["camera_intrinsics"])

            # compute flow
            flow = uv - get_meshgrid_torch(W, H, device=uv.device)
            flow[~valid_mask, :] = 0.0

            # compute occlusion based on depth reprojection error thresholding
            expected_norm_depth = torch.linalg.norm(current_points_in_other[valid_mask], dim=-1)
            norm_depth_in_other_view = z_depthmap_to_norm_depthmap_batched(
                other_view["depthmap"], other_view["camera_intrinsics"]
            )

            view["flow"] = flow.permute(0, 3, 1, 2)

            # compute correspondence validity
            view["fov_mask"] = valid_mask  # FIXME: Here we should actually throw away points from invalid self depth!!!
            view["depth_validity"] = view["depthmap"] > 0

        if "covisible_rendering_parameters" in view:
            absolute_coeffs = view["covisible_rendering_parameters"][:, 0][:, None, None] * torch.ones_like(
                valid_mask, dtype=torch.float32
            )
            absolute_coeffs = absolute_coeffs[valid_mask]

            temperature_coeffs = view["covisible_rendering_parameters"][:, 1][:, None, None] * torch.ones_like(
                valid_mask, dtype=torch.float32
            )
            temperature_coeffs = temperature_coeffs[valid_mask]

            relative_coeffs = view["covisible_rendering_parameters"][:, 2][:, None, None] * torch.ones_like(
                valid_mask, dtype=torch.float32
            )
            relative_coeffs = relative_coeffs[valid_mask]

            error_threshold = absolute_coeffs + relative_coeffs * expected_norm_depth - np.log(0.5) * temperature_coeffs
        else:
            error_threshold = (
                depth_error_threshold
                + relative_depth_error_threshold * expected_norm_depth
                - np.log(0.5) * depth_error_temperature
            )

        # to determine occlusion, we will threshold the error between the distance of projected point to the other camera center
        # v.s. the norm-depth value recorded in the otherview's depthmap at the projected pixel location. If they met, then the point
        # is the rendered point in the other view, and is not occluded. Otherwise, it is occluded.
        uv_copy = uv.clone()
        valid_uv = uv[valid_mask]
        view["valid_uv"] = valid_uv
        if (
            opt_iters > 0
        ):  # if opt_iters is 0, we will not optimize the uv_residual, and there are no need to create the optimizer and the residual tensor
            uv_residual = torch.zeros_like(
                valid_uv, requires_grad=True
            )  # we optimize uv_residual to estimate the lower bound of the depth error
            opt = torch.optim.Adam([uv_residual], lr=1e-1, weight_decay=1e-1)
            valid_uv = valid_uv + uv_residual
            opt.zero_grad()

        # select the possibly occluded pixels to check for non-occlusion
        possibly_occluded_mask = valid_mask.clone()
        possible_occlusion_in_valid_pixels = torch.ones(
            size=(valid_mask.sum(),), dtype=torch.bool, device=valid_mask.device
        )
        checked_uv = valid_uv  # [possible_occlusion_in_valid_pixels]
        checked_expected_norm_depth = expected_norm_depth  # [possible_occlusion_in_valid_pixels]
        checked_threshold = error_threshold  # [possible_occlusion_in_valid_pixels]

        opt_iteration = 0
        while True:
            # compute the reprojection error of the selected pixels and check if they are non-occluded
            reprojection_error = compute_reprojection_error(
                checked_uv, checked_expected_norm_depth, norm_depth_in_other_view, possibly_occluded_mask
            )

            occluded_selected_uv = reprojection_error >= checked_threshold

            # update the occlusion mask, uv_combined, and expected_norm_depth with the non_occluded_selected_uv
            possibly_occluded_mask_new = possibly_occluded_mask.clone()
            possibly_occluded_mask_new[possibly_occluded_mask] = occluded_selected_uv

            possible_occlusion_in_valid_pixels_new = possible_occlusion_in_valid_pixels.clone()
            possible_occlusion_in_valid_pixels_new[possible_occlusion_in_valid_pixels] = occluded_selected_uv

            possibly_occluded_mask = possibly_occluded_mask_new
            possible_occlusion_in_valid_pixels = possible_occlusion_in_valid_pixels_new

            if opt_iters == 0 or opt_iteration >= opt_iters:
                break

            # optimize the uv_residual
            loss = torch.sum(reprojection_error)
            loss.backward()
            opt.step()
            with torch.no_grad():
                uv_residual.clamp_(-0.707, 0.707)
            opt.zero_grad()

            opt_iteration += 1

            checked_uv = valid_uv[possible_occlusion_in_valid_pixels]
            checked_expected_norm_depth = expected_norm_depth[possible_occlusion_in_valid_pixels]
            checked_threshold = error_threshold[possible_occlusion_in_valid_pixels]

        # the non-occlsion mask is the invert of the possibly occluded mask
        non_occluded_mask = ~possibly_occluded_mask

        view["non_occluded_mask"] = non_occluded_mask & valid_mask

    # finally, account for depth invalidity in the other view
    for view, other_view in zip(views, reversed(views)):
        other_view_depth_validity = query_projected_mask(
            view["valid_uv"].detach(), other_view["depth_validity"], view["fov_mask"]
        )
        view["other_view_depth_validity"] = other_view_depth_validity

        # occlusion should be supervised at
        # 1. self depth is valid, once projected will land out of bound in the other view
        # OR
        # 2. self depth is valid, once projected will land in the bound of other view, landing position shows valid depth

        view["occlusion_supervision_mask"] = (view["depth_validity"] & (~view["fov_mask"])) | (
            view["fov_mask"] & other_view_depth_validity
        )

    # condition fov_mask to be valid only if the depth is valid
    for view in views:
        view["fov_mask"] = view["fov_mask"] & view["depth_validity"]
        view["non_occluded_mask"] = view["non_occluded_mask"] & view["depth_validity"]
