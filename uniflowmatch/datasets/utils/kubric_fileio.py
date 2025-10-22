# Copyright 2023 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is copied from https://github.com/basilevh/gcd/blob/ae6336906543fbb4e5d8e2f1208fdf4220f2580c/data-gen/kubric/kubric/file_io.py#L2
# modified to remove the dependency on tensorflow, gfile.

import json
import logging
from typing import Any, Dict, Optional, Tuple

import imageio
import numpy as np
import png

logger = logging.getLogger(__name__)


def read_tiff(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        img = imageio.imread(filename, format="tiff")

    if img.ndim == 2:
        img = img[:, :, None]
    return img


def read_png(filename: str, rescale_range=None) -> np.ndarray:
    png_reader = png.Reader(filename)
    width, height, pngdata, info = png_reader.read()
    del png_reader

    bitdepth = info["bitdepth"]
    if bitdepth == 8:
        dtype = np.uint8
    elif bitdepth == 16:
        dtype = np.uint16
    else:
        raise NotImplementedError(f"Unsupported bitdepth: {bitdepth}")

    plane_count = info["planes"]
    pngdata = np.vstack(list(map(dtype, pngdata)))
    if rescale_range is not None:
        minv, maxv = rescale_range
        pngdata = pngdata / 2**bitdepth * (maxv - minv) + minv

    return pngdata.reshape((height, width, plane_count))


# Copyright 2023 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long, unexpected-keyword-arg
"""TODO(klausg): description."""
import json

import numpy as np

DEFAULT_LAYERS = ("rgba", "segmentation", "forward_flow", "backward_flow", "depth", "normal", "object_coordinates")


def load_scene_directory(scene_dir, target_size: Optional[Tuple[int, int]], layers=DEFAULT_LAYERS):
    example_key = f"{scene_dir.name}"

    with open(str(scene_dir / "data_ranges.json"), "r") as fp:
        data_ranges = json.load(fp)

    with open(str(scene_dir / "metadata.json"), "r") as fp:
        metadata = json.load(fp)

    with open(str(scene_dir / "events.json"), "r") as fp:
        events = json.load(fp)

    num_frames = metadata["metadata"]["num_frames"]

    result = {
        "metadata": {
            "video_name": example_key,
            "width": target_size[1],
            "height": target_size[0],
            "num_frames": num_frames,
            "num_instances": metadata["metadata"]["num_instances"],
        },
        "instances": [format_instance_information(obj) for obj in metadata["instances"]],
        "camera": format_camera_information(metadata),
        "events": format_events_information(events),
    }

    resolution = metadata["metadata"]["resolution"]

    assert resolution[1] / target_size[0] == resolution[0] / target_size[1]
    scale = resolution[1] / target_size[0]
    assert scale == resolution[1] // target_size[0]

    paths = {key: [scene_dir / f"{key}_{f:05d}.png" for f in range(num_frames)] for key in layers if key != "depth"}

    if "depth" in layers:
        depth_paths = [scene_dir / f"depth_{f:05d}.tiff" for f in range(num_frames)]
        depth_frames = np.array(
            [subsample_nearest_neighbor(read_tiff(frame_path), target_size) for frame_path in depth_paths]
        )
        depth_min, depth_max = np.min(depth_frames), np.max(depth_frames)
        result["depth"] = convert_float_to_uint16(depth_frames, depth_min, depth_max)
        result["metadata"]["depth_range"] = [depth_min, depth_max]

    if "forward_flow" in layers:
        result["metadata"]["forward_flow_range"] = [
            data_ranges["forward_flow"]["min"] / scale,
            data_ranges["forward_flow"]["max"] / scale,
        ]
        result["forward_flow"] = [
            subsample_nearest_neighbor(read_png(frame_path)[..., :2], target_size)
            for frame_path in paths["forward_flow"]
        ]

    if "backward_flow" in layers:
        result["metadata"]["backward_flow_range"] = [
            data_ranges["backward_flow"]["min"] / scale,
            data_ranges["backward_flow"]["max"] / scale,
        ]
        result["backward_flow"] = [
            subsample_nearest_neighbor(read_png(frame_path)[..., :2], target_size)
            for frame_path in paths["backward_flow"]
        ]

    for key in ["normal", "object_coordinates", "uv"]:
        if key in layers:
            result[key] = [subsample_nearest_neighbor(read_png(frame_path), target_size) for frame_path in paths[key]]

    if "segmentation" in layers:
        # somehow we ended up calling this "segmentations" in TFDS and
        # "segmentation" in kubric. So we have to treat it separately.
        result["segmentations"] = [
            subsample_nearest_neighbor(read_png(frame_path), target_size) for frame_path in paths["segmentation"]
        ]

    if "rgba" in layers:
        result["video"] = [subsample_avg(read_png(frame_path), target_size)[..., :3] for frame_path in paths["rgba"]]

    return example_key, result, metadata


def format_camera_information(metadata):
    return {
        "focal_length": metadata["camera"]["focal_length"],
        "sensor_width": metadata["camera"]["sensor_width"],
        "field_of_view": metadata["camera"]["field_of_view"],
        "positions": np.array(metadata["camera"]["positions"], np.float32),
        "quaternions": np.array(metadata["camera"]["quaternions"], np.float32),
    }


def format_events_information(events):
    return {
        "collisions": [
            {
                "instances": np.array(c["instances"], dtype=np.uint16),
                "frame": c["frame"],
                "force": c["force"],
                "position": np.array(c["position"], dtype=np.float32),
                "image_position": np.array(c["image_position"], dtype=np.float32),
                "contact_normal": np.array(c["contact_normal"], dtype=np.float32),
            }
            for c in events["collisions"]
        ],
    }


def format_instance_information(obj):
    return {
        "mass": obj["mass"],
        "friction": obj["friction"],
        "restitution": obj["restitution"],
        "positions": np.array(obj["positions"], np.float32),
        "quaternions": np.array(obj["quaternions"], np.float32),
        "velocities": np.array(obj["velocities"], np.float32),
        "angular_velocities": np.array(obj["angular_velocities"], np.float32),
        "bboxes_3d": np.array(obj["bboxes_3d"], np.float32),
        "image_positions": np.array(obj["image_positions"], np.float32),
        "bboxes": [tfds.features.BBox(*bbox) for bbox in obj["bboxes"]],
        "bbox_frames": np.array(obj["bbox_frames"], dtype=np.uint16),
        "visibility": np.array(obj["visibility"], dtype=np.uint16),
    }


def convert_float_to_uint16(array, min_val, max_val):
    return np.round((array - min_val) / (max_val - min_val) * 65535).astype(np.uint16)


def is_complete_dir(video_dir, layers=DEFAULT_LAYERS):
    video_dir = video_dir
    filenames = [d.name for d in video_dir.iterdir()]
    if not ("data_ranges.json" in filenames and "metadata.json" in filenames and "events.json" in filenames):
        return False
    nr_frames_per_category = {key: len([fn for fn in filenames if fn.startswith(key)]) for key in layers}

    nr_expected_frames = nr_frames_per_category["rgba"]
    if nr_expected_frames == 0:
        return False
    if not all(nr_frames == nr_expected_frames for nr_frames in nr_frames_per_category.values()):
        return False

    return True