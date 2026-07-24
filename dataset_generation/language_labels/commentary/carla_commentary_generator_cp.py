"""Commentary generator adapted to the four-view CP dataset layout.

The original commentary rules remain in ``carla_commentary_generator.py``.
This module only replaces dataset discovery, template selection, and scenario
lookup so the original generator can run on ``database/four_view_single``
without modifying the upstream implementation. The original front-view fields
are preserved, while the three additional image paths and four per-view
visibility flags are appended with CP view suffixes.
"""

import glob
import gzip
import json
import os
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import ujson

from dataset_generation.language_labels.commentary import carla_commentary_generator as base_commentary


REPO_ROOT = Path(__file__).resolve().parents[3]
_ACTIVE_CAMERA_VIEWS = ()


def _build_camera_views(image_size, front_fov):
    """Camera rig matching ``team_code/config_cp.py``."""

    width, height = image_size
    return (
        {
            "name": "front",
            "directory": "rgb",
            "position": np.array([0.80, 0.0, 1.60], dtype=float),
            "yaw": 0.0,
            "width": width,
            "height": height,
            "fov": front_fov,
        },
        {
            "name": "left_front",
            "directory": "rgb_left_front",
            "position": np.array([0.27, -0.55, 1.60], dtype=float),
            "yaw": -55.0,
            "width": width,
            "height": height,
            "fov": 70.0,
        },
        {
            "name": "right_front",
            "directory": "rgb_right_front",
            "position": np.array([0.27, 0.55, 1.60], dtype=float),
            "yaw": 55.0,
            "width": width,
            "height": height,
            "fov": 70.0,
        },
        {
            "name": "rear",
            "directory": "rgb_rear",
            "position": np.array([-2.0, 0.0, 1.60], dtype=float),
            "yaw": 180.0,
            "width": width,
            "height": height,
            "fov": 110.0,
        },
    )


def _object_corners_ego(object_box):
    if object_box is None or "position" not in object_box:
        return np.empty((0, 3), dtype=float)

    position = np.asarray(object_box["position"], dtype=float)
    extent = np.asarray(object_box.get("extent", [0.15, 0.15, 0.15]), dtype=float)
    yaw = float(object_box.get("yaw", 0.0))
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    corners = signs * extent
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
    )
    return corners @ rotation.T + position


def _is_object_visible_in_view(object_box, view, min_y=0, max_y=None):
    corners = _object_corners_ego(object_box)
    if len(corners) == 0:
        return False

    relative = corners - view["position"]
    yaw = np.deg2rad(view["yaw"])
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    forward = cos_yaw * relative[:, 0] + sin_yaw * relative[:, 1]
    right = -sin_yaw * relative[:, 0] + cos_yaw * relative[:, 1]
    up = relative[:, 2]
    valid = forward > 0.1
    if not np.any(valid):
        return False

    forward = forward[valid]
    right = right[valid]
    up = up[valid]
    focal = view["width"] / (2.0 * np.tan(np.deg2rad(view["fov"] / 2.0)))
    projected_x = focal * right / forward + view["width"] / 2.0
    projected_y = focal * -up / forward + view["height"] / 2.0
    roi_max_y = view["height"] if max_y is None else min(max_y, view["height"])

    return bool(
        projected_x.max() > 0
        and projected_x.min() < view["width"]
        and projected_y.max() > min_y
        and projected_y.min() < roi_max_y
    )


def visibility_by_view(object_box, camera_views=None, min_y=0, max_y=None):
    views = _ACTIVE_CAMERA_VIEWS if camera_views is None else camera_views
    return {
        view["name"]: _is_object_visible_in_view(
            object_box, view, min_y=min_y, max_y=max_y
        )
        for view in views
    }


def is_vehicle_visible_in_any_cp_view(
    vehicle_obj, min_x, max_x, min_y, max_y, camera_matrix
):
    """Drop-in replacement used by the original Commentary rule code."""

    del min_x, max_x, camera_matrix
    return any(visibility_by_view(vehicle_obj, min_y=min_y, max_y=max_y).values())


def get_scenario_name_cp(measurement_file_current):
    """Resolve the actual scenario type from the recorded route XML."""

    measurement_path = Path(measurement_file_current)
    boxes_path = measurement_path.parent.parent / "boxes" / measurement_path.name

    try:
        with gzip.open(measurement_path, "rt", encoding="utf-8") as file:
            measurement = json.load(file)
        with gzip.open(boxes_path, "rt", encoding="utf-8") as file:
            boxes = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    route_group = None
    for box in boxes:
        if box.get("class") == "ego_info":
            route_group = box.get("scenario")
            break

    route_match = re.search(
        r"Rep\d+_(\d+)_route_?(\d+)", measurement_path.parent.parent.name
    )
    if not route_group or route_match is None:
        return route_group

    special_scenarios = {
        "custom_parkinglane": "turn_in_parkinglane",
        "pedestrians": "pedestrians",
        "OpensDoor": "VehicleOpensDoor",
    }
    if route_group in special_scenarios:
        return special_scenarios[route_group]

    route_file, route_number = route_match.groups()
    candidates = sorted(
        (REPO_ROOT / "data/simlingo").glob(
            f"**/{route_group}/{route_file}.xml"
        )
    )
    if not candidates:
        return route_group

    try:
        root = ET.parse(candidates[0]).getroot()
        route = root.find(f'./route[@id="{route_number}"]')
        scenario_nodes = list(route.find("scenarios").iter("scenario"))
        position = np.asarray(measurement["pos_global"], dtype=float)
        yaw = float(measurement["theta"])
    except (AttributeError, KeyError, TypeError, ValueError, ET.ParseError):
        return route_group

    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    scenarios = []
    local_positions = []
    for scenario in scenario_nodes:
        trigger = scenario.find("trigger_point")
        if trigger is None:
            continue
        trigger_position = np.array(
            [float(trigger.attrib["x"]), float(trigger.attrib["y"])]
        )
        local_positions.append(rotation.T @ (trigger_position - position))
        scenarios.append(scenario.attrib["type"])

    if not scenarios:
        return route_group

    distances = [
        np.linalg.norm(local_position)
        if local_position[0] < -10
        else 999999
        for local_position in local_positions
    ]
    return scenarios[int(np.argmin(distances))]


# ``create_commentary`` resolves this function in the original module's global
# namespace. Replacing it here keeps the original method intact while avoiding
# its old data/simlingo path parsing.
base_commentary.get_scenario_name = get_scenario_name_cp
base_commentary.is_vehicle_visible_in_image = is_vehicle_visible_in_any_cp_view


class COMsGenerator(base_commentary.COMsGenerator):
    """Original Commentary generator with four-view CP path configuration."""

    def __init__(self, args):
        self.TARGET_IMAGE_SIZE = args.target_image_size
        self.ORIGINAL_IMAGE_SIZE = args.original_image_size
        self.ORIGINAL_FOV = args.original_fov

        self.MIN_X = args.min_x
        self.MAX_X = args.max_x
        self.MIN_Y = args.min_y
        self.MAX_Y = args.max_y

        self.SKIP_FIRST_FRAMES = 10
        self.HISTORY_LEN = 5
        self.FUTURE_LEN = 10

        self.random_subset_count = args.random_subset_count
        self.sample_frame_mode = args.sample_frame_mode
        self.sample_uniform_interval = args.sample_uniform_interval

        self.save_examples = args.save_examples
        self.visualize_projection = args.visualize_projection
        self.filter_routes_by_result = args.filter_routes_by_result

        self.data_directory = args.data_directory
        self.path_keyframes = args.path_keyframes
        self.output_directory = args.output_directory
        self.output_examples_directory = args.output_examples_directory
        self.skip_existing = args.skip_existing

        self.CAMERA_MATRIX = base_commentary.build_projection_matrix(
            self.ORIGINAL_IMAGE_SIZE[0],
            self.ORIGINAL_IMAGE_SIZE[1],
            self.ORIGINAL_FOV,
        )
        self.camera_views = _build_camera_views(
            self.ORIGINAL_IMAGE_SIZE, self.ORIGINAL_FOV
        )
        global _ACTIVE_CAMERA_VIEWS
        _ACTIVE_CAMERA_VIEWS = self.camera_views

        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        if self.save_examples:
            Path(self.output_examples_directory).mkdir(parents=True, exist_ok=True)

        data_root = Path(self.data_directory)
        if (data_root / "data").is_dir():
            data_root = data_root / "data"

        self.data_boxes_paths_all = sorted(
            str(path) for path in data_root.glob("**/boxes/*.json.gz")
        )
        print(f"Number of boxes paths: {len(self.data_boxes_paths_all)}")

        self.data_boxes_paths = list(self.data_boxes_paths_all)
        if self.random_subset_count > 0:
            random.Random(42).shuffle(self.data_boxes_paths)
            self.data_boxes_paths = self.data_boxes_paths[:self.random_subset_count]
        self.data_boxes_paths.sort()

        # The released repository does not contain commentary.json. The
        # subsentence file contains the canonical templates expected by the
        # original rule generator.
        template_file = REPO_ROOT / "data/augmented_templates/commentary_subsentence.json"
        with template_file.open("r", encoding="utf-8") as file:
            self.templates = ujson.load(file)

        self.list_next_junction_id_minus_one = []
        self.all_labels = []
        self.stats = {
            "total_frames": 0,
            "frames_per_scenario": {},
            "num_visible_objects": 0,
            "num_not_visible_objects": 0,
            "num_visible_objects_left_front": 0,
            "num_visible_objects_right_front": 0,
            "num_visible_objects_rear": 0,
            "num_visible_objects_any_view": 0,
        }
        self.all_templates = []

        if self.sample_frame_mode == "keyframes":
            with open(self.path_keyframes, "r", encoding="utf-8") as file:
                self.keyframes_list = [line.strip() for line in file]
            self.keyframes_list = [
                path.replace("rgb", "boxes").replace(".jpg", ".json.gz")
                for path in self.keyframes_list
            ]

    def create_commentary(self, path_id):
        """Run the original generator and append four-view CP fields."""

        global _ACTIVE_CAMERA_VIEWS
        _ACTIVE_CAMERA_VIEWS = self.camera_views
        result = super().create_commentary(path_id)
        self._append_four_view_fields(self.data_boxes_paths[path_id])
        return result

    def _append_four_view_fields(self, boxes_path):
        if "/data/" not in boxes_path:
            return

        relative_output = boxes_path.split("/data/", 1)[1].replace(
            "/boxes/", "/commentary/"
        )
        output_path = Path(self.output_directory) / relative_output
        if not output_path.is_file():
            return

        try:
            with gzip.open(output_path, "rt", encoding="utf-8") as file:
                sample = json.load(file)
        except (OSError, json.JSONDecodeError):
            return

        image_paths = {
            "front": boxes_path.replace("/boxes/", "/rgb/").replace(
                ".json.gz", ".jpg"
            ),
            "left_front": boxes_path.replace(
                "/boxes/", "/rgb_left_front/"
            ).replace(".json.gz", ".jpg"),
            "right_front": boxes_path.replace(
                "/boxes/", "/rgb_right_front/"
            ).replace(".json.gz", ".jpg"),
            "rear": boxes_path.replace("/boxes/", "/rgb_rear/").replace(
                ".json.gz", ".jpg"
            ),
        }
        visibility = visibility_by_view(
            sample.get("cause_object"),
            camera_views=self.camera_views,
            min_y=self.MIN_Y,
            max_y=self.MAX_Y,
        )

        sample["image"] = image_paths["front"]
        sample["image_left_front"] = image_paths["left_front"]
        sample["image_right_front"] = image_paths["right_front"]
        sample["image_rear"] = image_paths["rear"]

        sample["cause_object_visible_in_image"] = visibility["front"]
        sample["cause_object_visible_in_image_left_front"] = visibility[
            "left_front"
        ]
        sample["cause_object_visible_in_image_right_front"] = visibility[
            "right_front"
        ]
        sample["cause_object_visible_in_image_rear"] = visibility["rear"]

        with gzip.open(output_path, "wt", encoding="utf-8") as file:
            json.dump(sample, file, ensure_ascii=False, indent=4)

    def save_stats_from_outputs(self):
        """Rebuild statistics after single- or multi-process generation."""

        stats = {
            "total_frames": 0,
            "frames_per_scenario": {},
            "num_visible_objects": 0,
            "num_not_visible_objects": 0,
            "num_visible_objects_left_front": 0,
            "num_visible_objects_right_front": 0,
            "num_visible_objects_rear": 0,
            "num_visible_objects_any_view": 0,
        }
        templates = set()

        pattern = os.path.join(self.output_directory, "**", "*.json.gz")
        for output_path in glob.glob(pattern, recursive=True):
            try:
                with gzip.open(output_path, "rt", encoding="utf-8") as file:
                    sample = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue

            if "commentary" not in sample:
                continue

            stats["total_frames"] += 1
            scenario = sample.get("scenario_name")
            scenario_key = "None" if scenario is None else str(scenario)
            stats["frames_per_scenario"][scenario_key] = (
                stats["frames_per_scenario"].get(scenario_key, 0) + 1
            )

            if sample.get("cause_object_visible_in_image"):
                stats["num_visible_objects"] += 1
            else:
                stats["num_not_visible_objects"] += 1

            visibility_values = [
                sample.get("cause_object_visible_in_image", False),
                sample.get("cause_object_visible_in_image_left_front", False),
                sample.get("cause_object_visible_in_image_right_front", False),
                sample.get("cause_object_visible_in_image_rear", False),
            ]
            if visibility_values[1]:
                stats["num_visible_objects_left_front"] += 1
            if visibility_values[2]:
                stats["num_visible_objects_right_front"] += 1
            if visibility_values[3]:
                stats["num_visible_objects_rear"] += 1
            if any(visibility_values):
                stats["num_visible_objects_any_view"] += 1

            template = sample.get("commentary_template")
            if template:
                templates.add(template)

        stats["num_different_templates"] = len(templates)
        stats["all_templates"] = sorted(templates)
        self.stats = stats

        with open(Path(self.output_directory) / "stats.json", "w", encoding="utf-8") as file:
            json.dump(stats, file, ensure_ascii=False, indent=4)
