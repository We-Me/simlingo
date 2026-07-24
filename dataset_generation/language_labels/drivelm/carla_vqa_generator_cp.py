"""Original DriveLM VQA generator adapted to the four-view CP dataset.

The QA rules and the front-view object annotations remain in
``carla_vqa_generator.py``.  This module only adapts dataset discovery,
scenario lookup, and the saved image paths so the released generator can run
directly on ``database/four_view_single``.
"""

import gzip
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dataset_generation.language_labels.drivelm import carla_vqa_generator as base_vqa


REPO_ROOT = Path(__file__).resolve().parents[3]
CAMERA_DIRECTORIES = {
    "CAM_FRONT": "rgb",
    "CAM_FRONT_LEFT": "rgb_left_front",
    "CAM_FRONT_RIGHT": "rgb_right_front",
    "CAM_BACK": "rgb_rear",
}


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
        return "noScenario"

    route_group = None
    for box in boxes:
        if box.get("class") == "ego_info":
            route_group = box.get("scenario")
            break

    route_match = re.search(
        r"Rep\d+_(\d+)_route_?(\d+)", measurement_path.parent.parent.name
    )
    if not route_group or route_match is None:
        return route_group or "noScenario"

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

    # Preserve the original generator's rule: use the closest scenario whose
    # trigger point is already more than 10 m behind the ego vehicle.
    distances = [
        np.linalg.norm(local_position)
        if local_position[0] < -10
        else 999999
        for local_position in local_positions
    ]
    return scenarios[int(np.argmin(distances))]


# The original method resolves this helper in its module-global namespace.
base_vqa.get_scenario_name = get_scenario_name_cp


class QAsGenerator(base_vqa.QAsGenerator):
    """Run the original VQA rules on the four-view CP directory layout."""

    def __init__(self, args):
        self.TARGET_IMAGE_SIZE = args.target_image_size
        self.ORIGINAL_IMAGE_SIZE = args.original_image_size
        self.ORIGINAL_FOV = args.original_fov

        self.MIN_X = args.min_x
        self.MAX_X = args.max_x
        self.MIN_Y = args.min_y
        self.MAX_Y = args.max_y

        self.random_subset_count = args.random_subset_count
        self.sample_frame_mode = args.sample_frame_mode
        self.sample_uniform_interval = args.sample_uniform_interval

        self.save_examples = args.save_examples
        self.visualize_projection = args.visualize_projection
        self.filter_routes_by_result = args.filter_routes_by_result
        self.remove_pedestrian_scenarios = args.remove_pedestrian_scenarios

        self.data_directory = args.data_directory
        self.path_keyframes = args.path_keyframes
        self.output_directory = args.output_directory
        self.skip_existing = args.skip_existing

        self.CAMERA_MATRIX = base_vqa.build_projection_matrix(
            self.ORIGINAL_IMAGE_SIZE[0],
            self.ORIGINAL_IMAGE_SIZE[1],
            self.ORIGINAL_FOV,
        )

        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

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
            self.data_boxes_paths = self.data_boxes_paths[
                : self.random_subset_count
            ]
        self.data_boxes_paths.sort()

        self.list_next_junction_id_minus_one = []
        self.reset_qa_stats()

        if self.sample_frame_mode == "keyframes":
            with open(self.path_keyframes, "r", encoding="utf-8") as file:
                self.keyframes_list = [line.strip() for line in file]
            self.keyframes_list = [
                path.replace("/rgb/", "/boxes/").replace(
                    ".jpg", ".json.gz"
                )
                for path in self.keyframes_list
            ]

    def create_qa_pairs(self, path_id):
        """Generate the original QA output, then attach four image paths."""

        result = super().create_qa_pairs(path_id)
        self._append_four_view_image_paths(self.data_boxes_paths[path_id])
        return result

    def _append_four_view_image_paths(self, boxes_path):
        if "/data/" not in boxes_path:
            return

        relative_output = boxes_path.split("/data/", 1)[1].replace(
            "/boxes/", "/vqa/"
        )
        output_path = Path(self.output_directory) / relative_output
        if not output_path.is_file():
            return

        try:
            with gzip.open(output_path, "rt", encoding="utf-8") as file:
                sample = json.load(file)
        except (OSError, json.JSONDecodeError):
            return

        image_paths = sample.setdefault("image_paths", {})
        for camera_name in (
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ):
            image_paths.setdefault(camera_name, None)

        for camera_name, directory in CAMERA_DIRECTORIES.items():
            image_paths[camera_name] = boxes_path.replace(
                "/boxes/", f"/{directory}/"
            ).replace(".json.gz", ".jpg")

        with gzip.open(output_path, "wt", encoding="utf-8") as file:
            json.dump(sample, file, ensure_ascii=False, indent=4)
