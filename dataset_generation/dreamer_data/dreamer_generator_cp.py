"""Original Dreamer generator adapted to the four-view CP dataset layout.

Trajectory forecasting, alternative action generation, collision checking,
and safety labels remain implemented by ``dreamer_generator.py``.
"""

import gzip
import json
import random
from pathlib import Path

from dataset_generation.dreamer_data import dreamer_generator as base_dreamer
from dataset_generation.dreamer_data.dreamer_instructions_cp import (
    add_four_view_paths,
    get_info,
)


# ``process_data`` resolves this function in the original module namespace.
base_dreamer.get_info = get_info


class CarlaAlternativeCreator(base_dreamer.CarlaAlternativeCreator):
    """Run the original Dreamer implementation on four-view CP data."""

    def __init__(self, args):
        self.random_subset_count = args.random_subset_count
        self.sample_uniform_interval = args.sample_uniform_interval
        self.filter_routes_by_result = args.filter_routes_by_result

        self.save_viz = args.save_viz
        self.viz_for_video = args.viz_for_video
        self.save_samples = args.save_samples
        self.overwrite = args.overwrite
        self.save_instructions = args.save_instructions

        self.data_directory = args.data_directory
        self.viz_save_path = args.viz_save_path
        self.save_folder_name = args.save_folder_name

        self.original_image_size = args.original_image_size
        self.target_image_size = args.target_image_size
        self.original_fov = args.original_fov
        self.MIN_X = (
            self.original_image_size[0] // 2
            - self.target_image_size[0] // 2
        )
        self.MAX_X = (
            self.original_image_size[0] // 2
            + self.target_image_size[0] // 2
        )
        self.MIN_Y = args.min_y
        self.MAX_Y = args.max_y
        self.CAMERA_MATRIX = base_dreamer.build_projection_matrix(
            self.original_image_size[0],
            self.original_image_size[1],
            self.original_fov,
        )

        if self.save_viz:
            Path(self.viz_save_path).mkdir(parents=True, exist_ok=True)

        data_root = Path(self.data_directory)
        if (data_root / "data").is_dir():
            data_root = data_root / "data"

        self.data_boxes_paths = sorted(
            str(path) for path in data_root.glob("**/boxes/*.json.gz")
        )
        print(f"Number of data boxes: {len(self.data_boxes_paths)}")

        if self.random_subset_count > 0:
            random.Random(42).shuffle(self.data_boxes_paths)
            self.data_boxes_paths = self.data_boxes_paths[
                : self.random_subset_count
            ]
            self.data_boxes_paths.sort()

        if self.sample_uniform_interval > 1:
            self.data_boxes_paths = self.data_boxes_paths[
                :: self.sample_uniform_interval
            ]

    def process_data(self, path_id):
        """Generate original Dreamer labels and ensure four image fields."""

        result = super().process_data(path_id)
        self._append_four_view_paths(self.data_boxes_paths[path_id])
        return result

    def _append_four_view_paths(self, boxes_path):
        output_path = Path(
            boxes_path.replace(
                "/data/", f"/{self.save_folder_name}/"
            ).replace("boxes", self.save_folder_name)
        )
        if not output_path.is_file():
            return

        try:
            with gzip.open(output_path, "rt", encoding="utf-8") as file:
                dreamer_data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return

        if isinstance(dreamer_data, dict):
            sample_groups = dreamer_data.values()
        elif isinstance(dreamer_data, list):
            sample_groups = [dreamer_data]
        else:
            return

        for samples in sample_groups:
            for sample in samples:
                if "rgb_path" in sample:
                    add_four_view_paths(sample, "rgb_path")
                elif "path_rgb_image" in sample:
                    add_four_view_paths(sample, "path_rgb_image")

        with gzip.open(output_path, "wt", encoding="utf-8") as file:
            json.dump(dreamer_data, file, ensure_ascii=False, indent=4)
