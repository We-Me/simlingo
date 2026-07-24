"""Thin four-view output adapter for the original Dreamer instructions."""

from dataset_generation.dreamer_data import dreamer_instructions as base_instructions


VIEW_DIRECTORIES = {
    "left_front": "rgb_left_front",
    "right_front": "rgb_right_front",
    "rear": "rgb_rear",
}


def add_four_view_paths(sample, front_key="rgb_path"):
    """Keep the original front field and append the other CP image paths."""

    front_path = sample.get(front_key)
    if not front_path or "/rgb/" not in front_path:
        return sample

    for view_name, directory in VIEW_DIRECTORIES.items():
        sample[f"{front_key}_{view_name}"] = front_path.replace(
            "/rgb/", f"/{directory}/"
        )
    return sample


def get_info(
    alternative_trajectories,
    route_adjusted,
    route_original,
    current_measurement,
    walker_close,
    ego_info,
):
    """Run the released instruction logic and only extend image paths."""

    dreamer_dict = base_instructions.get_info(
        alternative_trajectories,
        route_adjusted,
        route_original,
        current_measurement,
        walker_close,
        ego_info,
    )
    for samples in dreamer_dict.values():
        for sample in samples:
            add_four_view_paths(sample)
    return dreamer_dict
