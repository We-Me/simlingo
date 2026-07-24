"""Run the original Dreamer pipeline on four-view CP data."""

import argparse
import os
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


random.seed(42)


def _add_boolean_pair(group, name, default, help_text=None):
    destination = name.replace("-", "_")
    group.add_argument(
        f"--{name}",
        dest=destination,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        f"--no-{name}",
        dest=destination,
        action="store_false",
    )
    group.set_defaults(**{destination: default})


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate original Dreamer labels for four-view CP data"
    )

    path_group = parser.add_argument_group("Dataset and Path Settings")
    path_group.add_argument(
        "--data-directory",
        default="database/four_view_single",
        help="Dataset root containing data/**/boxes",
    )
    path_group.add_argument(
        "--save-folder-name",
        default="dreamer",
        help="Output folder replacing data/ and boxes/ in the original layout",
    )
    path_group.add_argument(
        "--viz-save-path", default="viz/dreamer_cp"
    )

    image_group = parser.add_argument_group("Image Parameters")
    image_group.add_argument(
        "--original-image-size", nargs=2, type=int, default=[1024, 512]
    )
    image_group.add_argument(
        "--target-image-size", nargs=2, type=int, default=[1024, 384]
    )
    image_group.add_argument(
        "--original-fov",
        type=float,
        default=70.0,
        help="FOV of the CP front camera",
    )
    image_group.add_argument("--min-y", type=int, default=0)
    image_group.add_argument("--max-y", type=int, default=384)

    sampling_group = parser.add_argument_group("Sampling Parameters")
    sampling_group.add_argument("--random-subset-count", type=int, default=-1)
    sampling_group.add_argument(
        "--sample-uniform-interval", type=int, default=1
    )

    output_group = parser.add_argument_group("Filtering and Output")
    _add_boolean_pair(
        output_group,
        "filter-routes-by-result",
        False,
        "Disabled because CP route results are stored separately",
    )
    _add_boolean_pair(output_group, "overwrite", False)
    _add_boolean_pair(output_group, "save-samples", True)
    _add_boolean_pair(output_group, "save-instructions", True)
    output_group.add_argument("--save-viz", action="store_true", default=False)
    output_group.add_argument(
        "--viz-for-video", action="store_true", default=False
    )
    output_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Use one worker for the first smoke test",
    )
    output_group.add_argument("--chunksize", type=int, default=100)

    return parser.parse_args()


def main():
    args = parse_arguments()

    import tqdm

    from dataset_generation.dreamer_data.dreamer_generator_cp import (
        CarlaAlternativeCreator,
    )

    creator = CarlaAlternativeCreator(args)
    num_frames = len(creator.data_boxes_paths)
    print(f"Processing {num_frames} data boxes")

    if args.workers > 1 and num_frames > 0:
        from tqdm.contrib.concurrent import process_map

        process_map(
            creator.process_data,
            range(num_frames),
            max_workers=min(args.workers, os.cpu_count() or args.workers),
            chunksize=args.chunksize,
        )
    else:
        for index in tqdm.tqdm(range(num_frames)):
            creator.process_data(index)


if __name__ == "__main__":
    main()
