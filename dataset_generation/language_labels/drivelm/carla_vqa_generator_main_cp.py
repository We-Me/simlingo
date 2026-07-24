"""Run the original DriveLM VQA pipeline on four-view CP data."""

import argparse
import os
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate original DriveLM VQA labels for four-view CP data"
    )

    path_group = parser.add_argument_group("Dataset and Path Settings")
    path_group.add_argument(
        "--path-keyframes",
        type=str,
        default="path/to/keyframes.txt",
    )
    path_group.add_argument(
        "--data-directory",
        type=str,
        default="database/four_view_single",
    )
    path_group.add_argument(
        "--output-directory",
        type=str,
        default="database/four_view_single/drivelm",
    )

    image_group = parser.add_argument_group("Image and Camera Parameters")
    image_group.add_argument(
        "--target-image-size", nargs=2, type=int, default=[1024, 384]
    )
    image_group.add_argument(
        "--original-image-size", nargs=2, type=int, default=[1024, 512]
    )
    image_group.add_argument(
        "--original-fov",
        type=float,
        default=70.0,
        help="FOV of the CP front camera",
    )
    image_group.add_argument("--min-y", type=int, default=0)
    image_group.add_argument("--max-y", type=int, default=358)

    sampling_group = parser.add_argument_group("Sampling Parameters")
    sampling_group.add_argument("--random-subset-count", type=int, default=-1)
    sampling_group.add_argument(
        "--sample-frame-mode",
        choices=["all", "keyframes", "uniform"],
        default="all",
    )
    sampling_group.add_argument(
        "--sample-uniform-interval", type=int, default=1
    )

    output_group = parser.add_argument_group("Filtering and Output")
    output_group.add_argument("--save-examples", action="store_true", default=False)
    output_group.add_argument(
        "--visualize-projection", action="store_true", default=False
    )
    output_group.add_argument(
        "--filter-routes-by-result",
        dest="filter_routes_by_result",
        action="store_true",
        help="Disabled by default because CP results are stored separately",
    )
    output_group.add_argument(
        "--no-filter-routes-by-result",
        dest="filter_routes_by_result",
        action="store_false",
    )
    output_group.add_argument(
        "--remove-pedestrian-scenarios",
        action="store_true",
        default=False,
    )
    output_group.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
    )
    output_group.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
    )
    output_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Use one worker for the first smoke test",
    )
    output_group.add_argument("--chunksize", type=int, default=100)
    parser.set_defaults(filter_routes_by_result=False, skip_existing=True)

    args = parser.parse_args()
    args.min_x = (
        args.original_image_size[0] // 2 - args.target_image_size[0] // 2
    )
    args.max_x = (
        args.original_image_size[0] // 2 + args.target_image_size[0] // 2
    )
    if args.max_y is None:
        args.max_y = args.target_image_size[1]
    return args


def main():
    args = parse_arguments()

    from tqdm import tqdm

    from dataset_generation.language_labels.drivelm.carla_vqa_generator_cp import (
        QAsGenerator,
    )

    generator = QAsGenerator(args)
    num_frames = len(generator.data_boxes_paths)

    if args.workers > 1 and num_frames > 0:
        from tqdm.contrib.concurrent import process_map

        process_map(
            generator.create_qa_pairs,
            range(num_frames),
            max_workers=min(args.workers, os.cpu_count() or args.workers),
            chunksize=args.chunksize,
        )
        print("Stats are not aggregated when using multiple workers.")
    else:
        for index in tqdm(range(num_frames)):
            generator.create_qa_pairs(index)

        if generator.frame_num > 0:
            generator.save_stats()
        else:
            print("No VQA frames were generated; stats.json was not written.")


if __name__ == "__main__":
    main()
