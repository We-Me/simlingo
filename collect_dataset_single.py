#!/usr/bin/env python3
"""Select one route and run local data collection."""

import argparse
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def select_route(route_root, seed):
    """Use the same route groups as collect_dataset_slurm.py."""
    routes = list(route_root.glob("**/*balanced*/*.xml"))
    routes += list(route_root.glob("**/*lb1*/**/*.xml"))
    routes = sorted(set(path.resolve() for path in routes))

    if not routes:
        raise RuntimeError(f"No routes found under {route_root}")

    random.Random(seed).shuffle(routes)
    print(f"Found {len(routes)} candidate routes; seed={seed}")
    return routes[0]


def read_route(route_path):
    routes = ET.parse(str(route_path)).getroot().findall("route")
    if len(routes) != 1:
        raise RuntimeError(f"Expected one route in {route_path}, found {len(routes)}")
    return routes[0].get("town"), routes[0].get("id", "0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, default=None)
    parser.add_argument("--route-root", type=Path, default=ROOT / "data" / "simlingo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "database" / "four_view_single")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--tm-seed", type=int, default=100)
    args = parser.parse_args()

    route_path = args.route.resolve() if args.route else select_route(args.route_root.resolve(), args.seed)
    if not route_path.is_file():
        raise RuntimeError(f"Route not found: {route_path}")

    town, route_id = read_route(route_path)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    data_dir = args.output.resolve() / "data"
    result_dir = args.output.resolve() / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    result_name = f"{route_path.stem}_route{route_id}_{timestamp}"
    checkpoint = result_dir / f"{result_name}.json"
    debug_checkpoint = result_dir / f"{result_name}.txt"
    agent = ROOT / "team_code" / "data_agent_cp.py"
    evaluator = ROOT / "leaderboard" / "leaderboard" / "leaderboard_evaluator_local.py"

    env = os.environ.copy()
    env.update({
        "DATAGEN": "1",
        "SAVE_PATH": str(data_dir),
        "SAVE_TF_LABELS": os.environ.get("SAVE_TF_LABELS", "0"),
        "TMP_VISU": os.environ.get("TMP_VISU", "0"),
        "TOWN": town,
        "REPETITION": os.environ.get("REPETITION", "0"),
        "ROUTES": str(route_path),
        "TEAM_AGENT": str(agent),
        "TEAM_CONFIG": str(route_path),
        "DEBUG_CHALLENGE": "0",
    })

    command = [
        sys.executable,
        str(evaluator),
        f"--port={args.port}",
        f"--traffic-manager-port={args.tm_port}",
        f"--traffic-manager-seed={args.tm_seed}",
        f"--routes={route_path}",
        "--repetitions=1",
        "--track=MAP",
        f"--checkpoint={checkpoint}",
        f"--debug-checkpoint={debug_checkpoint}",
        f"--agent={agent}",
        f"--agent-config={route_path}",
        "--debug=0",
        "--resume=0",
        f"--timeout={os.environ.get('TIMEOUT', '600')}",
    ]

    print(f"Route: {route_path}")
    print(f"Town: {town}")
    print(f"Output: {args.output.resolve()}")
    return subprocess.run(command, cwd=str(ROOT), env=env).returncode


if __name__ == "__main__":
    sys.exit(main())

