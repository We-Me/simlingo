#!/usr/bin/env python3
"""Run reproducible SimLingo instruction-following demos on Bench2Drive routes.

When executed directly this file is a small test orchestrator.  When imported
by the Leaderboard evaluator it exposes InstructionLingoAgent, a thin wrapper
around the repository's regular LingoAgent.  The wrapper only injects one
English instruction and adds a display-only translation overlay; it does not
change the base driving or control implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = Path(__file__).with_name("instructions_en.json")
TRANSLATION_PATH = Path(__file__).with_name("instructions_zh.json")


def get_entry_point() -> str:
    """Leaderboard hook used when this file is loaded as an agent module."""

    return "InstructionLingoAgent"


# Route bindings deliberately separate the language suite from the CARLA test
# placement.  "partial" means the stock route covers only part of a compound
# instruction, which is printed as a warning before execution.
ROUTE_BINDINGS: Mapping[str, Tuple[str, str, str]] = {
    "S1-01": ("3178", "Town12", "compatible: clear daytime vanilla route"),
    "S1-02": ("3178", "Town12", "compatible: clear daytime vanilla route"),
    "S1-03": ("3178", "Town12", "partial: clear daytime; verify the speed limit before judging 60 km/h"),
    "S1-04": ("3178", "Town12", "partial: clear daytime; verify the speed limit before judging 16.7 m/s"),
    "S1-05": ("3178", "Town12", "compatible: clear daytime vanilla route"),
    "S1-06": ("3178", "Town12", "compatible: clear daytime vanilla route"),
    "S1-07": ("4937", "Town12", "exact: clear daytime signalized right turn"),
    "S1-08": ("2086", "Town12", "exact: clear daytime non-signalized left turn"),
    "S1-09": ("17655", "Town12", "exact: clear daytime sequential lane change"),
    "S1-10": ("17655", "Town12", "exact: clear daytime sequential lane change"),
    "S1-11": ("3178", "Town12", "compatible: clear daytime vanilla route"),
    "S1-12": ("3178", "Town12", "partial: run after a stopped-state test"),
    "S2-01": ("14194", "Town12", "partial: pedestrian crossing; no guaranteed slow vehicle afterward"),
    "S2-02": ("14194", "Town12", "partial: pedestrian crossing; no guaranteed slow vehicle afterward"),
    "S2-03": ("3248", "Town13", "analog: curbside parked-vehicle occlusion and crossing pedestrian"),
    "S2-04": ("3248", "Town13", "analog: curbside parked-vehicle occlusion and crossing pedestrian"),
    "S2-05": ("1773", "Town12", "exact: parked obstacle"),
    "S2-06": ("1825", "Town12", "exact: two-way construction obstacle"),
    "S2-07": ("3086", "Town12", "compatible: bicycle crossing flow"),
    "S2-08": ("2204", "Town12", "exact: blocked intersection"),
    "S3-01": ("1825", "Town12", "exact environment: two-way construction hazard in heavy rain at night"),
    "S3-02": ("3072", "Town12", "exact environment: highway cut-in in rain at night"),
    "S3-03": ("1825", "Town12", "compatible: construction and lane obstruction in heavy rain at night"),
    "S3-04": ("1825", "Town12", "compatible: construction and narrowed lane in heavy rain at night"),
    "S3-05": ("3072", "Town12", "compatible: highway following in rain at night"),
    "S3-06": ("3540", "Town13", "compatible: hard-brake route in heavy rain at night"),
    "S3-07": ("3540", "Town13", "exact: hard-brake route in heavy rain at night"),
    "S3-08": ("3380", "Town13", "exact: yield to emergency vehicle in heavy rain at night"),
    "S3-09": ("3482", "Town13", "exact hazard: opening vehicle door at night"),
    "S3-10": ("3737", "Town13", "exact hazard: vehicle-turning pedestrian in heavy rain"),
}

PRESETS: Mapping[str, Sequence[str]] = {
    # One representative command from each required basic-track scene.
    "three-scenes": ("S1-07", "S2-01", "S3-02"),
    # Extra Town13 emergency/OOD coverage.
    "town13-emergency": ("S3-07", "S3-08", "S3-09", "S3-10"),
}


if __name__ != "__main__":
    # Heavy model/CARLA imports are intentionally skipped in the lightweight
    # parent process that launches the evaluator.  This prevents the launcher
    # from creating a second CUDA context on a 24 GB GPU.
    from team_code.agent_simlingo import LingoAgent as _BaseLingoAgent

    class InstructionLingoAgent(_BaseLingoAgent):
        """Base SimLingo agent with one immutable English instruction."""

        def setup(self, path_to_conf_file, route_index=None):
            instruction = os.environ.get("SIMLINGO_INSTRUCTION_TEXT", "").strip()
            if not instruction:
                raise RuntimeError("SIMLINGO_INSTRUCTION_TEXT is required")

            super().setup(path_to_conf_file, route_index)
            if os.environ.get("SIMLINGO_REQUIRE_COT", "1") == "1" and not self.config.use_cot:
                raise RuntimeError("This verification run requires GlobalConfig.use_cot=True")

            # user_flag=1 keeps route conditioning and activates the trained
            # <INSTRUCTION_FOLLOWING> prompt.  Only the English string below is
            # passed into the model.
            self.user_flag = 1
            self.user_command = instruction
            self.custom_prompt = instruction
            self._verify_instruction_id = os.environ.get("SIMLINGO_INSTRUCTION_ID", "")
            self._verify_translation = os.environ.get("SIMLINGO_INSTRUCTION_TRANSLATION", "")
            self._verify_translation_font = self._load_translation_font()

            print(
                f"[instruction] {self._verify_instruction_id}: {self.custom_prompt}",
                flush=True,
            )
            if self._verify_translation:
                print(f"[display-only translation] {self._verify_translation}", flush=True)

        def _load_translation_font(self):
            """Load an optional CJK font without making it a runtime requirement."""

            if self._pygame is None:
                return None
            candidates = [
                os.environ.get("SIMLINGO_CJK_FONT", ""),
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            ]
            for candidate in candidates:
                if candidate and Path(candidate).is_file():
                    try:
                        return self._pygame.font.Font(candidate, 19)
                    except Exception:
                        continue
            if self._verify_translation:
                print(
                    "[visualization] No CJK font found; the Chinese translation "
                    "will remain available in the terminal.",
                    flush=True,
                )
            return None

        @staticmethod
        def _wrap_overlay_text(font, text: str, max_width: int) -> List[str]:
            lines: List[str] = []
            current = ""
            for character in text:
                candidate = current + character
                if current and font.size(candidate)[0] > max_width:
                    lines.append(current)
                    current = character
                else:
                    current = candidate
            if current:
                lines.append(current)
            return lines[:2]

        def _render_live_viz(self, pred_speed_wps, pred_route, language, speed, control):
            # The base window renders the CARLA camera, predicted paths, control,
            # exact English model prompt, and generated language answer.
            super()._render_live_viz(pred_speed_wps, pred_route, language, speed, control)

            # Draw the Chinese translation only after inference and after the base
            # prompt has been rendered.  It can never enter the model input.
            if (
                not self._verify_translation
                or self._verify_translation_font is None
                or self._pygame is None
                or self._viz_screen is None
            ):
                return

            width, _ = self._viz_screen.get_size()
            lines = self._wrap_overlay_text(
                self._verify_translation_font,
                f"用户译文（不输入模型）：{self._verify_translation}",
                width - 36,
            )
            overlay_height = 12 + 27 * len(lines)
            overlay = self._pygame.Surface((width, overlay_height), self._pygame.SRCALPHA)
            overlay.fill((8, 10, 14, 210))
            for index, line in enumerate(lines):
                rendered = self._verify_translation_font.render(line, True, (245, 245, 248))
                overlay.blit(rendered, (18, 6 + index * 27))
            self._viz_screen.blit(overlay, (0, 0))
            self._pygame.display.flip()


def _load_suite() -> Tuple[Dict[str, dict], Dict[str, str]]:
    with SUITE_PATH.open("r", encoding="utf-8") as handle:
        suite = json.load(handle)
    with TRANSLATION_PATH.open("r", encoding="utf-8") as handle:
        translation_data = json.load(handle)

    instructions: Dict[str, dict] = {}
    for scene in suite["scenes"]:
        for item in scene["instructions"]:
            enriched = dict(item)
            enriched["scene_id"] = scene["scene_id"]
            enriched["scene_name"] = scene["name"]
            instructions[item["id"]] = enriched
    translations = translation_data["translations"]
    return instructions, translations


def _read_routes(routes_path: Path) -> Dict[str, dict]:
    root = ET.parse(routes_path).getroot()
    routes: Dict[str, dict] = {}
    for route in root.iter("route"):
        scenario_types = [
            scenario.attrib.get("type", "unknown")
            for scenario in route.findall("./scenarios/scenario")
        ]
        weather_element = route.find("./weathers/weather")
        weather = dict(weather_element.attrib) if weather_element is not None else {}
        routes[route.attrib["id"]] = {
            "town": route.attrib.get("town", "unknown"),
            "scenarios": scenario_types,
            "weather": weather,
        }
    return routes


def _print_instruction(item: dict, translation: str, route_id: str) -> None:
    binding = ROUTE_BINDINGS[item["id"]]
    print(f"\n[{item['id']}] {item['scene_name']}")
    print(f"English (model input): {item['instruction']}")
    if translation:
        print(f"Chinese (display only): {translation}")
    print(f"Expected: {item['expected_behavior']}")
    print(f"Route: {route_id} / {binding[1]} / {binding[2]}")


def _select_instruction_ids(args, instructions: Mapping[str, dict]) -> List[str]:
    if args.instruction_id:
        if args.instruction_id not in instructions:
            raise ValueError(f"Unknown instruction id: {args.instruction_id}")
        return [args.instruction_id]
    if args.scene:
        selected = [
            item_id
            for item_id, item in instructions.items()
            if item["scene_id"] == args.scene
            and (not args.core_only or item["priority"] == "core")
        ]
        return selected
    if args.preset == "core":
        return [
            item_id
            for item_id, item in instructions.items()
            if item["priority"] == "core"
        ]
    if args.preset == "all":
        return list(instructions)
    return list(PRESETS[args.preset or "three-scenes"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run visible CARLA + Pygame SimLingo instruction tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--instruction-id", help="Run one instruction, for example S1-03")
    selection.add_argument("--scene", choices=("S1", "S2", "S3"), help="Run one scene's suite")
    selection.add_argument(
        "--preset",
        choices=("three-scenes", "core", "all", "town13-emergency"),
        help="Run a predefined group; the default covers all three basic-track scenes",
    )
    selection.add_argument("--list", action="store_true", help="List instructions without running CARLA")
    parser.add_argument("--core-only", action="store_true", help="With --scene, run only core items")
    parser.add_argument("--route-id", help="Override the route for a single instruction")
    parser.add_argument(
        "--routes",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "data" / "bench2drive220.xml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--gpu-rank", type=int, default=int(os.environ.get("GPU_RANK", "0")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "10000")))
    parser.add_argument("--tm-port", type=int, default=int(os.environ.get("TM_PORT", "30000")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("TIMEOUT", "600")))
    parser.add_argument("--traffic-manager-seed", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--hide-translation", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only")
    return parser


def _validate_runtime(args, checkpoint: Path) -> Tuple[Path, Path]:
    routes_path = args.routes.expanduser().resolve()
    if not routes_path.is_file():
        raise FileNotFoundError(f"Routes XML not found: {routes_path}")
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model_config = checkpoint.parent.parent.parent / ".hydra" / "config.yaml"
    if not model_config.is_file():
        raise FileNotFoundError(
            f"Hydra config not found: {model_config}. Keep the downloaded model tree intact."
        )
    carla_root = Path(os.environ.get("CARLA_ROOT", "")).expanduser()
    if not (carla_root / "CarlaUE4.sh").is_file():
        raise FileNotFoundError(f"CARLA launcher not found under CARLA_ROOT={carla_root}")
    return routes_path, checkpoint


def _run_case(
    args,
    item: dict,
    translation: str,
    route_id: str,
    route_info: dict,
    checkpoint: Path,
    result_root: Path,
    case_index: int,
) -> int:
    case_dir = result_root / item["id"] / f"route_{route_id}"
    save_path = case_dir / "viz"

    evaluator = REPO_ROOT / "Bench2Drive" / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
    command = [
        sys.executable,
        str(evaluator),
        f"--routes={args.routes.resolve()}",
        f"--routes-subset={route_id}",
        f"--repetitions={args.repetitions}",
        "--track=SENSORS",
        f"--checkpoint={case_dir / 'result.json'}",
        f"--debug-checkpoint={case_dir / 'live.txt'}",
        f"--timeout={args.timeout}",
        f"--agent={Path(__file__).resolve()}",
        f"--agent-config={checkpoint}",
        f"--traffic-manager-seed={args.traffic_manager_seed}",
        f"--port={args.port + case_index * 10}",
        f"--traffic-manager-port={args.tm_port + case_index * 10}",
        f"--gpu-rank={args.gpu_rank}",
    ]

    env = os.environ.copy()
    env.update(
        {
            "CARLA_RENDER_OFFSCREEN": "0",
            "SIMLINGO_VIZ": "1",
            "SIMLINGO_REQUIRE_COT": "1",
            "SIMLINGO_INSTRUCTION_ID": item["id"],
            "SIMLINGO_INSTRUCTION_TEXT": item["instruction"],
            "SIMLINGO_INSTRUCTION_TRANSLATION": "" if args.hide_translation else translation,
            "SIMLINGO_VIZ_TITLE": f"SimLingo instruction {item['id']} - route {route_id}",
            "ROUTES": str(args.routes.resolve()),
            "SAVE_PATH": f"{save_path}{os.sep}",
        }
    )

    _print_instruction(item, "" if args.hide_translation else translation, route_id)
    print(f"CARLA placement: {route_info['town']} / {', '.join(route_info['scenarios'])}")
    weather = route_info["weather"]
    if weather:
        print(
            "Weather: "
            f"cloud={weather.get('cloudiness', '?')}%, "
            f"rain={weather.get('precipitation', '?')}%, "
            f"fog={weather.get('fog_density', '?')}%, "
            f"sun_altitude={weather.get('sun_altitude_angle', '?')} deg"
        )
    print("Command:", " ".join(command), flush=True)
    if args.dry_run:
        return 0

    case_dir.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "instruction_id": item["id"],
        "scene_id": item["scene_id"],
        "scene_name": item["scene_name"],
        "english_model_input": item["instruction"],
        "chinese_display_only": "" if args.hide_translation else translation,
        "expected_behavior": item["expected_behavior"],
        "prompt_prefix": "<INSTRUCTION_FOLLOWING>",
        "with_cot_config": True,
        "route_id": route_id,
        "route_info": route_info,
        "route_binding_note": ROUTE_BINDINGS[item["id"]][2],
        "checkpoint": str(checkpoint),
        "command": command,
    }
    (case_dir / "case.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return completed.returncode


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    instructions, translations = _load_suite()

    if args.list:
        for item_id, item in instructions.items():
            route_id = ROUTE_BINDINGS[item_id][0]
            _print_instruction(item, translations.get(item_id, ""), route_id)
        return 0

    if args.route_id and not args.instruction_id:
        parser.error("--route-id is valid only together with --instruction-id")

    selected_ids = _select_instruction_ids(args, instructions)
    checkpoint_default = Path(
        os.environ.get(
            "CHECKPOINT",
            REPO_ROOT / "pretrained" / "simlingo" / "checkpoints" / "epoch=013.ckpt" / "pytorch_model.pt",
        )
    )
    args.routes, checkpoint = _validate_runtime(args, args.checkpoint or checkpoint_default)
    known_routes = _read_routes(args.routes)

    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    result_root = (args.result_dir or REPO_ROOT / "eval_results" / "verify_instruction" / timestamp)
    result_root = result_root.expanduser().resolve()

    print("SimLingo mode: with language conditioning / GlobalConfig.use_cot=True")
    print("CARLA window: enabled")
    print("Pygame visualization: enabled")
    print(f"Selected tests: {', '.join(selected_ids)}")
    print(f"Results: {result_root}")

    for index, item_id in enumerate(selected_ids):
        item = instructions[item_id]
        route_id = args.route_id if args.route_id else ROUTE_BINDINGS[item_id][0]
        if route_id not in known_routes:
            raise ValueError(f"Route id {route_id} is not present in {args.routes}")
        if not args.route_id:
            expected_town = ROUTE_BINDINGS[item_id][1]
            actual_town = known_routes[route_id]["town"]
            if actual_town != expected_town:
                raise ValueError(
                    f"Route binding mismatch for {item_id}: expected {expected_town}, got {actual_town}"
                )
        return_code = _run_case(
            args,
            item,
            translations.get(item_id, ""),
            route_id,
            known_routes[route_id],
            checkpoint,
            result_root,
            index,
        )
        if return_code != 0:
            print(f"Test {item_id} failed with evaluator exit code {return_code}.", file=sys.stderr)
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
