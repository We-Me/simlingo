#!/usr/bin/env python3
"""Run reproducible SimLingo instruction-following demos on Bench2Drive routes.

When executed directly this file is a small test orchestrator.  When imported
by the Leaderboard evaluator it exposes InstructionLingoAgent, a thin wrapper
around the repository's regular LingoAgent.  The wrapper injects either one
English instruction or a declarative sequence and adds a display-only
translation overlay; it does not change the base driving or control
implementation.
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
SCENE1_ROUTE_PATH = Path(__file__).with_name("routes") / "scene1_basic_sequence.xml"
SCENE1_SEQUENCE_PATH = Path(__file__).with_name("scene1_sequence.json")


def get_entry_point() -> str:
    """Leaderboard hook used when this file is loaded as an agent module."""

    return "InstructionLingoAgent"


# Route bindings deliberately separate the language suite from the CARLA test
# placement.  "partial" means the stock route covers only part of a compound
# instruction, which is printed as a warning before execution.
ROUTE_BINDINGS: Mapping[str, Tuple[str, str, str]] = {
    "S1-SEQUENCE": ("9001", "Town12", "custom: 2.2 km, clear daytime, no active scenarios"),
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


def _load_sequence_config(path: Path) -> dict:
    """Load and validate the small declarative instruction schedule."""

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    route_id = config.get("route_id")
    warmup = config.get("warmup")
    phases = config.get("phases")
    if not isinstance(route_id, str) or not route_id:
        raise ValueError(f"Sequence route_id must be a non-empty string: {path}")
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"Instruction sequence has no phases: {path}")

    supported_triggers = {
        "distance",
        "phase_distance",
        "nav_command",
        "nav_command_complete",
        "lane_change_or_distance",
    }
    phase_ids = set()
    if warmup is not None:
        if not isinstance(warmup, dict):
            raise ValueError(f"Sequence warmup must be an object: {path}")
        warmup_id = warmup.get("id")
        if not isinstance(warmup_id, str) or not warmup_id:
            raise ValueError(f"Sequence warmup has no id: {path}")
        if warmup.get("mode") != "instruction_following":
            raise ValueError(f"Sequence warmup must use instruction_following mode: {path}")
        warmup_instruction = warmup.get("instruction")
        if not isinstance(warmup_instruction, str) or not warmup_instruction.strip():
            raise ValueError(f"Sequence warmup has no English instruction: {path}")
        trigger = warmup.get("advance_when")
        if not isinstance(trigger, dict) or trigger.get("type") != "speed":
            raise ValueError(f"Sequence warmup must have a speed trigger: {path}")
        if float(trigger.get("min_speed_mps", 0.0)) <= 0.0:
            raise ValueError(f"Sequence warmup min_speed_mps must be positive: {path}")
        if int(trigger.get("consecutive_ticks", 0)) <= 0:
            raise ValueError(f"Sequence warmup consecutive_ticks must be positive: {path}")
        if float(trigger.get("timeout_s", 0.0)) <= 0.0:
            raise ValueError(f"Sequence warmup timeout_s must be positive: {path}")
        phase_ids.add(warmup_id)

    for index, phase in enumerate(phases):
        phase_id = phase.get("id")
        instruction = phase.get("instruction")
        if not isinstance(phase_id, str) or not phase_id:
            raise ValueError(f"Sequence phase {index + 1} has no id: {path}")
        if phase_id in phase_ids:
            raise ValueError(f"Duplicate sequence phase id {phase_id}: {path}")
        phase_ids.add(phase_id)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Sequence phase {phase_id} has no English instruction: {path}")

        trigger = phase.get("advance_when")
        is_last = index == len(phases) - 1
        if is_last and trigger is not None:
            raise ValueError(f"Final sequence phase {phase_id} must remain active to route end")
        if not is_last:
            if not isinstance(trigger, dict):
                raise ValueError(f"Sequence phase {phase_id} has no advance_when rule")
            trigger_type = trigger.get("type")
            if trigger_type not in supported_triggers:
                raise ValueError(
                    f"Unsupported trigger {trigger_type!r} in sequence phase {phase_id}"
                )

    return config


if __name__ != "__main__":
    # Heavy model/CARLA imports are intentionally skipped in the lightweight
    # parent process that launches the evaluator.  This prevents the launcher
    # from creating a second CUDA context on a 24 GB GPU.
    # The upstream agent historically imports scenario_logger as a top-level
    # module because its own directory is normally inserted into sys.path by
    # Leaderboard.  This wrapper lives elsewhere, so provide the legacy alias
    # explicitly before importing the base agent.
    from team_code import scenario_logger as _scenario_logger

    sys.modules.setdefault("scenario_logger", _scenario_logger)

    if os.environ.get("SIMLINGO_DISABLE_BACKGROUND", "0") == "1":
        # RouteScenario creates moving background traffic and periodically
        # spawns parked meshes even without active route scenarios. Disable
        # both only for the isolated Scene 1 command run.
        from leaderboard.scenarios import route_scenario as _route_scenario
        from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle as _Idle

        def _disabled_background_behavior(*_args, **kwargs):
            return _Idle(name=kwargs.get("name", "BackgroundActivityDisabled"))

        def _disabled_parked_vehicle_spawning(*_args, **_kwargs):
            return None

        _route_scenario.BackgroundBehavior = _disabled_background_behavior
        _route_scenario.RouteScenario.spawn_parked_vehicles = _disabled_parked_vehicle_spawning
        print(
            "[scene1] Moving traffic and parked-vehicle spawning disabled for the "
            "isolated basic-command route.",
            flush=True,
        )

    import carla as _carla

    from team_code.agent_simlingo import LingoAgent as _BaseLingoAgent
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

    class InstructionLingoAgent(_BaseLingoAgent):
        """SimLingo agent with either one instruction or a Scene 1 sequence."""

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
            self._set_instruction_following(instruction)
            self._verify_instruction_id = os.environ.get("SIMLINGO_INSTRUCTION_ID", "")
            self._verify_translation = os.environ.get("SIMLINGO_INSTRUCTION_TRANSLATION", "")
            self._hide_translation = os.environ.get("SIMLINGO_HIDE_TRANSLATION", "0") == "1"
            self._verify_translation_font = self._load_translation_font()
            self._sequence_phases = None
            self._sequence_warmup = None
            self._sequence_warmup_active = False
            self._sequence_warmup_start_step = None
            self._sequence_warmup_speed_ticks = 0
            self._sequence_index = 0
            self._sequence_total_distance = None
            self._sequence_progress_m = 0.0
            self._sequence_phase_start_m = 0.0
            self._sequence_seen_nav_command = False
            self._sequence_origin_lane = None
            self._sequence_lane_opportunity_seen = False
            self._sequence_last_transition_status = "start"
            self._sequence_pending_transition = None
            self._sequence_log_path = os.environ.get("SIMLINGO_SEQUENCE_LOG", "")

            sequence_path = os.environ.get("SIMLINGO_SEQUENCE_PATH", "").strip()
            if sequence_path:
                sequence_config = _load_sequence_config(Path(sequence_path))
                self._sequence_phases = sequence_config["phases"]
                self._sequence_warmup = sequence_config.get("warmup")
                if self._sequence_warmup:
                    self._activate_sequence_warmup("route start")
                else:
                    self._activate_sequence_phase(0, "route start", status="start")

            if not self._sequence_phases:
                print(
                    f"[instruction] {self._verify_instruction_id}: {self.custom_prompt}",
                    flush=True,
                )
                if self._verify_translation:
                    print(f"[display-only translation] {self._verify_translation}", flush=True)

        def _init(self):
            super()._init()
            if self._sequence_phases:
                self._sequence_total_distance = float(sum(self._route_planner.route_distances))
                print(
                    f"[scene1] Interpolated route length: {self._sequence_total_distance:.1f} m",
                    flush=True,
                )

        def tick(self, input_data):
            # Apply transitions at the start of a model tick so the English
            # prompt, Chinese overlay, and inference always describe the same
            # active phase. Trigger observations are collected one tick earlier.
            if self._sequence_pending_transition is not None:
                next_index, reason, status = self._sequence_pending_transition
                self._sequence_pending_transition = None
                self._activate_sequence_phase(next_index, reason, status=status)
            result = super().tick(input_data)
            if self._sequence_phases and self._sequence_total_distance is not None:
                current_speed_mps = float(input_data["speed"][1]["speed"])
                self._update_instruction_sequence(current_speed_mps)
            return result

        def _set_instruction_following(self, instruction):
            """Apply the exact prompt setup shared by single and chained tests."""

            self.user_flag = 1
            self.user_command = instruction
            self.custom_prompt = instruction

        def _activate_sequence_warmup(self, reason):
            """Observe unscored startup before phase-one measurement begins."""

            warmup = self._sequence_warmup
            self._sequence_warmup_active = True
            self._sequence_warmup_start_step = None
            self._sequence_warmup_speed_ticks = 0
            self._verify_instruction_id = warmup["id"]
            self._verify_translation = "" if self._hide_translation else warmup.get("translation_zh", "")

            # Match the single-instruction invocation exactly: the same
            # <INSTRUCTION_FOLLOWING> path and the same keep-lane text are used.
            self._set_instruction_following(warmup["instruction"])
            self._sequence_last_transition_status = "warmup"

            record = {
                "step": getattr(self, "step", -1),
                "progress_m": round(self._sequence_progress_m, 2),
                "phase_index": -1,
                "phase_id": warmup["id"],
                "mode": "instruction_following",
                "scored": False,
                "instruction": warmup["instruction"],
                "translation_zh": warmup.get("translation_zh", ""),
                "reason": reason,
                "transition_status": "warmup",
            }
            print(
                f"[scene1 warm-up] {warmup['id']}: {warmup['instruction']} "
                f"(not scored; {reason})",
                flush=True,
            )
            if self._verify_translation:
                print(f"[display-only translation] {self._verify_translation}", flush=True)
            if self._sequence_log_path:
                with Path(self._sequence_log_path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        def _activate_sequence_phase(self, index, reason, status="observed"):
            self._sequence_warmup_active = False
            self._sequence_index = index
            phase = self._sequence_phases[index]
            self._verify_instruction_id = phase["id"]
            self._set_instruction_following(phase["instruction"])
            self._verify_translation = "" if self._hide_translation else phase.get("translation_zh", "")
            self._sequence_phase_start_m = self._sequence_progress_m
            self._sequence_seen_nav_command = False
            self._sequence_origin_lane = None
            self._sequence_lane_opportunity_seen = False
            self._sequence_last_transition_status = status

            trigger = phase.get("advance_when") or {}
            if trigger.get("type") == "nav_command_complete":
                target_value = int(trigger["value"])
                self._sequence_seen_nav_command = (
                    self._nav_command_value(self.last_command_tmp) == target_value
                )
            elif trigger.get("type") == "lane_change_or_distance":
                self._sequence_origin_lane = self._legal_left_lane_origin()
                self._sequence_lane_opportunity_seen = self._sequence_origin_lane is not None

            record = {
                "step": getattr(self, "step", -1),
                "progress_m": round(self._sequence_progress_m, 2),
                "phase_index": index,
                "phase_id": phase["id"],
                "instruction": phase["instruction"],
                "translation_zh": phase.get("translation_zh", ""),
                "reason": reason,
                "transition_status": status,
            }
            print(
                f"[scene1 phase {index + 1}/{len(self._sequence_phases)}] "
                f"{phase['id']} at {self._sequence_progress_m:.1f} m: "
                f"{phase['instruction']} ({status}: {reason})",
                flush=True,
            )
            if self._verify_translation:
                print(f"[display-only translation] {self._verify_translation}", flush=True)
            if self._sequence_log_path:
                with Path(self._sequence_log_path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        @staticmethod
        def _nav_command_value(command):
            return getattr(command, "value", command)

        @staticmethod
        def _current_waypoint():
            try:
                hero = CarlaDataProvider.get_hero_actor()
                carla_map = CarlaDataProvider.get_map()
            except (RuntimeError, ValueError):
                return None
            if hero is None or carla_map is None:
                return None
            return carla_map.get_waypoint(hero.get_location())

        @classmethod
        def _current_lane_identity(cls):
            waypoint = cls._current_waypoint()
            if waypoint is None:
                return None
            return waypoint.road_id, waypoint.lane_id

        @classmethod
        def _legal_left_lane_origin(cls):
            waypoint = cls._current_waypoint()
            if waypoint is None or waypoint.is_junction:
                return None
            left_lane = waypoint.get_left_lane()
            if (
                left_lane is None
                or left_lane.lane_type != _carla.LaneType.Driving
                or left_lane.lane_id * waypoint.lane_id <= 0
            ):
                return None
            lane_change = waypoint.left_lane_marking.lane_change
            if lane_change not in (_carla.LaneChange.Left, _carla.LaneChange.Both):
                return None
            return (
                waypoint.road_id,
                waypoint.lane_id,
                left_lane.road_id,
                left_lane.lane_id,
            )

        def _lane_change_completed(self):
            current_lane = self._current_lane_identity()
            origin_lane = self._sequence_origin_lane
            if current_lane is None or origin_lane is None:
                return False
            current_road, current_lane_id = current_lane
            _, _, target_road, target_lane_id = origin_lane
            return current_road == target_road and current_lane_id == target_lane_id

        def _update_instruction_sequence(self, current_speed_mps):
            remaining = float(sum(self._route_planner.route_distances))
            self._sequence_progress_m = max(0.0, self._sequence_total_distance - remaining)

            if self._sequence_warmup_active:
                trigger = self._sequence_warmup["advance_when"]
                min_speed_mps = float(trigger["min_speed_mps"])
                consecutive_ticks = int(trigger["consecutive_ticks"])
                timeout_s = float(trigger["timeout_s"])

                if self._sequence_warmup_start_step is None:
                    self._sequence_warmup_start_step = self.step
                if current_speed_mps >= min_speed_mps:
                    self._sequence_warmup_speed_ticks += 1
                else:
                    self._sequence_warmup_speed_ticks = 0

                if self._sequence_warmup_speed_ticks >= consecutive_ticks:
                    self._sequence_pending_transition = (
                        0,
                        f"warm-up speed reached {current_speed_mps:.1f} m/s",
                        "warmup_complete",
                    )
                    return

                elapsed_s = (
                    self.step - self._sequence_warmup_start_step
                ) / float(self.config.carla_fps)
                if elapsed_s >= timeout_s:
                    raise RuntimeError(
                        "Scene 1 warm-up failed: the single-instruction-compatible "
                        "keep-lane call did not reach "
                        f"{min_speed_mps:.1f} m/s for {consecutive_ticks} consecutive "
                        f"ticks within {timeout_s:.1f} simulation seconds "
                        f"(current speed {current_speed_mps:.1f} m/s)"
                    )
                return

            phase = self._sequence_phases[self._sequence_index]
            trigger = phase.get("advance_when")
            if not trigger or self._sequence_index >= len(self._sequence_phases) - 1:
                return

            trigger_type = trigger["type"]
            active_distance = self._sequence_progress_m - self._sequence_phase_start_m
            should_advance = False
            reason = ""

            if trigger_type == "distance":
                target_distance = float(trigger["at_m"])
                should_advance = self._sequence_progress_m >= target_distance
                reason = f"route progress reached {target_distance:.0f} m"
                transition_status = "scheduled"

            elif trigger_type == "phase_distance":
                target_distance = float(trigger["distance_m"])
                should_advance = active_distance >= target_distance
                reason = f"phase distance reached {target_distance:.0f} m"
                transition_status = "scheduled"

            elif trigger_type == "nav_command":
                nav_value = self._nav_command_value(self.last_command_tmp)
                target_value = int(trigger["value"])
                after_m = float(trigger.get("after_m", 0.0))
                should_advance = self._sequence_progress_m >= after_m and nav_value == target_value
                reason = f"navigation command {target_value} became active"
                transition_status = "observed"
                deadline_m = trigger.get("deadline_m")
                if (
                    not should_advance
                    and deadline_m is not None
                    and self._sequence_progress_m > float(deadline_m)
                ):
                    raise RuntimeError(
                        "Scene 1 route did not expose navigation command "
                        f"{target_value} by {float(deadline_m):.0f} m; current command is {nav_value}"
                    )

            elif trigger_type == "nav_command_complete":
                nav_value = self._nav_command_value(self.last_command_tmp)
                target_value = int(trigger["value"])
                if nav_value == target_value:
                    self._sequence_seen_nav_command = True
                min_active_m = float(trigger.get("min_active_m", 10.0))
                should_advance = (
                    self._sequence_seen_nav_command
                    and nav_value != target_value
                    and active_distance >= min_active_m
                )
                reason = f"navigation maneuver {target_value} completed"
                transition_status = "observed"
                max_active_m = trigger.get("max_active_m")
                if (
                    not should_advance
                    and max_active_m is not None
                    and active_distance > float(max_active_m)
                ):
                    raise RuntimeError(
                        "Navigation maneuver did not complete within "
                        f"{float(max_active_m):.0f} m after phase activation"
                    )

            elif trigger_type == "lane_change_or_distance":
                max_active_m = float(trigger.get("max_active_m", 150.0))
                if self._sequence_origin_lane is None:
                    self._sequence_origin_lane = self._legal_left_lane_origin()
                    if self._sequence_origin_lane is not None:
                        self._sequence_lane_opportunity_seen = True
                        print(
                            f"[scene1] Legal same-direction left lane detected at "
                            f"{self._sequence_progress_m:.1f} m.",
                            flush=True,
                        )
                lane_changed = self._lane_change_completed()
                should_advance = lane_changed or active_distance >= max_active_m
                if lane_changed:
                    reason = "ego entered the adjacent same-direction lane"
                    transition_status = "observed"
                elif self._sequence_lane_opportunity_seen:
                    reason = f"left lane change was not observed within {max_active_m:.0f} m"
                    transition_status = "not_observed"
                else:
                    reason = f"no legal same-direction left lane was found within {max_active_m:.0f} m"
                    transition_status = "invalid_route"

            else:
                raise RuntimeError(f"Unsupported sequence trigger type: {trigger_type}")

            if should_advance:
                self._sequence_pending_transition = (
                    self._sequence_index + 1,
                    reason,
                    transition_status,
                )

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
            sequence_status = ""
            if self._sequence_phases:
                if self._sequence_warmup_active:
                    sequence_status = f"[warm-up | {self._sequence_progress_m:.0f} m] "
                else:
                    sequence_status = (
                        f"[{self._sequence_index + 1}/{len(self._sequence_phases)} | "
                        f"{self._sequence_progress_m:.0f} m] "
                    )
            lines = self._wrap_overlay_text(
                self._verify_translation_font,
                f"{sequence_status}用户译文（不输入模型）：{self._verify_translation}",
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

        def destroy(self, results=None):
            """Release demo resources without assuming optional Hydra keys.

            The upstream destroy() directly reads cfg.data_module.encoder, but
            that key is absent from the released SimLingo checkpoint config.
            This verification agent does not need encoder-specific cleanup, so
            delete only resources that were actually created.
            """

            self.running = False
            if getattr(self, "_pygame", None) is not None:
                self._pygame.quit()
                self._pygame = None
                self._viz_screen = None

            for attribute in ("model", "processor", "config"):
                if hasattr(self, attribute):
                    delattr(self, attribute)


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
        scenario_elements = route.findall("./scenarios/scenario")
        scenario_types = [
            scenario.attrib.get("type", "unknown")
            for scenario in scenario_elements
            if scenario.attrib.get("metadata_only", "false").lower() != "true"
        ]
        metadata_anchors = [
            scenario.attrib.get("name", "unnamed")
            for scenario in scenario_elements
            if scenario.attrib.get("metadata_only", "false").lower() == "true"
        ]
        weather_element = route.find("./weathers/weather")
        weather = dict(weather_element.attrib) if weather_element is not None else {}
        routes[route.attrib["id"]] = {
            "town": route.attrib.get("town", "unknown"),
            "scenarios": scenario_types,
            "metadata_anchors": metadata_anchors,
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
        "--scene1-sequence",
        action="store_true",
        help="Run the chained Scene 1 commands on the custom 2.2 km Town12 route",
    )
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
    if args.dry_run:
        return routes_path, checkpoint
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
    sequence_config: Optional[dict] = None,
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
            "SIMLINGO_HIDE_TRANSLATION": "1" if args.hide_translation else "0",
            "SIMLINGO_VIZ_TITLE": f"SimLingo instruction {item['id']} - route {route_id}",
            "SIMLINGO_SEQUENCE_PATH": "",
            "SIMLINGO_SEQUENCE_LOG": "",
            "SIMLINGO_DISABLE_BACKGROUND": "0",
            "ROUTES": str(args.routes.resolve()),
            "SAVE_PATH": f"{save_path}{os.sep}",
        }
    )
    if sequence_config is not None:
        env.update(
            {
                "SIMLINGO_SEQUENCE_PATH": str(SCENE1_SEQUENCE_PATH.resolve()),
                "SIMLINGO_SEQUENCE_LOG": str((case_dir / "sequence_transitions.jsonl").resolve()),
                "SIMLINGO_DISABLE_BACKGROUND": "1",
            }
        )

    _print_instruction(item, "" if args.hide_translation else translation, route_id)
    scenario_label = ", ".join(route_info["scenarios"]) or "no active scenarios"
    print(f"CARLA placement: {route_info['town']} / {scenario_label}")
    if sequence_config is not None:
        warmup = sequence_config.get("warmup")
        if warmup:
            trigger = warmup["advance_when"]
            print(
                "Warm-up (not scored): single-instruction-compatible keep-lane call until "
                f"{float(trigger['min_speed_mps']):.1f} m/s for "
                f"{int(trigger['consecutive_ticks'])} consecutive ticks"
            )
        print("Chained phases:")
        for index, phase in enumerate(sequence_config["phases"], start=1):
            print(f"  {index}. {phase['id']}: {phase['instruction']}")
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
    if sequence_config is not None:
        metadata["instruction_sequence"] = sequence_config
        metadata["background_traffic_disabled"] = True
        metadata["parked_vehicle_spawning_disabled"] = True
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

    sequence_config = None
    if args.scene1_sequence:
        sequence_config = _load_sequence_config(SCENE1_SEQUENCE_PATH)
        args.routes = SCENE1_ROUTE_PATH
        selected_ids = ["S1-SEQUENCE"]
    else:
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

    if sequence_config is not None:
        first_phase = sequence_config["phases"][0]
        sequence_item = {
            "id": "S1-SEQUENCE",
            "scene_id": "S1",
            "scene_name": "Basic voice-controlled driving (chained custom route)",
            "instruction": first_phase["instruction"],
            "expected_behavior": "Execute all five Scene 1 commands in order on one continuous route.",
        }
        route_id = sequence_config["route_id"]
        if route_id not in known_routes:
            raise ValueError(f"Custom Scene 1 route id {route_id} is missing from {args.routes}")
        return _run_case(
            args,
            sequence_item,
            first_phase.get("translation_zh", ""),
            route_id,
            known_routes[route_id],
            checkpoint,
            result_root,
            0,
            sequence_config=sequence_config,
        )

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
