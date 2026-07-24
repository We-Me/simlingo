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


COMMENTARY_OUTPUT_KEY_ORDER = (
    "image",
    "image_left_front",
    "image_right_front",
    "image_rear",
    "commentary",
    "commentary_with_weather_front",
    "commentary_with_weather_rear",
    "only_weather",
    "commentary_template",
    "placeholder",
    "cause_object_string",
    "cause_object",
    "cause_object_visible_in_image",
    "cause_object_visible_in_image_left_front",
    "cause_object_visible_in_image_right_front",
    "cause_object_visible_in_image_rear",
    "scenario_name",
)


COLOR_NAMES_ZH = {
    "black": "黑色",
    "blue": "蓝色",
    "dark blue": "深蓝色",
    "gray": "灰色",
    "grey": "灰色",
    "green": "绿色",
    "dark green": "深绿色",
    "maroon": "深红色",
    "navy": "深蓝色",
    "olive": "橄榄绿色",
    "orange": "橙色",
    "red": "红色",
    "white": "白色",
    "yellow": "黄色",
}


def analyze_weather(weather_box):
    """Map continuous CARLA weather parameters to one discrete category."""

    if not weather_box:
        return "unknown"

    try:
        sun_altitude = float(weather_box.get("sun_altitude_angle", 70.0))
        precipitation = float(weather_box.get("precipitation", 0.0))
        fog_density = float(weather_box.get("fog_density", 0.0))
        cloudiness = float(weather_box.get("cloudiness", 0.0))
    except (TypeError, ValueError):
        return "unknown"

    if sun_altitude < 0.0:
        light = "night"
    elif sun_altitude <= 15.0:
        light = "twilight"
    else:
        light = "day"

    # Use the most visually dominant condition when independently sampled
    # CARLA parameters describe more than one phenomenon at the same time.
    if fog_density >= 40.0:
        condition = "dense_fog"
    elif fog_density >= 10.0:
        condition = "fog"
    elif precipitation >= 60.0:
        condition = "heavy_rain"
    elif precipitation > 0.0:
        condition = "rain"
    elif cloudiness >= 60.0:
        condition = "overcast"
    else:
        condition = "clear"

    return f"{light}_{condition}"


def render_weather_commentary(commentary, weather_text):
    """Place one complete weather template before and after Commentary."""

    commentary = commentary.strip()
    weather_sentence = weather_text.strip()
    if weather_sentence and weather_sentence[-1] not in "。！？":
        weather_sentence += "。"
    return (
        f"{weather_sentence}{commentary}",
        f"{commentary}{weather_sentence}",
    )


def order_commentary_sample(sample):
    """Return a stable, human-readable field order without dropping extras."""

    ordered = {
        key: sample[key]
        for key in COMMENTARY_OUTPUT_KEY_ORDER
        if key in sample
    }
    ordered.update(
        (key, value)
        for key, value in sample.items()
        if key not in ordered
    )
    return ordered


def _vehicle_type_zh(object_box):
    type_id = object_box.get("type_id", "")
    if "firetruck" in type_id:
        return "消防车"
    if "police" in type_id:
        return "警车"
    if "ambulance" in type_id:
        return "救护车"
    if "jeep" in type_id:
        return "吉普车"
    if "micro" in type_id:
        return "小型车辆"
    if "nissan.patrol" in type_id:
        return "运动型多用途车"
    if "european_hgv" in type_id:
        return "重型货车"
    if "sprinter" in type_id:
        return "厢式货车"

    base_type = object_box.get("base_type", "vehicle")
    return {
        "bicycle": "自行车",
        "bus": "公交车",
        "car": "轿车",
        "motorcycle": "摩托车",
        "truck": "卡车",
        "van": "厢式货车",
        "vehicle": "车辆",
    }.get(base_type, "车辆")


def get_object_appearance_zh(object_box, english_appearance=""):
    """Describe the cause object in natural Chinese word order."""

    if object_box is None:
        return {
            "construction site": "前方施工区域",
            "red traffic light": "前方红灯",
            "stop sign": "前方停车标志",
            "traffic light": "前方交通信号灯",
            "vehicle": "前方车辆",
            "walker": "前方行人",
        }.get(english_appearance, "前方目标")

    object_class = object_box.get("class")
    if object_class == "traffic_light":
        state = str(object_box.get("state", "")).lower()
        state_zh = {"green": "绿灯", "red": "红灯", "yellow": "黄灯"}.get(
            state, "交通信号灯"
        )
        return f"前方{state_zh}"
    if object_class == "stop_sign":
        return "前方停车标志"
    if object_class == "static":
        return "前方施工区域"
    if object_class == "walker":
        return "前方儿童" if object_box.get("age") == "child" else "前方行人"
    if object_class != "car":
        return "前方目标"

    lateral_position = float(object_box.get("position", [0.0, 0.0])[1])
    if lateral_position > 2.0:
        direction = "右前方"
    elif lateral_position < -2.0:
        direction = "左前方"
    else:
        direction = "前方"

    color_name = str(object_box.get("color_name") or "").lower()
    color = COLOR_NAMES_ZH.get(color_name, "")
    color_rgb = object_box.get("color_rgb")
    if color_rgb in ([0, 28, 0], [12, 42, 12], [0, 21, 0]):
        color = "深绿色"
    elif color_rgb == [0, 12, 58]:
        color = "深蓝色"
    elif color_rgb == [211, 142, 0]:
        color = "黄色"
    elif color_rgb == [145, 255, 181]:
        color = "蓝色"
    elif color_rgb == [215, 88, 0]:
        color = "橙色"

    return f"{direction}的{color}{_vehicle_type_zh(object_box)}"


def _strip_sentence(text):
    return text.strip().rstrip(".").strip()


def _pick_template(templates_zh, key):
    return random.choice(templates_zh[key])


def _translate_route_action(route_action, templates_zh):
    route_action = _strip_sentence(route_action)
    direct = {
        "": "",
        "Do a lane change": "执行变道。",
        "Exit the parking lot": "驶出停车区域。",
        "Follow the route": "沿规划路线行驶。",
        "Prepare to do a lane change": "准备变道。",
        "Turn left": "在前方路口左转。",
        "Turn right": "在前方路口右转。",
    }
    if route_action in direct:
        return direct[route_action]

    scenario_actions = {
        "steer clear of the parked vehicle": _pick_template(templates_zh, "ParkedObstacle"),
        "overtake the bikes on your lane": _pick_template(templates_zh, "HazardAtSideLane"),
        "avoid the vehicle that is opening its door": _pick_template(templates_zh, "VehicleOpensDoor"),
        "give way to the emergency vehicle": _pick_template(templates_zh, "YieldToEmergencyVehicle"),
        "avoid the accident on your lane": _pick_template(templates_zh, "Accident"),
        "go around the construction site": _pick_template(templates_zh, "ConstructionObstacle"),
        "steer clear of the oncoming traffic entering your lane due to the construction cones": _pick_template(templates_zh, "InvadingTurn"),
        "avoid the oncoming traffic that is invading your lane": _pick_template(templates_zh, "InvadingTurnOLD"),
    }
    lowered = route_action.lower()
    if lowered in scenario_actions:
        return scenario_actions[lowered]

    if lowered.startswith("prepare to "):
        action = scenario_actions.get(lowered[len("prepare to ") :])
        if action:
            return f"准备{action.rstrip('。')}。"

    stay_match = re.fullmatch(r"stay on (.+?) to (.+)", lowered)
    if stay_match:
        lane_en, action_en = stay_match.groups()
        lane_zh = {
            "neighbouring lane": "相邻车道",
            "your current lane": "当前车道",
            "your current (oncoming) lane": "当前对向车道",
        }.get(lane_en, "当前车道")
        action_zh = scenario_actions.get(action_en)
        if action_zh:
            return f"继续沿{lane_zh}行驶，并{action_zh.rstrip('。')}。"

    if "make space for the traffic that invades the lane" in lowered:
        return f"{_pick_template(templates_zh, 'shift_right')}，为进入本车道的对向车辆留出空间。"
    if lowered.startswith("return to your original route"):
        return _pick_template(templates_zh, "go_back")

    # Unknown route-specific English should never leak into the Chinese label.
    return "沿规划路线行驶。"


def _distance_suffix(distance_value):
    if distance_value is None:
        return ""
    return f"，当前相距{distance_value}米"


def _translate_reason(reason, object_zh, distance_value, templates_zh):
    reason = _strip_sentence(reason)
    lowered = reason.lower()
    distance_suffix = _distance_suffix(distance_value)

    if not lowered:
        return ""
    if (
        "you cleared the stop sign" in lowered
        or "you cleared <object>" in lowered
    ):
        result = f"，因为{_pick_template(templates_zh, 'cleared_stop')}"
        if "junction is clear" in lowered:
            result += "，且路口已经畅通"
        elif "vehicle in the junction is moving away" in lowered:
            result += "，且路口内的车辆正在驶离"
        elif "pay attention to" in lowered:
            result += "，同时注意驶向路口的车辆"
        return result
    if "avoid a collision with" in lowered or "prevent a crash with" in lowered:
        phrase = _pick_template(templates_zh, "prevent_collision")
        return f"，{phrase.replace('<OBJECT>', object_zh)}"
    if "intersecting your path" in lowered or "crossing in front of you" in lowered:
        phrase = _pick_template(templates_zh, "cross_path")
        return f"，{phrase.replace('<OBJECT>', object_zh)}{distance_suffix}"
    if "that is crossing the road" in lowered:
        return f"，注意避让{object_zh}，该目标正在横穿道路{distance_suffix}"
    if "stopped because of a red traffic light" in lowered:
        return f"，与{object_zh}保持安全距离，该车正因红灯停车{distance_suffix}"
    if "slowing down because of a red traffic light" in lowered:
        return f"，跟随{object_zh}，该车正因红灯减速{distance_suffix}"
    if "drive closer to the stationary" in lowered:
        return f"，逐步靠近{object_zh}，该目标当前静止{distance_suffix}"
    if "to follow the" in lowered or "to follow <object>" in lowered:
        return f"，跟随{object_zh}{distance_suffix}"
    if "stay behind" in lowered:
        phrase = _pick_template(templates_zh, "stay_behind")
        return f"，{phrase.replace('<OBJECT>', object_zh)}{distance_suffix}"
    if "drive closer to" in lowered:
        phrase = _pick_template(templates_zh, "drive_closer")
        return f"，{phrase.replace('<OBJECT>', object_zh)}{distance_suffix}"
    if "reach the speed limit" in lowered:
        return "，逐步达到道路限速"
    if "drive according to the speed limit" in lowered:
        return "，按照道路限速行驶"
    if lowered.startswith("due to the") or lowered.startswith("due to <object>"):
        return f"，受{object_zh}影响{distance_suffix}"
    if "traffic light is green" in lowered:
        result = "，当前信号灯为绿灯"
        if "junction is clear" in lowered:
            result += "，且路口已经畅通"
        elif "pay attention" in lowered:
            result += "，但仍需注意驶向路口的车辆"
        return result
    if "drive through the junction" in lowered:
        result = "，安全通过当前路口"
        if "junction is clear" in lowered:
            result += "，路口当前畅通"
        elif "pay attention" in lowered:
            result += "，同时注意路口内及驶向路口的车辆"
        return result
    if "drive with the target speed" in lowered:
        return "，逐步达到目标车速"
    if "change to the neighbouring lane" in lowered:
        return f"，变入相邻车道，{_pick_template(templates_zh, 'gap_big')}"
    if "change to the oncoming lane" in lowered:
        return f"，变入对向车道，{_pick_template(templates_zh, 'gap_big')}"
    if "junction is clear" in lowered:
        return "，确认路口畅通后继续行驶"
    if "vehicle in the junction is moving away" in lowered:
        return "，确认路口内车辆正在驶离后继续行驶"
    if "pay attention to the vehicle" in lowered:
        return "，注意路口内及驶向路口的车辆"
    if object_zh:
        return f"，注意{object_zh}并根据路况调整车速{distance_suffix}"
    return "，根据当前道路状况调整车速"


def translate_commentary_zh(
    commentary, english_object, object_zh, templates_zh
):
    """Reassemble original Commentary semantics in Chinese word order."""

    text = commentary.strip()
    attention_en = "Pay attention to the walker and brake if necessary."
    has_walker_attention = attention_en in text
    text = text.replace(attention_en, "")

    pedestrian_exit_attention = (
        "Pay attention to the pedestrian on the exit of the junction."
    )
    has_exit_attention = pedestrian_exit_attention in text
    text = text.replace(pedestrian_exit_attention, "")

    if english_object:
        text = text.replace(f"the {english_object}", "<OBJECT>")
        text = text.replace(english_object, "<OBJECT>")
    text = text.replace("a vehicle", "<OBJECT>")

    distance_values = []

    def replace_distance(match):
        distance_values.append(match.group(1))
        return "<DISTANCE>"

    text = re.sub(
        r"\b(?:in|at) (-?\d+(?:\.\d+)?) meters\b",
        replace_distance,
        text,
    )
    distance_value = distance_values[0] if distance_values else None

    speed_actions = (
        ("Maintain the reduced speed", "maintain_reduced_speed"),
        ("Maintain your current speed", "maintain_speed"),
        ("Remain stopped", "remain_stopped"),
        ("Stop now", "stop_now"),
        ("Accelerate", "accelerate"),
        ("Decelerate", "decelerate"),
        ("Wait for a gap in the traffic before changing lanes", "wait_gap"),
    )
    matches = [
        (text.find(phrase), phrase, key)
        for phrase, key in speed_actions
        if text.find(phrase) >= 0
    ]
    if not matches:
        route_zh = _translate_route_action(text, templates_zh)
        speed_sentence = ""
    else:
        action_index, action_phrase, action_key = min(matches, key=lambda item: item[0])
        route_en = text[:action_index].strip()
        reason_en = text[action_index + len(action_phrase) :].strip()
        route_zh = _translate_route_action(route_en, templates_zh)

        if action_key == "wait_gap":
            speed_text = _pick_template(templates_zh, action_key)
            if "lane with oncoming traffic" in reason_en.lower():
                speed_text += "，并变入对向车道"
            speed_sentence = f"{speed_text}。"
        else:
            reason_zh = _translate_reason(
                reason_en, object_zh, distance_value, templates_zh
            )
            speed_sentence = f"{_pick_template(templates_zh, action_key)}{reason_zh}。"

    parts = [part for part in (route_zh, speed_sentence) if part]
    if has_exit_attention:
        parts.append("通过路口时注意出口附近的行人。")
    if has_walker_attention:
        parts.append("注意行人动态，必要时制动。")
    result = "".join(parts)
    return result.replace("。。", "。").replace("，。", "。")


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

        # Keep the English canonical fragments for the released rule engine,
        # then reassemble the selected semantics with dedicated Chinese
        # templates in ``generate_commentary``.
        template_file = (
            REPO_ROOT / "data/augmented_templates/commentary_subsentence.json"
        )
        with template_file.open("r", encoding="utf-8") as file:
            self.templates = ujson.load(file)
        template_file_zh = (
            REPO_ROOT
            / "data/augmented_templates/commentary_subsentence_zh.json"
        )
        with template_file_zh.open("r", encoding="utf-8") as file:
            self.templates_zh = ujson.load(file)
        weather_template_file = (
            REPO_ROOT / "data/augmented_templates/weather_template.json"
        )
        with weather_template_file.open("r", encoding="utf-8") as file:
            self.weather_templates = ujson.load(file)

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

    def generate_commentary(
        self,
        current_boxes,
        last_measurements_oldest_first,
        future_measurements,
        scenario_name,
    ):
        """Use original semantics and render the final text in Chinese."""

        commentary, visible, cause_object, english_object = (
            base_commentary.COMsGenerator.generate_commentary(
                self,
                current_boxes,
                last_measurements_oldest_first,
                future_measurements,
                scenario_name,
            )
        )
        object_zh = get_object_appearance_zh(cause_object, english_object)
        commentary_zh = translate_commentary_zh(
            commentary,
            english_object,
            object_zh,
            self.templates_zh,
        )
        return commentary_zh, visible, cause_object, object_zh

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

        commentary = sample.get("commentary", "")
        cause_object_string = sample.get("cause_object_string")
        commentary_template = commentary
        placeholder = {}
        if cause_object_string and cause_object_string in commentary_template:
            commentary_template = commentary_template.replace(
                cause_object_string, "<OBJECT>"
            )
            placeholder["<OBJECT>"] = cause_object_string

        distance_match = re.search(r"-?\d+(?:\.\d+)?米", commentary_template)
        if distance_match is not None:
            distance_text = distance_match.group(0)
            commentary_template = re.sub(
                r"-?\d+(?:\.\d+)?米",
                "<DISTANCE>",
                commentary_template,
            )
            placeholder["<DISTANCE>"] = distance_text

        sample["commentary_template"] = commentary_template
        sample["placeholder"] = placeholder

        weather_box = None
        try:
            with gzip.open(boxes_path, "rt", encoding="utf-8") as file:
                current_boxes = json.load(file)
            weather_box = next(
                (
                    box
                    for box in current_boxes
                    if box.get("class") == "weather"
                ),
                None,
            )
        except (OSError, json.JSONDecodeError, TypeError):
            pass

        weather_category = analyze_weather(weather_box)
        weather_options = self.weather_templates.get(
            weather_category,
            self.weather_templates["unknown"],
        )
        only_weather = random.choice(weather_options)
        commentary_with_weather_front, commentary_with_weather_rear = (
            render_weather_commentary(commentary, only_weather)
        )
        sample["commentary_with_weather_front"] = commentary_with_weather_front
        sample["commentary_with_weather_rear"] = commentary_with_weather_rear
        sample["only_weather"] = only_weather
        sample = order_commentary_sample(sample)

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
