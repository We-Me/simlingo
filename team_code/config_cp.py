"""Four-view data-collection configuration.

This module extends the original configuration without changing ``config.py``.
Camera extrinsics and output directory names are centralized in
``GlobalConfig.camera_views`` so the collection agent can iterate over them.
"""

from config import GlobalConfig as BaseGlobalConfig


class GlobalConfig(BaseGlobalConfig):
  """Original SimLingo configuration plus four fixed RGB camera views."""

  def __init__(self):
    super().__init__()

    # Fixed multi-view collection replaces the original random augmented camera.
    self.augment = 0

    # The original camera position below is tied to the ego blueprint spawned
    # by leaderboard_autopilot/leaderboard/scenarios/route_scenario.py.
    self.expected_ego_vehicle_type = "vehicle.lincoln.mkz_2020"

    # Use the surround-camera mounting positions from this repository's
    # Bench2Drive collector, which spawns the same vehicle.lincoln.mkz_2020.
    # This is more reliable than transferring LMDrive's MKZ 2017 extrinsics.
    # A negative yaw looks left and a positive yaw looks right. Keep the front
    # image in rgb/ for compatibility with existing SimLingo data consumers.
    self.camera_views = (
        {
            "name": "front",
            "id": "rgb",
            "save_dir": "rgb",
            "position": [0.80, 0.0, 1.60],
            "rotation": [0.0, 0.0, 0.0],
            "width": 1024,
            "height": 512,
            "fov": 70,
        },
        {
            "name": "left_front",
            "id": "rgb_left_front",
            "save_dir": "rgb_left_front",
            "position": [0.27, -0.55, 1.60],
            "rotation": [0.0, 0.0, -55.0],
            "width": 1024,
            "height": 512,
            "fov": 70,
        },
        {
            "name": "right_front",
            "id": "rgb_right_front",
            "save_dir": "rgb_right_front",
            "position": [0.27, 0.55, 1.60],
            "rotation": [0.0, 0.0, 55.0],
            "width": 1024,
            "height": 512,
            "fov": 70,
        },
        {
            "name": "rear",
            "id": "rgb_rear",
            "save_dir": "rgb_rear",
            "position": [-2.0, 0.0, 1.60],
            "rotation": [0.0, 0.0, 180.0],
            "width": 1024,
            "height": 512,
            "fov": 110,
        },
    )

    # Keep legacy single-camera projection helpers aligned with the new front
    # camera instead of the original [-1.5, 0.0, 2.0] position.
    self.camera_pos = list(self.camera_views[0]["position"])
    self.camera_rot_0 = list(self.camera_views[0]["rotation"])
    self.camera_width = self.camera_views[0]["width"]
    self.camera_height = self.camera_views[0]["height"]
    self.camera_fov = self.camera_views[0]["fov"]
