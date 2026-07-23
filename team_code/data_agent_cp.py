"""Four-view data collection agent.

The large bounding-box and expert-driving implementations are inherited from
the original modules. Only configuration selection, camera declaration,
per-frame camera parsing, and sensor persistence are specialized here.
"""

import gzip
import json
import os
from pathlib import Path

import cv2
import laspy
import numpy as np

import data_agent as base_data_agent
from autopilot_cp import AutoPilot
import transfuser_utils as t_u

from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign
from agents.navigation.local_planner import LocalPlanner
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_entry_point():
  return "DataAgent"


class DataAgent(base_data_agent.DataAgent, AutoPilot):
  """Collect front, left-front, right-front, and rear camera views."""

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    # Do not call base_data_agent.DataAgent.setup(): it creates the original
    # rgb_augmented directories. Initialize the same state with CP directories.
    AutoPilot.setup(self, path_to_conf_file, route_index, traffic_manager)

    self.SAVE_TF_LABELS = int(os.environ.get("SAVE_TF_LABELS", 0))
    self.weather_tmp = None
    self.step_tmp = 0
    self.tm = traffic_manager
    self.scenario_name = Path(path_to_conf_file).parent.name
    self.cutin_vehicle_starting_position = None

    if self.save_path is not None and self.datagen:
      (self.save_path / "lidar").mkdir()
      (self.save_path / "boxes").mkdir()
      for view in self.config.camera_views:
        (self.save_path / view["save_dir"]).mkdir()
        if self.SAVE_TF_LABELS:
          (self.save_path / self._label_dir("semantics", view)).mkdir()
          (self.save_path / self._label_dir("depth", view)).mkdir()
      if self.SAVE_TF_LABELS:
        (self.save_path / "bev_semantics").mkdir()

    self.tmp_visu = int(os.environ.get("TMP_VISU", 0))
    self._active_traffic_light = None
    self.last_lidar = None
    self.last_ego_transform = None

  @staticmethod
  def _label_id(modality, view):
    """Preserve the original front-view IDs and suffix the other views."""
    if view["name"] == "front":
      return modality
    return f'{modality}_{view["name"]}'

  @classmethod
  def _label_dir(cls, modality, view):
    return cls._label_id(modality, view)

  def _init(self, hd_map):
    # The original DataAgent creates a second BEV manager for the randomly
    # augmented camera. Four fixed views share one ego-centric BEV label.
    AutoPilot._init(self, hd_map)

    obs_config = {
        "width_in_pixels": self.config.lidar_resolution_width,
        "pixels_ev_to_bottom": self.config.lidar_resolution_height / 2.0,
        "pixels_per_meter": self.config.pixels_per_meter_collection,
        "history_idx": [-1],
        "scale_bbox": True,
        "scale_mask_col": 1.0,
        "map_folder": "maps_2ppm_cv",
    }

    self.stop_sign_criteria = RunStopSign(self._world)
    self.ss_bev_manager = ObsManager(obs_config, self.config)
    self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=self.stop_sign_criteria)
    self._local_planner = LocalPlanner(self._vehicle, opt_dict={}, map_inst=self.world_map)

  def sensors(self):
    ego_vehicle = CarlaDataProvider.get_hero_actor()
    if ego_vehicle is not None and ego_vehicle.type_id != self.config.expected_ego_vehicle_type:
      raise RuntimeError(
          "The four-view camera rig is calibrated for "
          f"{self.config.expected_ego_vehicle_type}, but the spawned ego vehicle is "
          f"{ego_vehicle.type_id}. Recalibrate camera_views before collecting data."
      )

    sensors = AutoPilot.sensors(self)

    if self.save_path is not None and (self.datagen or self.tmp_visu):
      for view in self.config.camera_views:
        position = view["position"]
        rotation = view["rotation"]
        sensors.append({
            "type": "sensor.camera.rgb",
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "roll": rotation[0],
            "pitch": rotation[1],
            "yaw": rotation[2],
            "width": view["width"],
            "height": view["height"],
            "fov": view["fov"],
            "id": view["id"],
        })

    sensors.append({
        "type": "sensor.lidar.ray_cast",
        "x": self.config.lidar_pos[0],
        "y": self.config.lidar_pos[1],
        "z": self.config.lidar_pos[2],
        "roll": self.config.lidar_rot[0],
        "pitch": self.config.lidar_rot[1],
        "yaw": self.config.lidar_rot[2],
        "rotation_frequency": self.config.lidar_rotation_frequency,
        "points_per_second": self.config.lidar_points_per_second,
        "id": "lidar",
    })

    if self.SAVE_TF_LABELS:
      for view in self.config.camera_views:
        position = view["position"]
        rotation = view["rotation"]
        for modality, sensor_type in (
            ("semantics", "sensor.camera.semantic_segmentation"),
            ("depth", "sensor.camera.depth"),
        ):
          sensors.append({
              "type": sensor_type,
              "x": position[0],
              "y": position[1],
              "z": position[2],
              "roll": rotation[0],
              "pitch": rotation[1],
              "yaw": rotation[2],
              "width": view["width"],
              "height": view["height"],
              "fov": view["fov"],
              "id": self._label_id(modality, view),
          })

    return sensors

  def tick(self, input_data):
    rgb_views = {}
    semantic_views = {}
    depth_views = {}

    if self.save_path is not None and (self.datagen or self.tmp_visu):
      for view in self.config.camera_views:
        rgb_views[view["name"]] = input_data[view["id"]][1][:, :, :3]
        if self.SAVE_TF_LABELS:
          semantic_id = self._label_id("semantics", view)
          depth_id = self._label_id("depth", view)
          semantic_views[view["name"]] = input_data[semantic_id][1][:, :, 2]
          raw_depth = input_data[depth_id][1][:, :, :3]
          depth_views[view["name"]] = (t_u.convert_depth(raw_depth) * 255.0 + 0.5).astype(np.uint8)
    else:
      for view in self.config.camera_views:
        rgb_views[view["name"]] = None
        if self.SAVE_TF_LABELS:
          semantic_views[view["name"]] = None
          depth_views[view["name"]] = None

    # The 10 Hz LiDAR delivers half a sweep per 20 Hz simulation step.
    if self.last_lidar is not None:
      ego_transform = self._vehicle.get_transform()
      ego_location = ego_transform.location
      last_ego_location = self.last_ego_transform.location
      relative_translation = np.array([
          ego_location.x - last_ego_location.x,
          ego_location.y - last_ego_location.y,
          ego_location.z - last_ego_location.z,
      ])

      ego_yaw = ego_transform.rotation.yaw
      last_ego_yaw = self.last_ego_transform.rotation.yaw
      relative_rotation = np.deg2rad(t_u.normalize_angle_degree(ego_yaw - last_ego_yaw))
      orientation_target = np.deg2rad(ego_yaw)
      rotation_matrix = np.array([
          [np.cos(orientation_target), -np.sin(orientation_target), 0.0],
          [np.sin(orientation_target), np.cos(orientation_target), 0.0],
          [0.0, 0.0, 1.0],
      ])
      relative_translation = rotation_matrix.T @ relative_translation
      lidar_last = t_u.algin_lidar(self.last_lidar, relative_translation, relative_rotation)
      lidar_360 = np.concatenate((input_data["lidar"], lidar_last), axis=0)
    else:
      lidar_360 = input_data["lidar"]

    bounding_boxes = self.get_bounding_boxes(lidar=lidar_360)
    self.stop_sign_criteria.tick(self._vehicle)

    result = {
        "lidar": lidar_360,
        "rgb": rgb_views["front"],
        "rgb_views": rgb_views,
        "bounding_boxes": bounding_boxes,
    }

    if self.SAVE_TF_LABELS:
      bev_semantics = self.ss_bev_manager.get_observation(self.close_traffic_lights)
      if self.tmp_visu:
        self.visualuize(bev_semantics["rendered"], rgb_views["front"])
      result.update({
          "semantic_views": semantic_views,
          "depth_views": depth_views,
          "bev_semantics": bev_semantics["bev_semantic_classes"],
      })

    return result

  def augment_camera(self, sensors):
    """Fixed camera views do not require per-frame augmentation updates."""

  def save_sensors(self, tick_data):
    frame = self.step // self.config.data_save_freq

    for view in self.config.camera_views:
      view_name = view["name"]
      cv2.imwrite(
          str(self.save_path / view["save_dir"] / f"{frame:04}.jpg"),
          tick_data["rgb_views"][view_name],
      )

      if self.SAVE_TF_LABELS:
        cv2.imwrite(
            str(self.save_path / self._label_dir("semantics", view) / f"{frame:04}.png"),
            tick_data["semantic_views"][view_name],
        )
        cv2.imwrite(
            str(self.save_path / self._label_dir("depth", view) / f"{frame:04}.png"),
            tick_data["depth_views"][view_name],
        )

    if self.SAVE_TF_LABELS:
      cv2.imwrite(
          str(self.save_path / "bev_semantics" / f"{frame:04}.png"),
          tick_data["bev_semantics"],
      )

    header = laspy.LasHeader(point_format=self.config.point_format)
    header.offsets = np.min(tick_data["lidar"], axis=0)
    header.scales = np.array([
        self.config.point_precision,
        self.config.point_precision,
        self.config.point_precision,
    ])

    with laspy.open(self.save_path / "lidar" / f"{frame:04}.laz", mode="w", header=header) as writer:
      point_record = laspy.ScaleAwarePointRecord.zeros(tick_data["lidar"].shape[0], header=header)
      point_record.x = tick_data["lidar"][:, 0]
      point_record.y = tick_data["lidar"][:, 1]
      point_record.z = tick_data["lidar"][:, 2]
      writer.write_points(point_record)

    with gzip.open(self.save_path / "boxes" / f"{frame:04}.json.gz", "wt", encoding="utf-8") as file:
      json.dump(tick_data["bounding_boxes"], file, indent=4)
