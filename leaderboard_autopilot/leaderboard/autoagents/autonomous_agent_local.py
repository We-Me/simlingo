#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla
from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface


class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

class AutonomousAgent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file, route_index=None):
        # Evaluator 调用 self.agent_instance = agent_class_obj(args.agent_config, config.index) 实际创建的是 DataAgent
        # 继承关系 DataAgent → AutoPilot → AutonomousAgentLocal
        # 前两者没有定义 __init__()，所以最终进入这里

        # 给 Agent 设置一个默认 Track
        # 之后会覆盖为 autonomous_agent.Track.MAP
        # 这里使用 autonomous_agent 的估计是为了对齐框架
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None                    # 为 GPS 坐标路线表示预留变量
        self._global_plan_world_coord = None        # 为 CARLA 世界坐标路线表示预留变量

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()   # 创建 Agent 自己的传感器数据接口

        self.wallclock_t0 = None                    # 保存 Agent 第一次运行时的现实时间起点

    # override
    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        pass

    # override
    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    # override
    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    # override
    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def __call__(self, sensors=None):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data(GameTime.get_frame())

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        sim_ratio = 0 if wallclock_diff == 0 else timestamp/wallclock_diff

        print('=== [Agent] -- Wallclock = {} -- System time = {} -- Game time = {} -- Ratio = {}x'.format(
            str(wallclock)[:-3], format(wallclock_diff, '.3f'), format(timestamp, '.3f'), format(sim_ratio, '.3f')))

        control = self.run_step(input_data, timestamp, sensors)
        control.manual_gear_shift = False

        return control

    @staticmethod
    def get_ros_version():
        return -1

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        # 两个参数表示同一条路线，只是坐标系不同
        self.org_dense_route_gps = global_plan_gps                      # 保存完整的稠密 GPS 路线
        self.org_dense_route_world_coord = global_plan_world_coord      # 保存完整的 CARLA 世界坐标路线

        ds_ids = downsample_route(global_plan_world_coord, 200)         # 对完整路线降采样

        # 按照 ds_ids 从稠密世界坐标路线中选点
        self._global_plan_world_coord = [
            (
                global_plan_world_coord[x][0], 
                global_plan_world_coord[x][1]
            )
            for x in ds_ids
        ]
        # 使用同一组索引，对 GPS 路线降采样
        # 确保 _global_plan_world_coord 和 _global_plan 路径点一致
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        