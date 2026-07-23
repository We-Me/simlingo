#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import print_function
import math
import os
import time

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.autoagents.ros_base_agent import ROSBaseAgent

DATAGEN = int(os.environ.get('DATAGEN', 0))

# 传感器安装半径限制
MAX_ALLOWED_RADIUS_SENSOR = 3.0
# 两组字典定义每种类型最多允许多少个传感器
# Qualifier 是资格赛/预选赛类赛道，限制更严格
QUALIFIER_SENSORS_LIMITS = {
    'sensor.camera.rgb': 4,
    'sensor.lidar.ray_cast': 1,
    'sensor.other.radar': 2,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1,
    'sensor.camera.depth': 4, # for data generation
    'sensor.camera.semantic_segmentation': 4 # for data generation
}
SENSORS_LIMITS = {
    'sensor.camera.rgb': 8,
    'sensor.lidar.ray_cast': 2,
    'sensor.other.radar': 4,
    'sensor.other.gnss': 1,
    'sensor.other.imu': 1,
    'sensor.opendrive_map': 1,
    'sensor.speedometer': 1,
    'sensor.camera.depth': 4, # for data generation
    'sensor.camera.semantic_segmentation': 4 # for data generation
}
# 合法传感器类型
ALLOWED_SENSORS = SENSORS_LIMITS.keys()


class AgentError(Exception):
    """
    Exceptions thrown when the agent returns an error during the simulation
    Agent 运行失败的异常
    AgentWrapper 自己并不会主动抛出 AgentError，真正包装异常的是 ScenarioManager
    """

    def __init__(self, message):
        super(AgentError, self).__init__(message)


class AgentWrapperFactory(object):

    @staticmethod
    def get_wrapper(agent):
        if isinstance(agent, ROSBaseAgent):
            return ROSAgentWrapper(agent)
        else:
            return AgentWrapper(agent)


def validate_sensor_configuration(
    sensors,            # agent.sensors() 返回的配置列表
    agent_track,        # Agent 的 self.track
    selected_track      # 命令行的 --track 字符串
):
    """
    Ensure that the sensor configuration is valid, in case the challenge mode is used
    Returns true on valid configuration, false otherwise
    """
    # 这个函数不是 ScenarioManager 调用的，而是 Evaluator 在 load_scenario() 之前调用

    # 检查 Track 是否一致
    if Track(selected_track) != agent_track:
        raise SensorConfigurationInvalid("You are submitting to the wrong track [{}]!".format(Track(selected_track)))

    sensor_count = {}
    sensor_ids = []

    for sensor in sensors:

        # Check if the is has been already used
        sensor_id = sensor['id']
        if sensor_id in sensor_ids:
            # 每个传感器 ID 必须唯一，因为后面数据通过 ID 组织
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(sensor_id))
        else:
            sensor_ids.append(sensor_id)

        # Check if the sensor is valid
        # Track.SENSORS → 禁止 OpenDRIVE
        # Track.MAP     → 允许 OpenDRIVE
        if agent_track == Track.SENSORS:
            if sensor['type'].startswith('sensor.opendrive_map'):
                raise SensorConfigurationInvalid("Illegal sensor 'sensor.opendrive_map' used for Track [{}]!".format(agent_track))

        # Check the sensors validity
        if sensor['type'] not in ALLOWED_SENSORS:
            # 只允许顶部 SENSORS_LIMITS 中出现的传感器类型
            raise SensorConfigurationInvalid("Illegal sensor '{}' used for Track [{}]!".format(sensor['type'], agent_track))

        # Check the extrinsics of the sensor
        # 检查安装半径
        if 'x' in sensor and 'y' in sensor and 'z' in sensor:
            if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > MAX_ALLOWED_RADIUS_SENSOR:
                raise SensorConfigurationInvalid(
                    "Illegal sensor extrinsics used for sensor '{}'. Max allowed radius is {}m".format(sensor['id'], MAX_ALLOWED_RADIUS_SENSOR))

        # Check the amount of sensors
        # 统计类型数量
        if sensor['type'] in sensor_count:
            sensor_count[sensor['type']] += 1
        else:
            sensor_count[sensor['type']] = 1

    # 选择数量限制
    if agent_track in (Track.SENSORS_QUALIFIER, Track.MAP_QUALIFIER):
        sensor_limits = SENSORS_LIMITS
    else:
        sensor_limits = QUALIFIER_SENSORS_LIMITS

    for sensor_type, max_instances_allowed in sensor_limits.items():
        if sensor_type in sensor_count and sensor_count[sensor_type] > max_instances_allowed:
            raise SensorConfigurationInvalid(
                "Too many {} used! "
                "Maximum number allowed is {}, but {} were requested.".format(sensor_type,
                                                                              max_instances_allowed,
                                                                              sensor_count[sensor_type]))


class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """
    _agent = None
    _sensors_list = []
    sensor_list_names = []

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent     # 没有重新创建 Agent，只保存引用

    def __call__(self):
        """
        Pass the call directly to the agent
        """
        # 传感器准备完成后，ScenarioManager 每帧执行 ego_action = self._agent_wrapper() 调用 Agent 对象
        # 由于 Agent 继承 AutonomousAgentLocal，下一步才会进入 AutonomousAgentLocal.__call__(sensors)
        # Agent 返回什么，Wrapper 就返回什么
        return self._agent(self.sensor_list_names)
        # return self._agent()

    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]     # CARLA 传感器类型
        id_ = sensor_spec["id"]         # Agent 访问数据时使用的名字
        attributes = {}                 # 稍后写入 CARLA blueprint 的参数

        if type_ == 'sensor.opendrive_map':
            attributes['reading_frequency'] = sensor_spec['reading_frequency']
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()
        elif type_ == 'sensor.speedometer':
            delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
            attributes['reading_frequency'] = 1 / delta_time
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_.startswith('sensor.camera'):
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        elif type_ == 'sensor.lidar.ray_cast':
            attributes['range'] = str(85)   # 固定最大距离
            if DATAGEN==1:
                attributes['rotation_frequency'] = str(sensor_spec['rotation_frequency'])
                attributes['points_per_second'] = str(sensor_spec['points_per_second'])
            else:
                attributes['rotation_frequency'] = str(10)
                attributes['points_per_second'] = str(600000)
            attributes['channels'] = str(64)
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(0.45)
            attributes['dropoff_intensity_limit'] = str(0.8)
            attributes['dropoff_zero_intensity'] = str(0.4)

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        elif type_ == 'sensor.other.radar':
            attributes['horizontal_fov'] = str(sensor_spec['horizontal_fov'])  # degrees
            attributes['vertical_fov'] = str(sensor_spec['vertical_fov'])  # degrees
            attributes['points_per_second'] = '1500'
            attributes['range'] = '100'  # meters

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        elif type_ == 'sensor.other.gnss':
            attributes['noise_alt_stddev'] = str(0.000005)
            attributes['noise_lat_stddev'] = str(0.000005)
            attributes['noise_lon_stddev'] = str(0.000005)
            attributes['noise_alt_bias'] = str(0.0)
            attributes['noise_lat_bias'] = str(0.0)
            attributes['noise_lon_bias'] = str(0.0)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation()
        elif type_ == 'sensor.other.imu':
            attributes['noise_accel_stddev_x'] = str(0.001)
            attributes['noise_accel_stddev_y'] = str(0.001)
            attributes['noise_accel_stddev_z'] = str(0.015)
            attributes['noise_gyro_stddev_x'] = str(0.001)
            attributes['noise_gyro_stddev_y'] = str(0.001)
            attributes['noise_gyro_stddev_z'] = str(0.001)

            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
            
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes

    def setup_sensors(self, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        # 取得当前已经加载地图的 CARLA world
        world = CarlaDataProvider.get_world()
        # 获取 Blueprint Library
        bp_library = world.get_blueprint_library()

        # 之前 Evaluator 验证配置时已经调用过一次
        # 第一次调用用于 validate，第二次调用用于真正创建
        # 要求 sensors() 是确定性的
        for sensor_spec in self._agent.sensors():
            type_, id_, sensor_transform, attributes = \
                self._preprocess_sensor_spec(sensor_spec)

            # These are the pseudosensors (not spawned)
            # 创建伪传感器
            if type_ == 'sensor.opendrive_map':
                sensor = OpenDriveMapReader(vehicle, attributes['reading_frequency'])
            elif type_ == 'sensor.speedometer':
                sensor = SpeedometerReader(vehicle, attributes['reading_frequency'])

            # These are the sensors spawned on the carla world
            # 创建 CARLA Sensor Actor
            else:
                bp = bp_library.find(type_)
                for key, value in attributes.items():
                    bp.set_attribute(str(key), str(value))
                sensor = CarlaDataProvider\
                    .get_world()\
                    .spawn_actor(
                        bp,                 # 传感器 Blueprint
                        sensor_transform,   # 相对车辆的安装位姿
                        vehicle             # 父 Actor，即 ego vehicle
                    )

            # setup callback
            # 注册数据回调
            sensor.listen(CallBack(id_, type_, sensor, self._agent.sensor_interface))
            # 保存所有传感器对象，用于最后清理
            self._sensors_list.append(sensor)
            # 保存 ID 和真实对象的对应关系
            self.sensor_list_names.append([sensor_spec['id'], sensor])

        # Some sensors miss sending data during the first ticks, so tick several times and remove the data
        # 创建传感器后连续推进 10 帧，让传感器开始稳定输出
        for _ in range(10):
            world.tick()

    def cleanup(self):
        """
        Remove and destroy all sensors
        清理传感器
        """
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None

        self._sensors_list.clear()
        self.sensor_list_names.clear()

        # Tick once to destroy the sensors
        # CARLA 中很多 actor 操作会在下一次 tick 正式生效
        # 这一步确保销毁请求被 Server 处理
        CarlaDataProvider.get_world().tick()


class ROSAgentWrapper(AgentWrapper):

    SENSOR_TYPE_REMAPS = {
        "sensor.opendrive_map": "sensor.pseudo.opendrive_map",
        "sensor.speedometer": "sensor.pseudo.speedometer"
    }

    def __init__(self, agent):
        super(ROSAgentWrapper, self).__init__(agent)

    def _preprocess_sensor_spec(self, sensor_spec):
        type_, id_, sensor_transform, attributes = super(ROSAgentWrapper, self)._preprocess_sensor_spec(sensor_spec)
        new_type = self.SENSOR_TYPE_REMAPS.get(type_, type_)
        return new_type, id_, sensor_transform, attributes

    def setup_sensors(self, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        for sensor_spec in self._agent.sensors():
            type_, id_, transform, attributes = self._preprocess_sensor_spec(sensor_spec)
            uid = self._agent.spawn_object(type_, id_, transform, attributes, attach_to=vehicle.id)
            self._sensors_list.append(uid)

        # Tick once to spawn the sensors
        CarlaDataProvider.get_world().tick()

    def cleanup(self):
        for uid in self._sensors_list:
            self._agent.destroy_object(uid)
        self._sensors_list.clear()

        # Tick once to destroy the sensors
        CarlaDataProvider.get_world().tick()
        self._sensors_list.clear()
        self.sensor_list_names.clear()
