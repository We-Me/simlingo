import copy
import logging
import numpy as np
import os
import time
from threading import Thread

from queue import Queue
from queue import Empty

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SensorConfigurationInvalid(Exception):
    """
    Exceptions thrown when the sensors used by the agent are not allowed for that specific submissions
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class GenericMeasurement(object):
    def __init__(self, data, frame):
        # 让 Python 伪传感器的数据具有与 CARLA Measurement 类似的接口
        # 因此伪传感器也能走统一的 CallBack 管线
        self.data = data
        self.frame = frame


class BaseReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle                         # 保存 ego vehicle，用来直接读取车辆状态
        self._reading_frequency = reading_frequency     # 例如 speedometer 为 20 Hz
        self._callback = None                           # 此时 CallBack 还没有通过 listen() 注册
        self._run_ps = True                             # 控制后台线程是否继续运行

        # 这里看起来像直接调用死循环，实际上 run() 被 @threaded 装饰，因此会启动新线程后立即返回
        self.run()

    def __call__(self):
        # 伪传感器数据生成方法
        pass

    @threaded
    def run(self):
        first_time = True                       # 用于强制第一帧立即输出
        latest_time = GameTime.get_time()       # 记录上一次发送数据的仿真时间
        while self._run_ps:                     # 只要没有清理传感器，线程就一直运行
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                # 距离上一次发送的仿真时间超过采样周期或这是第一条数据，并且正式仿真帧号不为 0
                if current_time - latest_time > (1 / self._reading_frequency) \
                        or (first_time and GameTime.get_frame() != 0):
                    # 这里的 self.__call__() 会根据真实子类动态分派
                    # SpeedometerReader → SpeedometerReader.__call__()
                    # OpenDriveMapReader → OpenDriveMapReader.__call__()
                    self._callback(GenericMeasurement(self.__call__(), GameTime.get_frame()))
                    # 更新状态
                    latest_time = GameTime.get_time()
                    first_time = False

                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class SpeedometerReader(BaseReader):
    """
    Sensor to measure the speed of the vehicle.
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        # 这是车辆在世界坐标系中的速度向量
        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        # 把车辆姿态从角度转成弧度
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        # 计算车辆朝前方向的单位向量
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        # 将世界坐标速度投影到车辆朝前方向
        speed = np.dot(vel_np, orientation)
        return speed

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        # 最多尝试 10 次，每次失败等待 0.2 秒
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                velocity = self._vehicle.get_velocity()
                transform = self._vehicle.get_transform()
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        # 正常情况下最终返回 {"speed": forward_speed}
        return {'speed': self._get_forward_speed(transform=transform, velocity=velocity)}


class OpenDriveMapReader(BaseReader):
    def __call__(self):
        # 获取当前 CARLA 地图，并转换成 OpenDRIVE XML 字符串
        return {'opendrive': CarlaDataProvider.get_map().to_opendrive()}


"""
外部调用位于 AgentWrapperLocal.setup_sensors()
sensor.listen(
        CallBack(id_, type_, sensor, self._agent.sensor_interface)
)
python 会先执行 CallBack(...) 得到可调用对象
再执行 sensor.listen(callback_object)
"""


class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        # CallBack 在创建时立即注册传感器
        self._data_provider.register_sensor(tag, sensor_type, sensor)

    def __call__(self, data):
        """
        Image               →   _parse_image_cb
        LidarMeasurement    →   _parse_lidar_cb
        RadarMeasurement    →   _parse_radar_cb
        GnssMeasurement     →   _parse_gnss_cb
        IMUMeasurement      →   _parse_imu_cb
        GenericMeasurement  →   _parse_pseudosensor
        """
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        # 保存四个通道为 BGRA
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        # 每个 LiDAR 点有四个值 x, y, z, intensity
        # 最终形状 (number_of_points, 4)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        # 与 LiDAR 类似，每个目标四个浮点数
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        # 把每一行的四列逆序，转换成下游代码期望的排列方式
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        # [latitude, longitude, altitude]
        # (3,)
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)


"""
Evaluator 创建 DataAgent
└── AutonomousAgentLocal.__init__()
    └── SensorInterface.__init__()

ScenarioManager.load_scenario()
└── AgentWrapperLocal.setup_sensors()
    ├── 创建 CARLA 真实传感器
    ├── 创建 SpeedometerReader / OpenDriveMapReader
    └── sensor.listen(CallBack(...))
        └── CallBack.__init__()
            └── SensorInterface.register_sensor()

每次 world.tick()
├── CARLA 触发真实传感器回调
│   └── CallBack.__call__()
│       └── _parse_xxx_cb()
│           └── SensorInterface.update_sensor()
│
└── AgentWrapperLocal.__call__()
    └── AutonomousAgentLocal.__call__()
        └── SensorInterface.get_data(current_frame)
            └── DataAgent.run_step(input_data, ...)
"""


class SensorInterface(object):
    def __init__(self):
        # 保存已经注册的传感器，sensor id/tag → sensor object
        self._sensors_objects = {}
        # 创建线程安全队列
        # 生产者是 CARLA 传感器回调线程和伪传感器的 BaseReader 后台线程
        # 消费者是 Agent 主线程中的 get_data()
        # 格式为 (tag, frame, data)
        self._data_buffers = Queue()
        # 每次阻塞读取队列时，最多等 10 秒
        self._queue_timeout = 10

        # Only sensor that doesn't get the data on tick, needs special treatment
        # 记录 OpenDRIVE 地图传感器的 tag
        # OpenDRIVE 地图基本只发送一次，因此后续帧不能一直等待它
        self._opendrive_tag = None

    def register_sensor(self, tag, sensor_type, sensor):
        if tag in self._sensors_objects:
            # 重复 tag 检查
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(tag))

        # 保存传感器
        self._sensors_objects[tag] = sensor

        # 特殊记录 OpenDRIVE
        if sensor_type == 'sensor.opendrive_map': 
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, frame):
        if tag not in self._sensors_objects:
            raise SensorConfigurationInvalid("The sensor with tag [{}] has not been created!".format(tag))
        # 向缓存队列写入新数据
        self._data_buffers.put((tag, frame, data))

    def get_data(self, frame):
        """Read the queue to get the sensors data"""
        try:
            # 创建一个新的当前帧数据字典
            data_dict = {}
            # 持续读取队列，直到当前帧所有传感器都到齐
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):
                # Don't wait for the opendrive sensor
                # 条件表示：
                #   存在 OpenDRIVE 传感器
                #   当前帧还没有 OpenDRIVE 数据
                #   除 OpenDRIVE 外，其他传感器已经全部到齐
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    break

                sensor_data = self._data_buffers.get(True, self._queue_timeout)
                if sensor_data[1] != frame:
                    # 过滤非当前帧
                    continue
                # 写入当前帧字典
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send their data")

        return data_dict
