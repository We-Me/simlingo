#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import importlib
import os
import sys
import signal
import socket

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager_local import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper_local import AgentError, validate_sensor_configuration
from leaderboard.utils.statistics_manager_local import StatisticsManager, FAILURE_MESSAGES
from leaderboard.utils.route_indexer import RouteIndexer

import pathlib


sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer',
    'sensor.camera.semantic_segmentation': 'carla_camera', # for datagen
    'sensor.camera.depth':      'carla_camera', # for datagen
}

class LeaderboardEvaluator(object):
    """
    Main class of the Leaderboard. Everything is handled from here,
    from parsing the given files, to preparing the simulation, to running the route.
    """

    # Tunable parameters
    # CARLA Client 默认超时
    client_timeout = 10.0  # in seconds
    # 仿真频率 20 Hz
    # 固定仿真步长为 0.05s = 50ms
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.world = None                               # 当前 CARLA World，此时还没加载具体 Town
        self.manager = None                             # 后面保存 ScenarioManager
        self.sensors = None                             # 保存 Agent 声明的传感器配置列表
        self.sensors_initialized = False                # 记录传感器配置是否成功完成初始化与校验
        self.sensor_icons = []                          # 保存传感器对应的 Leaderboard 图标名称
        self.agent_instance = None                      # 后面保存动态创建的 DataAgent 或其他 Agent 对象
        self.route_scenario = None                      # 后面保存当前正在执行的路线场景

        self.statistics_manager = statistics_manager    # 保存统计管理器

        # This is the ROS1 bridge server instance. This is not encapsulated inside the ROS1 agent because the same
        # instance is used on all the routes (i.e., the server is not restarted between routes). This is done
        # to avoid reconnection issues between the server and the roslibpy client.
        self._ros1_server = None                        # 这里用不到

        # Setup the simulation
        # 初始化 CARLA 仿真
        self.client, \
        self.client_timeout, \
        self.traffic_manager, \
        self.traffic_manager_port \
            = self._setup_simulation(args)
        # 至此，Evaluator 已经持有 CARLA Client 和 Traffic Manager

        # Load agent
        # 提取 Agent 模块名
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        # 动态导入 Agent 模块
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        # 创建 ScenarioManager
        self.manager = ScenarioManager(args.timeout, self.statistics_manager, args.debug)

        # Time control for summary purposes
        # 记录时间状态
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Prepare the agent timer
        # 初始化 Agent watchdog
        self._agent_watchdog = None
        # 注册 SIGINT 信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)

        # 初始化客户端超时状态
        self._client_timed_out = False

        """
        CARLA Client                已创建并设置超时
        CARLA World                 尚未加载具体路线 Town
        Traffic Manager             已创建并启用同步/混合物理
        Agent Python 模块           已导入
        Agent 实例                  尚未创建
        ScenarioManager             已创建
        RouteScenario               尚未创建
        传感器                      尚未创建
        SIGINT handler              已注册
        """
        pass

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt.
        Either the agent initialization watchdog is triggered, or the runtime one at scenario manager
        它处理 SIGINT，主要用于 Ctrl+C 和 watchdog 中断
        """
        # 检查 Agent 初始化是否超时
        # Agent 初始化 watchdog 存在并且 watchdog 状态为失败
        # 说明 Agent 初始化或 setup() 超时
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            # 抛出异常，终止 Agent 初始化流程
            raise RuntimeError("Timeout: Agent took longer than {}s to setup".format(self.client_timeout))
        elif self.manager:
            # 如果不是 Agent 初始化超时，并且 ScenarioManager 已创建，就把信号转交给 Manager
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        这是 Python 的析构方法。对象准备被回收时，Python可能自动调用它
        __del__() 更像最后的引用释放兜底，不能替代正常清理流程
        """
        if hasattr(self, 'manager') and self.manager:
            # 删除 ScenarioManager 引用
            del self.manager
        if hasattr(self, 'world') and self.world:
            # 删除 World 引用
            del self.world

    def _get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        木有调用，感觉没有
        """
        if self._agent_watchdog:
            return self._agent_watchdog.get_status()
        return False

    def _cleanup(self, results=None):
    # def _cleanup(self):
        """
        Remove and destroy all actors
        清理当前路线留下的 Agent、场景 Actor、Manager 状态和传感器
        """
        # 清理 CarlaDataProvider 管理的 actor 和内部缓存状态
        CarlaDataProvider.cleanup()

        # 如果 Agent 初始化过程中发生异常，watchdog 可能还在运行，因此清理时确保将它停止
        if self._agent_watchdog:
            self._agent_watchdog.stop()

        try:
            # 销毁 Agent
            if self.agent_instance:
                self.agent_instance.destroy(results)
                del self.agent_instance
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

        if self.route_scenario:
            # 如果当前路线场景已经创建，删除它管理的车辆、行人和其他场景 Actor
            self.route_scenario.remove_all_actors()
            # 清除当前路线场景引用，避免下一条路线误用旧场景
            self.route_scenario = None
            # 通知统计管理器当前场景已经结束，不再保留对 RouteScenario 的引用
            if self.statistics_manager:
                self.statistics_manager.remove_scenario()

        if self.manager:
            # 读取 ScenarioManager 的运行状态
            # 如果 Manager 的 watchdog 状态不正常
            # 则 self._client_timed_out = True
            # 这个标志之后会影响 _reset_world_settings()
            self._client_timed_out = not self.manager.get_running_status()
            # 重置 ScenarioManager
            self.manager.cleanup()

        # Make sure no sensors are left streaming
        # 从当前 CARLA World 中找出所有类型名包含 sensor 的 Actor
        alive_sensors = self.world.get_actors().filter('*sensor*')
        # 对每个残留 Sensor Actor
        for sensor in alive_sensors:
            # 停止数据回调和数据流
            sensor.stop()
            # 从 CARLA World 中销毁该传感器
            sensor.destroy()

    def find_free_port(self, start_port=2_000, end_port=40_000):
        # 遍历候选端口
        for port in range(start_port, end_port + 1):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       # 创建临时 socket
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)     # 设置地址复用

            try:
                s.bind(('localhost', port))                             # 尝试绑定端口
                return port
            except OSError as e: # Address already in use or Address already in use (WinError)
                pass
            finally:
                s.close()

        return None

    def _setup_simulation(self, args):
        """
        Prepares the simulation by getting the client, and setting up the world and traffic manager settings
        """
        # 创建 CARLA Client
        client = carla.Client(args.host, args.port)
        if args.timeout:
            client_timeout = args.timeout
        # 设置 Client 超时
        client.set_timeout(client_timeout)
        # 创建世界配置
        settings = carla.WorldSettings(
            synchronous_mode = True,                        # 开启同步模式，等待客户端显式调用 world.tick()
            fixed_delta_seconds = 1.0 / self.frame_rate,    # 每次 world.tick()，游戏时间固定增加 0.05 秒
            deterministic_ragdolls = True,                  # 使行人等对象的 ragdoll 物理尽可能确定，减少重复运行时的随机差异
            spectator_as_ego = False                        # 不把 spectator 当成需要参与大地图流式加载的 ego actor
        )
        client.get_world().apply_settings(settings)         # 应用世界配置

        traffic_manager_port = self.find_free_port()                        # 查找 Traffic Manager 端口
        traffic_manager = client.get_trafficmanager(traffic_manager_port)   # 启动对应端口的 CARLA Traffic Manager
        traffic_manager.set_synchronous_mode(True)                          # # 开启同步模式
        traffic_manager.set_hybrid_physics_mode(True)                       # 开启混合物理模式，降低大量背景车辆带来的仿真开销

        return client, client_timeout, traffic_manager, traffic_manager_port

    def _reset_world_settings(self):
        """
        Changes the modified world settings back to asynchronous
        """
        # Has simulation failed?
        # 判断能否安全恢复
        # self.world 已成功加载，self.manager 存在，Client/ScenarioManager 没有发生超时
        if self.world and self.manager and not self._client_timed_out:
            # Reset to asynchronous mode
            self.world.tick()  # TODO: Make sure all scenario actors have been destroyed
            settings = self.world.get_settings()        # 取得当前 Town 的世界设置对象
            settings.synchronous_mode = False           # 关闭同步模式
            settings.fixed_delta_seconds = None         # 移除固定时间步
            settings.deterministic_ragdolls = False     # 恢复 ragdoll 设置
            settings.spectator_as_ego = True            # 恢复 spectator 行为
            self.world.apply_settings(settings)         # 应用恢复后的配置

            # Make the TM back to async
            self.traffic_manager.set_synchronous_mode(False)        # 关闭同步模式
            self.traffic_manager.set_hybrid_physics_mode(False)     # 关闭混合物理模式

    def _load_and_wait_for_world(self, args, town):
        """
        Load a new CARLA world without changing the settings and provide data to CarlaDataProvider
        """

        # 通知 CARLA Server 加载指定 Town，并把返回的 World 保存到 self.world
        self.world = self.client.load_world(town, reset_settings=False)

        # Large Map settings are always reset, for some reason
        # 注释说明，大地图相关设置在 load_world() 后可能被 CARLA 自动重置，所以这里重新配置
        # 读取刚加载地图的运行设置
        settings = self.world.get_settings()
        # 设置地图 tile 的流式加载距离为 650 米
        settings.tile_stream_distance = 650
        # 设置 actor 保持活跃的距离为 650 米
        settings.actor_active_distance = 650
        # 把修改后的大地图配置提交给 CARLA Server
        self.world.apply_settings(settings)

        # 将地图中的所有交通灯恢复到初始状态
        self.world.reset_all_traffic_lights()
        # 把当前 CARLA Client 注册到全局数据提供器
        CarlaDataProvider.set_client(self.client)
        # 告诉 CarlaDataProvider 当前实际使用的 Traffic Manager 端口
        CarlaDataProvider.set_traffic_manager_port(self.traffic_manager_port)
        # 把刚加载的 CARLA World 注册到全局数据提供器
        CarlaDataProvider.set_world(self.world)
        # 设置 CarlaDataProvider 使用的随机种子
        CarlaDataProvider.set_random_seed(args.traffic_manager_seed)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        # 设置 Traffic Manager 的随机种子
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)

        # Wait for the world to be ready
        # 由于 World 处于同步模式，必须主动调用 tick() 才会推进
        self.world.tick()

        # 获取实际地图名称
        # CARLA 地图名称可能是完整路径，例如：/Game/Carla/Maps/Town12
        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        # 比较 CARLA 实际加载的地图 map_name 和当前路线要求的地图 town
        if map_name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            " This scenario requires the use of map {}".format(town))

    def _register_statistics(
        self, 
        route_date_string,  # 当前路线运行标识，例如 22_0_route0_07_22_15_30_08
        route_index,        # 当前路线在索引器中的编号
        entry_status,       # 当前运行状态，例如 "Started" 或失败状态
        crash_message=""    # 失败原因，默认空字符串
    ):
        """
        Computes and saves the route statistics
        """
        print("\033[1m> Registering the route statistics\033[0m")
        # 把当前 entry_status 保存到统计管理器
        self.statistics_manager.save_entry_status(entry_status)
        current_stats_record = self.statistics_manager.compute_route_statistics(
            route_date_string,
            route_index,
            self.manager.scenario_duration_system,  # 路线实际消耗的现实时间
            self.manager.scenario_duration_game,    # CARLA 仿真世界中经过的游戏时间
            crash_message
        )
        # 把当前路线统计对象返回给调用方
        return current_stats_record

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""              # crash_message 为空表示暂时没有崩溃
        entry_status = "Started"        # entry_status 表示本次评测已经开始

        # 打印路线信息
        # 路线名称 config.name  重复编号 config.repetition_index
        print(
            "\n\033[1m========= Preparing {} (repetition {}) =========\033[0m"
            .format(config.name, config.repetition_index)
        )

        # Prepare the statistics of the route
        # 创建路线统计记录
        route_name = f"{config.name}_rep{config.repetition_index}"
        # 提前创建路线统计记录
        # 这样即使后面加载地图或 Agent 时崩溃，checkpoint 中也能保留这条路线的失败记录
        self.statistics_manager.create_route_data(route_name, config.index)

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            # 加载当前路线要求的 Town，并等待 CARLA World 就绪
            self._load_and_wait_for_world(args, config.town)
            # 根据当前路线配置创建 RouteScenario
            self.route_scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            # 把当前场景交给统计管理器，方便之后读取路线长度、完成率和违规事件
            self.statistics_manager.set_scenario(self.route_scenario)

        except Exception:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print(f"\n{traceback.format_exc()}\033[0m")
            # 从失败信息表中取得仿真崩溃对应的路线状态和崩溃信息
            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            # 尝试登记失败结果
            self._register_statistics(config.index, entry_status, crash_message)
            # 清理当前场景资源，并告诉上层发生了致命错误，终止后续路线
            self._cleanup()
            return True

        print("\033[1m> Setting up the agent\033[0m")

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            now = datetime.now()
            # route_string = pathlib.Path(os.environ.get('ROUTES', '')).stem + '_'
            # 取得 route XML 文件名，但去掉路径和扩展名
            route_string = pathlib.Path(args.routes).stem + '_'
            # 追加 XML 内部路线索引
            route_string += f'route{config.index}'
            # 追加当前时间，22_0_route0_07_22_15_30_08
            route_date_string = route_string + '_' + '_'.join(
                map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second))
            )

            # 初始化 watchdog
            # 如果 Agent 的导入、创建或 setup() 长时间卡住，watchdog 可以中断运行
            self._agent_watchdog = Watchdog(args.timeout)
            self._agent_watchdog.start()
            # 从之前导入的 Agent 模块中取得 get_entry_point() 函数并调用
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            # 再根据类名从模块中取得实际的 Agent 类对象
            agent_class_obj = getattr(self.module_agent, agent_class_name)

            # Start the ROS1 bridge server only for ROS1 based agents.
            # ROS1 对当前来说无用，不用管
            if getattr(agent_class_obj, 'get_ros_version')() == 1 and self._ros1_server is None:
                from leaderboard.autoagents.ros1_agent import ROS1Server
                self._ros1_server = ROS1Server()
                self._ros1_server.start()

            # self.agent_instance = agent_class_obj(args.host, args.port, args.debug)
            # 创建 Agent 实例
            if int(os.environ.get('DATAGEN', 0))==1:
                # 数据采集模式，第二个构造参数为当前路线索引
                self.agent_instance = agent_class_obj(args.agent_config, config.index)
            else:
                # 非采集模式下，第二个构造参数改为路线时间标识
                self.agent_instance = agent_class_obj(args.agent_config, route_date_string)

            # 设置全局路线，gps_route：GPS 坐标路线，route：CARLA 世界坐标路线
            self.agent_instance.set_global_plan(self.route_scenario.gps_route, self.route_scenario.route)
            # 调用 Agent.setup()
            self.agent_instance.setup(args.agent_config, route_date_string, self.traffic_manager)

            # Check and store the sensors
            # 获取并校验传感器配置，只在第一次路线运行时执行传感器配置验证
            # 后续路线会复用第一次保存的配置，不再重复验证
            if not self.sensors:
                # 取得 Agent 请求的传感器配置，例如摄像头、LiDAR、IMU 和地图传感器
                self.sensors = self.agent_instance.sensors()
                # 取得 Agent 请求的传感器配置，例如摄像头、LiDAR、IMU 和地图传感器
                track = self.agent_instance.track
                # 检查：
                # Agent track 是否和命令行 --track 一致。
                # 传感器类型是否合法。
                # 数量是否超限。
                # ID 是否重复。
                # 安装位置是否符合要求。
                validate_sensor_configuration(self.sensors, track, args.track)

                # 保存传感器展示信息
                # 将传感器类型转换为统计结果使用的图标名称
                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                # 把传感器信息写入 checkpoint
                self.statistics_manager.save_sensors(self.sensor_icons)
                self.statistics_manager.write_statistics()

                # 标记传感器配置已成功初始化和验证
                self.sensors_initialized = True

            # Agent 初始化成功，不再需要初始化阶段的 watchdog
            self._agent_watchdog.stop()
            self._agent_watchdog = None

        # 专门捕获传感器配置错误
        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print(f"{e}\033[0m\n")

            # 设置为传感器拒绝状态
            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            # 登记当前路线失败结果
            result = self._register_statistics(route_date_string, config.index, entry_status, crash_message)
            # 清理资源并返回 True
            self._cleanup(result)
            return True

        # 其他 Agent 初始化错误
        except Exception:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

            # 标记为 Agent 初始化失败
            entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
            result = self._register_statistics(route_date_string, config.index, entry_status, crash_message)
            self._cleanup(result)
            # 这意味着 Agent 初始化失败不会被视为致命 Simulation crash，上层可以继续下一条路线
            return False

        print("\033[1m> Running the route\033[0m")

        # 开始运行路线
        # Run the scenario
        try:
            # Load scenario and run it
            # 如果指定 --record，启动 CARLA 原生 recorder
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            # 将场景交给 ScenarioManager
            self.manager.load_scenario(self.route_scenario, self.agent_instance, config.index, config.repetition_index)
            # 正式执行逐帧仿真
            self.manager.run_scenario()

        # 专门捕获 Agent 单步执行错误
        except AgentError:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print(f"\n{traceback.format_exc()}\033[0m")
            # 设置为 Agent 运行时崩溃状态
            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]
            # 这里不立即返回，仍会进入后面的停止、统计和清理流程

        # 捕获其他仿真异常
        except Exception:
            print("\n\033[91mError during the simulation:")
            print(f"\n{traceback.format_exc()}\033[0m")
            # 标记为 Simulation crash
            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            # 同样不会立即返回，而是继续尝试停止和清理场景

        # 结束运行路线
        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            # 停止场景运行并终止相关行为
            self.manager.stop_scenario()
            # 计算并保存当前路线统计，返回当前路线的统计记录
            result = self._register_statistics(route_date_string, config.index, entry_status, crash_message)
            # 计算并保存当前路线统计，返回当前路线的统计记录
            if args.record:
                self.client.stop_recorder()
            # 把路线统计结果传给清理方法
            self._cleanup(result)

        # 收尾阶段失败
        except Exception:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print(f"\n{traceback.format_exc()}\033[0m")
            # 将 crash_message 改为 Simulation crash 信息
            _, crash_message = FAILURE_MESSAGES["Simulation"]

        # If the simulation crashed, stop the leaderboard, for the rest, move to the next route
        # 只有 crash_message 精确等于 Simulation crashed 才返回 True 否则继续下一条路线
        return crash_message == "Simulation crashed"

    def run(self, args):
        """
        Run the challenge mode
        """
        # 创建路线索引器
        # 当前只把它理解为一个路线迭代器，提供：
        #   总共有多少条待运行路线。
        #   当前运行到哪里。
        #   下一条路线的配置。
        #   checkpoint 续跑位置。
        route_indexer = RouteIndexer(args.routes, args.repetitions, args.routes_subset)

        if args.resume: # 用户请求续跑
            # checkpoint 是否验证成功、实际上能否续跑
            resume = route_indexer.validate_and_resume(args.checkpoint)
        else:
            resume = False

        # 初始化统计记录
        if resume:
            self.statistics_manager.add_file_records(args.checkpoint)
        else:
            self.statistics_manager.clear_records()
        # 写入初始进度
        self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
        # 立即把当前统计状态写入 checkpoint
        self.statistics_manager.write_statistics()

        crashed = False # 初始化崩溃标志
        while route_indexer.peek() and not crashed:
            # Run the scenario
            # 获取下一条路线配置
            config = route_indexer.get_next_config()
            # 执行当前路线
            crashed = self._load_and_run_scenario(args, config)

            # Save the progress and write the route statistics
            # 每条路线结束后保存进度
            self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
            # 将最新进度和路线结果写入 checkpoin
            self.statistics_manager.write_statistics()

        # Shutdown ROS1 bridge server if necessary
        # 关闭 ROS1 bridge，这里用不到
        if self._ros1_server is not None:
            self._ros1_server.shutdown()

        # Go back to asynchronous mode
        # 恢复 CARLA World 设置，更像是清理init中应用的设置
        self._reset_world_settings()

        # 只有没有发生致命 Simulation crash 时，才计算完整的全局统计
        if not crashed:
            # Save global statistics
            print("\033[1m> Registering the global statistics\033[0m")
            # 根据所有路线记录计算总体结果
            self.statistics_manager.compute_global_statistics()
            # 验证并写入最终统计
            self.statistics_manager.validate_and_write_statistics(self.sensors_initialized, crashed)

        return crashed

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--traffic-manager-port', default=8000, type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--traffic-manager-seed', default=100, type=int,
                        help='Seed used by the TrafficManager (default: 100)')
    parser.add_argument('--debug', type=int,
                        help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default=300.0, type=float,
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes', required=True,
                        help='Name of the routes file to be executed.')
    parser.add_argument('--routes-subset', default='', type=str,
                        help='Execute a specific set of routes')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str,
                        help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str,
                        help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS',
                        help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=int, default=False,
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--debug-checkpoint", type=str, default='./live_results.txt',
                        help="Path to checkpoint used for saving live results")

    arguments = parser.parse_args()

    # 创建当前路线的结果和进度文件路径
    pathlib.Path(arguments.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    # 创建统计管理器
    statistics_manager = StatisticsManager(arguments.checkpoint, arguments.debug_checkpoint)
    # 创建 Evaluator
    leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
    # 开始运行路线
    # 回值 crashed 是布尔值：
    # False：没有发生会中止整个 evaluator 的仿真崩溃。
    # True：发生了被判断为 "Simulation crashed" 的错误
    crashed = leaderboard_evaluator.run(arguments)

    del leaderboard_evaluator

    if crashed:
        sys.exit(-1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    """
    collect_dataset_slurm.py
    └── 启动 leaderboard_evaluator_local.py
        └── main()
            ├── StatisticsManager(...)
            ├── LeaderboardEvaluator.__init__()
            │   ├── _setup_simulation()
            │   ├── 动态导入 DataAgent
            │   └── ScenarioManager(...)
            └── LeaderboardEvaluator.run()
                └── 遍历 RouteIndexer
                    └── _load_and_run_scenario()
                        ├── _load_and_wait_for_world()
                        ├── RouteScenario(...)
                        ├── DataAgent(...)
                        ├── DataAgent.setup()
                        ├── ScenarioManager.load_scenario()
                        ├── ScenarioManager.run_scenario()
                        │   └── 循环 _tick_scenario()
                        ├── ScenarioManager.stop_scenario()
                        ├── _register_statistics()
                        └── _cleanup()
    """
    main()
