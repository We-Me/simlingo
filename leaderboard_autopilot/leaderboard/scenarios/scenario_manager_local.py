#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla
import threading

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper_local import AgentWrapperFactory, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """
    """
    ScenarioManager(...)
    ├── __init__()
    ├── load_scenario(...)
    │   └── AgentWrapper.setup_sensors(...)
    ├── run_scenario()
    │   ├── 启动 build_scenarios_loop() 后台线程
    │   └── while _running
    │       └── _tick_scenario()
    │           ├── world.tick()
    │           ├── 更新时间和 actor 状态
    │           ├── 调用 Agent
    │           ├── apply_control()
    │           └── scenario_tree.tick_once()
    ├── stop_scenario()
    │   ├── compute_duration_time()
    │   └── analyze_scenario()
    └── cleanup()
    """

    def __init__(self, timeout, statistics_manager, debug_mode=0):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.route_index = None                         # 当前路线编号
        self.scenario = None                            # 当前 RouteScenario
        self.scenario_tree = None                       # 路线行为树
        self.ego_vehicles = None                        # 自车列表
        self.other_actors = None                        # 场景车辆、行人等 actor

        self._debug_mode = debug_mode                   # 调试输出级别
        self._agent_wrapper = None                      # 还没有包装 Agent。load_scenario() 时创建
        self._running = False                           # 当前没有运行路线。run_scenario() 会改为 True
        self._timestamp_last_run = 0.0                  # 保存最后一次已处理的 CARLA 仿真时间，避免同一帧重复调用 Agent
        self._timeout = float(timeout)                  # CARLA 或 Agent 单步允许的最大等待时间

        # 这里同时维护两套时间
        # 二者可能不同。例如 CARLA 以 0.2 倍实时速度运行
        # 游戏时间：100 秒  现实时间：500 秒
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        # 初始化 watchdog
        self._watchdog = None                           # 监控 CARLA 仿真更新
        self._agent_watchdog = None                     # 监控 Agent 单帧执行

        self._statistics_manager = statistics_manager   # 路线统计管理器，仅在 debug 实时统计和路线结果分析时使用

        # Use the callback_id inside the signal handler to allow external interrupts
        # 注册信号处理函数
        # 外部代码之后可能重新注册自己的 SIGINT handler，再把信号转发给这里
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Agent took longer than {}s to send its command".format(self._timeout))
        elif self._watchdog and not self._watchdog.get_status():
            raise RuntimeError("The simulation took longer than {}s to update".format(self._timeout))
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._spectator = None
        self._watchdog = None
        self._agent_watchdog = None

    def load_scenario(
        self, 
        scenario,           # 已经创建好的 RouteScenario
        agent,              # 已经创建并完成 setup() 的 Agent
        route_index,        # 当前路线索引
        rep_number          # 当前重复执行编号
    ):
        """
        Load a new scenario
        """

        # 清零 Leaderboard 内部维护的游戏时间、帧号、上一次时间戳、wall-clock 起点
        # 传感器初始化期间的预热 tick 不会直接作为正式采集步骤处理
        GameTime.restart()
        # 根据 Agent 类型选择 wrapper
        self._agent_wrapper = AgentWrapperFactory.get_wrapper(agent)
        # 保存路线索引
        self.route_index = route_index
        # 保存 RouteScenario
        self.scenario = scenario
        # 取得行为树
        self.scenario_tree = scenario.scenario_tree
        # 保存自车
        self.ego_vehicles = scenario.ego_vehicles
        # 保存其他 actor
        self.other_actors = scenario.other_actors
        # 保存重复编号
        self.repetition_number = rep_number

        # 获取 CARLA 观察者
        # 它不属于 Agent 传感器，只控制 CARLA Viewer 的观察视角
        self._spectator = CarlaDataProvider.get_world().get_spectator()

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        # 创建并挂载传感器
        # 该调用会根据 Agent 声明的传感器配置：
        # 创建 CARLA sensor actor；
        # 将传感器附着到第一辆 ego vehicle；
        # 注册数据回调；
        # 把数据送入 Agent 的 SensorInterface；
        # 执行若干预热 tick。
        self._agent_wrapper.setup_sensors(self.ego_vehicles[0])

    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        # 只要主路线还在运行，后台线程就持续工作
        while self._running:
            # 创建临近动态场景，根据 ego vehicle 当前的位置，尝试创建已经接近自车的路线场景
            self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)
            # 根据自车当前位置生成临近停车车辆
            self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
            # 这里是现实时间的一秒，不是 CARLA 游戏时间的一秒
            time.sleep(1)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()            # 记录开始现实时间
        self.start_game_time = GameTime.get_time()      # 记录开始游戏时间

        # Detects if the simulation is down
        # 创建 CARLA watchdog，用于检测 CARLA 仿真是否长时间无法完成更新
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        # Stop the agent from freezing the simulation
        # 创建 Agent watchdog，用于检测 self._agent_wrapper() 是否长时间不返回
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()

        # 设置运行状态，之后主逐帧循环开始，后台场景构建线程也开始工作。
        self._running = True

        # Thread for build_scenarios
        # 创建场景构建线程
        t = threading.Thread(target=self.build_scenarios_loop, args=(self._debug_mode > 0, ))
        t.start()

        # 从这里开始有两个并行执行流
        # 主线程：不断执行 _tick_scenario()
        # 后台线程：不断执行 build_scenarios_loop()
        while self._running:
            self._tick_scenario()

    def _tick_scenario(self):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        # 只有同时满足以下条件才推进世界
        # 路线仍在运行以及 CARLA watchdog 状态正常
        if self._running and self.get_running_status():
            # 当前调用阻塞等待 CARLA 完成一帧，最长等待 _timeout 秒
            CarlaDataProvider.get_world().tick(self._timeout)

        # 取得刚刚完成的仿真帧信息
        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        # 当前仿真时间比上次已处理时间新以及路线仍然运行
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            # 记录已处理时间
            self._timestamp_last_run = timestamp.elapsed_seconds

            # 更新 CARLA watchdog，重新开始计算超时时间
            self._watchdog.update()

            # Update game time and actor information
            # 把当前 CARLA 时间戳同步到 Leaderboard 的全局游戏时钟
            GameTime.on_carla_tick(timestamp)
            # 更新 provider 缓存的 actor 信息
            CarlaDataProvider.on_carla_tick()

            # 暂停 CARLA watchdog
            # 下一阶段开始执行 Agent
            # 此时不应该继续计算 CARLA Server 超时
            # 否则 Agent 计算时间可能被错误判断为 CARLA 卡死。
            self._watchdog.pause()

            try:
                # 开始监控 Agent 单帧执行时间
                self._agent_watchdog.resume()
                # 把 Agent watchdog 的超时起点更新到当前时刻
                self._agent_watchdog.update()
                # 调用 Agent
                ego_action = self._agent_wrapper()
                # 暂停 Agent watchdog，Agent 已正常返回，不再计算 Agent 超时
                self._agent_watchdog.pause()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            # 恢复 CARLA watchdog，Agent 已执行完成，后续又回到仿真和场景处理阶段
            self._watchdog.resume()

            # 将 Agent 返回的控制交给 ego vehicle
            # 一般来说，该控制主要在下一次 world.tick() 中影响车辆物理状态
            self.ego_vehicles[0].apply_control(ego_action)
            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            # 控制写入行为树黑板
            # 场景 behavior 或 criteria 可以从 blackboard 读取当前 Agent 控制，overwrite=True 表示每帧覆盖旧控制
            py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
            # 推进行为树
            # 这一帧会推进：
            #   动态场景行为
            #   路线完成条件
            #   碰撞等 criteria
            #   场景成功或失败状态
            self.scenario_tree.tick_once()

            if self._debug_mode > 1:
                # 动态计算当前已经运行多久
                self.compute_duration_time()

                # Update live statistics
                # 生成当前帧的临时路线统计
                self._statistics_manager.compute_route_statistics(
                    self.route_index,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message=""
                )
                # 写入实时状态
                self._statistics_manager.write_live_results(
                    self.route_index,                                   # route index
                    self.ego_vehicles[0].get_velocity().length(),       # 自车速度
                    ego_action,                                         # 当前控制
                    self.ego_vehicles[0].get_location()                 # 自车位置
                )

            if self._debug_mode > 2:
                print("\n")
                # 打印当前行为树及所有节点状态
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                # 立即刷新输出，避免日志缓冲导致行为树信息延迟显示
                sys.stdout.flush()

            # 判断行为树是否结束
            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            # 获取自车位姿  location + rotation
            ego_trans = self.ego_vehicles[0].get_transform()

            # TODO: here we can change the spectator
            # self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=70),
                                                        #   carla.Rotation(pitch=-90)))
            
            # For third-person view
            # location = ego_trans.transform(carla.Location(x=-4.5, z=2.3))
            # self._spectator.set_transform(carla.Transform(location, carla.Rotation(pitch=-15.0, yaw=ego_trans.rotation.yaw)))
            
            # For bird's eye view
            # 设置鸟瞰 spectator，把 spectator 放到自车上方 30 米，并垂直向下看
            self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=30), carla.Rotation(pitch=-90)))

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        # 用于查询 CARLA 仿真 watchdog 是否正常
        if self._watchdog:
            return self._watchdog.get_status()
        return True

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        # 停止 CARLA watchdog
        if self._watchdog:
            self._watchdog.stop()

        # 停止 Agent watchdog
        if self._agent_watchdog:
            self._agent_watchdog.stop()

        # 计算最终运行时间
        self.compute_duration_time()

        # 只在 watchdog 正常时执行正式终止
        if self.get_running_status():
            if self.scenario is not None:
                # 终止场景
                self.scenario.terminate()

            if self._agent_wrapper is not None:
                # 清理 AgentWrapper
                self._agent_wrapper.cleanup()
                self._agent_wrapper = None

            # 分析场景结果
            self.analyze_scenario()

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        # 创建结果输出对象，并把当前 manager 传进去
        # 该对象可以读取当前信息从而输出当前路线的最终结果
        ResultOutputProvider(self)
