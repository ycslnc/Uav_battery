import numpy as np

from src.DH.DeviceManager import DeviceManagerParams, DeviceManager
from src.DH.BatteryManger import BatteryParams, BatteryManager
import src.Map.Map as Map
from src.DH.State import DHState, DHScenario
from src.base.BaseGrid import BaseGrid, BaseGridParams


class DHGridParams(BaseGridParams):
    def __init__(self):
        super().__init__()
        self.device_manager = DeviceManagerParams()
        self.battery_manager = BatteryParams()


class DHGrid(BaseGrid):
    # BaseGrid 中有地图信息，起降点的坐标，地图shape
    def __init__(self, params: DHGridParams, stats):
        super().__init__(params, stats)
        self.params = params
        self.device_list = None
        self.device_manager = DeviceManager(self.params.device_manager)
        # NOTE 添加充电桩相关列表
        self.battery_list = None
        self.battery_manager = BatteryManager(self.params.battery_manager)

        free_space = np.logical_not(
            np.logical_or(self.map_image.obstacles, self.map_image.start_land_zone))
        # 有障碍物或者有起降区域的地方先并起来（True），即非空闲区，然后再取反（False），这样空闲区域就是True
        free_idcs = np.where(free_space)
        # 找到空闲区域对应的索引
        self.device_positions = list(zip(free_idcs[1], free_idcs[0]))
        # NOTE 添加充电桩相关信息
        battery_space = np.logical_not(np.logical_or(self.map_image.nfz, self.map_image.start_land_zone))
        battery_idcs = np.where(battery_space)
        self.battery_positions = list(zip(battery_idcs[1], battery_idcs[0]))

    def get_comm_obstacles(self):
        return self.map_image.obstacles

    def get_data_map(self):
        return self.device_list.get_data_map(self.shape)

    def get_collected_map(self):
        return self.device_list.get_collected_map(self.shape)

    def get_device_list(self):
        return self.device_list

    def get_grid_params(self):
        return self.params

    def init_episode(self):
        self.device_list = self.device_manager.generate_device_list(self.device_positions)
        # 返回设备数量（从参数中范围中随机选取，没有相同的）、设备位置（从空余的位置中随机选取）、各设备对应的数据
        # TODO 这里要添加一个函数，获得充电桩的数量，位置（这里和上面有区别，充电桩不能设置在禁飞区），
        #  剩余电量（理论上是无穷），可以再记录一下冲了多少电  充电桩的类
        #  NOTE 返回battery_list

        self.battery_list = self.battery_manager.generate_device_list(self.battery_positions)
        # print("battery_list.num_devices", self.battery_list.num_devices)
        state = DHState(self.map_image)
        # DHState 包含 地图信息、设备信息、智能体位置、移动步数预算（即电量）
        # 包含一堆操作，如判断地图区域、计算指标等
        state.reset_devices(self.device_list)
        # 重置设备数据图、已收集数据、设备列表
        # NOTE reset battery
        state.reset_batterys(self.battery_list)
        # Replace False 确保代理的起始位置不同
        # Replace False insures that starting positions of the agents are different

        idx = np.random.randint(len(self.starting_vector))
        # starting_vector  起降区域的坐标索引
        # 随机选择位置索引
        state.position = self.starting_vector[idx]
        # 随机选择agent位置
        state.movement_budget = np.random.randint(low=self.params.movement_range[0],
                                                  high=self.params.movement_range[1] + 1, size=1)
        # 步数范围 从最小步数到最大步数之间随机选取
        state.initial_movement_budget = state.movement_budget.copy()
        # 初始budget为 上述随机产生后的步数
        return state

    def create_scenario(self, scenario: DHScenario):
        state = DHState(self.map_image)
        # 起始飞机的位置
        state.position = self.starting_vector[scenario.position_idx]
        # 起始电量
        state.movement_budget = scenario.movement_budget
        # 起始电量为具体场景中电量
        state.initial_movement_budget = scenario.movement_budget
        # 获取每个LOT_device的位置
        positions = [self.device_positions[idx] for idx in scenario.device_idcs]
        state.reset_devices(self.device_manager.generate_device_list_from_args(len(positions), positions,
                                                                               scenario.device_data))

        # NOTE 添加battery相关信息
        battery_positions = [self.battery_positions[idx] for idx in scenario.battery_idcs]
        state.reset_batterys(self.battery_manager.generate_battery_list_from_args(len(battery_positions),
                                                                                  battery_positions))

        return state

    def init_scenario(self, state: DHState):
        self.device_list = state.device_list

        return state

    def get_example_state(self):
        # 获得shape
        state = DHState(self.map_image)
        state.device_map = np.zeros(self.shape, dtype=float) * 2
        state.collected = np.zeros(self.shape, dtype=float)
        state.battery_map = np.zeros(self.shape, dtype=float)
        return state
