import numpy as np
from src.Map.Map import Map
from src.StateUtils import pad_centered
from src.base.BaseState import BaseState


class DHScenario:
    def __init__(self):
        self.device_idcs = []
        self.device_data = []
        self.position_idx = 0
        self.battery_idcs = []
        self.movement_budget = 100


class DHState(BaseState):
    # BaseState 有禁飞区、障碍物、起降区的信息
    def __init__(self, map_init: Map):
        super().__init__(map_init)
        self.device_list = None
        self.device_map = None
        # NOTE 添加充电桩相关信息
        self.battery_list = None
        self.battery_map = None
        self.initial_total_charge_time = 0
        # Floating point sparse matrix showing devices and their data to be collected
        # 浮点稀疏矩阵显示设备及其要收集的数据
        self.position = [0, 0]
        # agent位置
        self.movement_budget = 0
        # 移动步数（电量）
        self.landed = False
        # 降落
        self.terminal = False
        self.device_com = -1
        # TODO
        #  设置充电桩的初始状态
        # NOTE battery
        self.charged = None
        self.initial_movement_budget = 0
        self.initial_total_data = 0
        self.collected = None
        self.collide = False

    def set_landed(self, landed):
        self.landed = landed

    def set_position(self, position):
        self.position = position

    def set_collide(self, flag):
        self.collide = flag

    def decrement_movement_budget(self):
        self.movement_budget -= 1

    # NOTE  充电
    def increase_movement_budget(self):
        self.movement_budget += 4

    def set_terminal(self, terminal):
        self.terminal = terminal

    def set_device_com(self, device_com):
        self.device_com = device_com

    def get_remaining_data(self):
        return np.sum(self.device_map)

    def get_total_data(self):
        return self.initial_total_data

    def get_scalars(self):
        """
        Return the scalars without position, as it is treated individually
        """
        return np.array([self.movement_budget])

    def get_num_scalars(self):
        # NOTE 添加一个此时刻电量变化的标量
        return 1

    def get_boolean_map(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)

        padded_rest = pad_centered(self, np.expand_dims(self.landing_zone, -1), 0)

        # NOTE 在boolean_map中添加了一维充电桩可防止位置的地图
        #  0维是禁飞区，1维是障碍物区，2维是起降区，3维是充电区
        padded_battery = pad_centered(self, np.expand_dims(self.battery_zone, -1), 0)

        return np.concatenate([padded_red, padded_rest, padded_battery], axis=-1)

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_float_map(self):
        # NOTE 信息包含数据设备的位置和可收集数据的大小
        #  添加充电桩位置和充电时长信息？

        # a = pad_centered(self, np.expand_dims(self.device_map, -1), 0)
        # b = pad_centered(self, np.expand_dims(self.battery_map, -1), 0)
        # # NOTE 添加了一维信息
        # c = pad_centered(self,
        #                  np.concatenate([np.expand_dims(self.device_map, -1), np.expand_dims(self.battery_map, -1)],
        #                                 axis=-1), 0)
        return pad_centered(self,
                            np.concatenate([np.expand_dims(self.device_map, -1), np.expand_dims(self.battery_map, -1)],
                                           axis=-1), 0)

    def get_float_map_shape(self):
        return self.get_float_map().shape

    def is_in_landing_zone(self):
        return self.landing_zone[self.position[1]][self.position[0]]

    # NOTE 增加判断是非在充电区域的函数
    #  还要改一下，因为充电区域中的充电桩位置是不一定的
    def is_in_battery_zone(self):
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            return self.battery_zone[self.position[1], self.position[0]]
        return False

    def is_in_no_fly_zone(self):
        # Out of bounds is implicitly nfz
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            return self.no_fly_zone[self.position[1], self.position[0]]
        return True

    def get_collection_ratio(self):
        return np.sum(self.collected) / self.initial_total_data

    def get_collected_data(self):
        return np.sum(self.collected)

    def reset_devices(self, device_list):
        self.device_map = device_list.get_data_map(self.no_fly_zone.shape)
        # no_fly_zone.shape 为地图大小
        # 每个设备能收集的数据-已经收集的数据
        self.collected = np.zeros(self.no_fly_zone.shape, dtype=float)
        # 初始化已收集到的数据 = 0
        self.initial_total_data = device_list.get_total_data()
        # total data
        self.device_list = device_list

    # NOTE 添加 reset_battery
    def reset_batterys(self, battery_list):
        #  重置充电时间
        # for battery in battery_list.batterys:
        #     battery.charged_time = 0
        #     battery.battery_flag = 0

        self.battery_map = battery_list.get_battery_map(self.no_fly_zone.shape)
        # TODO 添加一个充电时长计数
        self.charged = np.zeros(self.no_fly_zone.shape, dtype=float)
        # self.charged = np.zeros(battery_list.num_devices, dtype=float)
        self.initial_total_charge_time = battery_list.get_total_charged_time()
        self.battery_list = battery_list

    def is_terminal(self):
        return self.terminal
