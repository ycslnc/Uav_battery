import numpy as np

from src.DH.Channel import ChannelParams, Channel
from src.DH.State import DHState
from src.ModelStats import ModelStats
from src.base.GridActions import GridActions
from src.base.GridPhysics import GridPhysics


class DHPhysicsParams:
    def __init__(self):
        self.channel_params = ChannelParams()
        self.comm_steps = 4


class DHPhysics(GridPhysics):

    def __init__(self, params: DHPhysicsParams, stats: ModelStats):

        super().__init__()

        self.channel = Channel(params.channel_params)

        self.params = params

        self.register_functions(stats)

        self.num_step = 0

    def register_functions(self, stats: ModelStats):
        stats.set_evaluation_value_callback(self.get_cral)

        stats.add_log_data_callback('cral', self.get_cral)
        stats.add_log_data_callback('cr', self.get_collection_ratio)
        stats.add_log_data_callback('successful_landing', self.has_landed)
        stats.add_log_data_callback('boundary_counter', self.get_boundary_counter)
        stats.add_log_data_callback('landing_attempts', self.get_landing_attempts)
        stats.add_log_data_callback('movement_ratio', self.get_movement_ratio)

    def reset(self, state: DHState):
        GridPhysics.reset(self, state)

        self.channel.reset(self.state.shape[0])

    def step(self, action: GridActions):
        old_position = self.state.position
        self.movement_step(action)
        # 如果发生碰撞，那么碰撞次数+1 并且重置为上一时刻的位置
        self.num_step += 1
        if not self.state.terminal:
            self.comm_step(old_position)
            # 计算数据收集
            # NOTE 添加充电step
            self.battery_step()

        return self.state

    def comm_step(self, old_position):
        positions = list(
            reversed(np.linspace(self.state.position, old_position, num=self.params.comm_steps, endpoint=False)))

        indices = []
        # 记录是第几个设备在收集数据
        device_list = self.state.device_list
        for position in positions:
            data_rate, idx = device_list.get_best_data_rate(position, self.channel)
            # 获取最佳数据速率和对应的设备索引
            device_list.collect_data(data_rate, idx)
            # 收集数据
            indices.append(idx)
            # 把设备索引添加到索引列表中

        self.state.collected = device_list.get_collected_map(self.state.shape)
        # 已收集的数据
        self.state.device_map = device_list.get_data_map(self.state.shape)
        # 8x8 可收集的数据 - 已收集的数据
        '''
        self.state.charged = 
        
        '''
        idx = max(set(indices), key=indices.count)
        self.state.set_device_com(idx)

        return idx

    def battery_step(self):

        old_position = self.state.position
        x, y = old_position

        battery_list = self.state.battery_list
        # 计算电池所在位置
        battery_position = [battery.position for battery in battery_list.batterys]
        # 判断是否在充电区域内
        if self.state.is_in_battery_zone():
            # 判断位置是否在设备上
            if tuple(self.state.position) in battery_position:
                battery_idx = battery_position.index(tuple(self.state.position))
                # 充电时间+1
                # NOTE 设为常数
                self.state.battery_list.batterys[battery_idx].charged_time += 1
                # self.state.battery_list.batterys[battery_idx].battery_flag = 1
                self.charge_time += 1
                # 需要判断充电之后总电量是否超过无人机电池的最大电量
                if self.state.movement_budget + 4 <= self.state.initial_movement_budget:
                    # 步数+4
                    self.state.increase_movement_budget()

        # 在获取电池map之前更新充电情况
        self.state.battery_map = battery_list.get_battery_map(self.state.shape)
        # NOTE 判断是否在充电区域，在充电区域的话，再计算具体冲了多少电, 要放在修改已充电时间后面
        # 总充电时长要有限制（不然训练的时候会一直卡在那里）
        self.state.set_terminal(self.charge_time >= 20)


    def get_example_action(self):
        return GridActions.HOVER

    def is_in_landing_zone(self):
        return self.state.is_in_landing_zone()

    def get_collection_ratio(self):
        return self.state.get_collection_ratio()

    def get_movement_budget_used(self):
        return self.state.initial_movement_budget - self.state.movement_budget

    def get_max_rate(self):
        return self.channel.get_max_rate()

    def get_average_data_rate(self):
        return self.state.get_collected_data() / self.get_movement_budget_used()

    def get_cral(self):
        return self.get_collection_ratio() * self.state.landed

    def get_boundary_counter(self):
        return self.boundary_counter

    def get_landing_attempts(self):
        return self.landing_attempts

    def get_movement_ratio(self):
        return float(self.get_movement_budget_used()) / float(self.state.initial_movement_budget)

    def has_landed(self):
        return self.state.landed

    # NOTE 增加查询的函数
    def get_charged_time(self, state: DHState):
        return state.charged
