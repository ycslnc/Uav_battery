import numpy as np


class BatteryParams:
    def __init__(self, position=(0, 0), color='blue'):
        self.position = position
        self.color = color


class Battery:
    charged_time: float
    battery_flag: bool

    def __init__(self, params: BatteryParams):
        self.params = params
        self.position = params.position  # fixed position can be later overwritten in reset
        self.color = params.color
        self.charged_time = 1

    def charge_time(self):
        return self.charged_time


class BatteryList:

    def __init__(self, params):
        self.batterys = [Battery(device) for device in params]

    def get_battery_map(self, shape):
        # 获取充电详情
        battery_map = np.zeros(shape, dtype=float)
        for battery in self.batterys:
            battery_map[battery.position[1], battery.position[0]] = battery.charged_time
        return battery_map
        # 含有每个充电桩冲了多少电的信息

    def get_batterys(self):
        return self.batterys

    def get_battery(self, idx):
        return self.batterys[idx]

    def get_total_charged_time(self):
        return sum(list([batery.charged_time for batery in self.batterys]))

    @property
    def num_devices(self):
        return len(self.batterys)
