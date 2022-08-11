import numpy as np

from src.DH.Battery import BatteryParams, BatteryList

ColorMap = ["forestgreen", "lightgreen", "darkgreen", "lime", "lightgreen", "purple"]

# "blue", "orange", "green", "red",


class BatteryManagerParams:
    def __init__(self):
        self.battery_count_range = (1, 2)
        self.fixed_devices = False
        self.devices = BatteryParams()


class BatteryManager:
    def __init__(self, params: BatteryManagerParams):
        self.params = params

    # NOTE 禁飞区可以放 decive
    #  设备和充电桩之间不存在冲突
    #  但是设备可以放置在禁飞区，而充电桩不能放置在禁飞区，只能放置在空闲的区域
    def generate_device_list(self, positions_vector):
        if self.params.fixed_batterys:
            # return BatteryList(self.params.devices)
            device_count = len(self.params.devices)
            positions = [tuple(poi) for poi in self.params.devices]
            return self.generate_battery_list_from_args(device_count, positions)

        device_count = np.random.randint(self.params.battery_count_range[0], self.params.battery_count_range[1] + 1)
        # print("battery count", device_count)
        position_idcs = np.random.choice(range(len(positions_vector)), device_count, replace=False)
        positions = [positions_vector[idx] for idx in position_idcs]

        return self.generate_battery_list_from_args(device_count, positions)
        # 返回设备数量（从参数中范围中随机选取）、设备位置（从空余的位置中随机选取）、各设备对应的数据

        # NOTE 增加一个产生充电桩设备列表的函数，返回充电桩数量，充电桩位置

    def generate_battery_list_from_args(self, device_count, positions):
        # get colors
        colors = ColorMap[0:max(device_count, len(ColorMap))]

        params = [BatteryParams(position=positions[k], color=colors[k % len(ColorMap)])
                  for k in range(device_count)]

        return BatteryList(params)
