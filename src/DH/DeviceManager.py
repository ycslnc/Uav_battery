import numpy as np

from src.DH.IoTDevice import IoTDeviceParams, DeviceList

ColorMap = ["blue", "orange", "violet", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]


class DeviceManagerParams:
    def __init__(self):
        self.device_count_range = (2, 5)
        self.data_range = (5.0, 20.0)
        self.fixed_devices = False
        self.devices = IoTDeviceParams()


class DeviceManager:
    """
    The DeviceManager is used to generate DeviceLists according to DeviceManagerParams
    """

    def __init__(self, params: DeviceManagerParams):
        self.params = params

    # NOTE 禁飞区可以放 decive
    #  设备和充电桩之间不存在冲突
    #  但是设备可以放置在禁飞区，而充电桩不能放置在禁飞区，只能放置在空闲的区域
    def generate_device_list(self, positions_vector):
        if self.params.fixed_devices:
            return DeviceList(self.params.devices)

        # Roll number of devices
        # range 左闭右开，所以右边没算 ，需要+1
        device_count = np.random.randint(self.params.device_count_range[0], self.params.device_count_range[1] + 1)
        # print("device count", device_count)
        # Roll Positions
        position_idcs = np.random.choice(range(len(positions_vector)), device_count, replace=False)
        # numpy.random.choice(a, size=None, replace=True, p=None)
        # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
        # replace:True表示可以取相同数字，False表示不可以取相同数字
        # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        positions = [positions_vector[idx] for idx in position_idcs]
        # 获取每个设备的位置（像素位置）
        # Roll Data
        datas = np.random.uniform(self.params.data_range[0], self.params.data_range[1], device_count)
        # 生成每个设备所能收集的最大数据量
        return self.generate_device_list_from_args(device_count, positions, datas)
        # 返回设备数量（从参数中范围中随机选取）、设备位置（从空余的位置中随机选取）、各设备对应的数据


    def generate_device_list_from_args(self, device_count, positions, datas):
        # get colors
        colors = ColorMap[0:max(device_count, len(ColorMap))]

        params = [IoTDeviceParams(position=positions[k],
                                  data=datas[k],
                                  color=colors[k % len(ColorMap)])
                  for k in range(device_count)]

        return DeviceList(params)
