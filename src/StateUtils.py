import math
import numpy as np


def pad_centered(state, map_in, pad_value):
    padding_rows = math.ceil(state.no_fly_zone.shape[0] / 2.0)
    # 向上取整  4行
    padding_cols = math.ceil(state.no_fly_zone.shape[1] / 2.0)
    # 4列
    position_x, position_y = state.position
    # agent的位置坐标
    position_row_offset = padding_rows - position_y
    position_col_offset = padding_cols - position_x
    # 中心化后的坐标
    return np.pad(map_in,
                  pad_width=[[padding_rows + position_row_offset - 1, padding_rows - position_row_offset],
                             [padding_cols + position_col_offset - 1, padding_cols - position_col_offset],
                             [0, 0]],
                  mode='constant',
                  constant_values=pad_value)
    # 边缘数值填充
