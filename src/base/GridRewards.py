from src.base.GridActions import GridActions


class GridRewardParams:
    def __init__(self):
        self.boundary_penalty = 1.0
        self.empty_battery_penalty = 150.0
        self.movement_penalty = 0.2


class GridRewards:
    def __init__(self, stats):
        self.params = GridRewardParams()
        self.cumulative_reward: float = 0.0

        stats.add_log_data_callback('cumulative_reward', self.get_cumulative_reward)

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def calculate_motion_rewards(self, state, action: GridActions, next_state):
        reward = 0.0
        # NOTE 任务未完成的惩罚  论文中的r_mov
        if not next_state.landed:
            # Penalize battery Consumption
            reward -= self.params.movement_penalty
        # NOTE 没有移动的惩罚 这个可以试一下去掉
        # Penalize not moving (This happens when it either tries to land or fly into a boundary or hovers or fly into
        # a cell occupied by another agent)
        # NOTE r_sc  针对的情况是  此动作下一时刻要发生碰撞，所以重置为上一时刻的位置即悬停
        # 下一时刻未降落、动作不是悬停 但是前后位置一样（发生碰撞，重置了位置）
        # NOTE 应该改为 发生碰撞给与惩罚，而悬停不给惩罚
        # if state.position == next_state.position and not next_state.landed and not action == GridActions.HOVER:
        if next_state.collide:
            reward -= self.params.boundary_penalty

        # Penalize battery dead
        # NOTE 论文中的r_crash
        if next_state.movement_budget == 0 and not next_state.landed:
            reward -= self.params.empty_battery_penalty

        # TODO 如果电量比例低于一定值  做出一点改变
        return reward

    def reset(self):
        self.cumulative_reward = 0
