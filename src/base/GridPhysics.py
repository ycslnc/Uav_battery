from src.base.GridActions import GridActions


class GridPhysics:
    def __init__(self):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = None
        # NOTE 充电时长
        self.charge_time = 0

    def movement_step(self, action: GridActions):
        old_position = self.state.position
        x, y = old_position

        if action == GridActions.NORTH:
            y += 1
        elif action == GridActions.SOUTH:
            y -= 1
        elif action == GridActions.WEST:
            x -= 1
        elif action == GridActions.EAST:
            x += 1
        elif action == GridActions.LAND:
            self.landing_attempts += 1
            if self.state.is_in_landing_zone():
                self.state.set_landed(True)
        # NOTE
        #  step一步，然后设置智能体新位置

        self.state.set_position([x, y])
        if self.state.is_in_no_fly_zone():
            # Reset state
            self.boundary_counter += 1
            x, y = old_position
            self.state.set_position([x, y])
        self.state.decrement_movement_budget()
        # step一步，电量就减少一格
        # TODO 设置如果在充电桩位置，那么就设置电量+k
        '''
                if (x,y) in battery.position:
            self.state.increase_movement_budget()
        '''
        # if self.state.is_in_battery_zone():
        #     self.state.increase_movement_budget()
        #     self.charge_time += 1

        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))

        return x, y

    def reset(self, state):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state
