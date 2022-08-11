import numpy as np
from src.Map.Map import Map
from src.StateUtils import pad_centered
from src.base.BaseState import BaseState


class DHMultiState(BaseState):
    def __init__(self, map_init: Map, num_agents: int):
        super().__init__(map_init)
        self.device_list = None
        self.device_map = None  # Floating point sparse matrix showing devices and their data to be collected

        # Multi-agent active agent decides on properties
        self.active_agent = 0
        self.num_agents = num_agents

        # Multi-agent is creating lists
        self.positions = [[0, 0]] * num_agents
        self.movement_budgets = [0] * num_agents
        self.landeds = [False] * num_agents
        self.terminals = [False] * num_agents
        self.device_coms = [-1] * num_agents

        self.initial_movement_budgets = [0] * num_agents
        self.initial_total_data = 0
        self.collected = None

    @property
    def position(self):
        return self.positions[self.active_agent]

    @property
    def movement_budget(self):
        return self.movement_budgets[self.active_agent]

    @property
    def initial_movement_budget(self):
        return self.initial_movement_budgets[self.active_agent]

    @property
    def landed(self):
        return self.landeds[self.active_agent]

    @property
    def terminal(self):
        return self.terminals[self.active_agent]

    @property
    def all_landed(self):
        return all(self.landeds)

    @property
    def all_terminal(self):
        return all(self.terminals)

    def is_terminal(self):
        return self.all_terminal

    def set_landed(self, landed):
        self.landeds[self.active_agent] = landed

    def set_position(self, position):
        self.positions[self.active_agent] = position

    def decrement_movement_budget(self):
        self.movement_budgets[self.active_agent] -= 1

    def set_terminal(self, terminal):
        self.terminals[self.active_agent] = terminal

    def set_device_com(self, device_com):
        self.device_coms[self.active_agent] = device_com

    def get_active_agent(self):
        return self.active_agent

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
        return len(self.get_scalars())

    def get_boolean_map(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        padded_rest = pad_centered(self,
                                   np.concatenate(
                                       [np.expand_dims(self.landing_zone, -1), self.get_agent_bool_maps()],
                                       axis=-1), 0)
        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_float_map(self):
        return pad_centered(self, np.concatenate([np.expand_dims(self.device_map, -1),
                                                  self.get_agent_float_maps()], axis=-1), 0)

    def get_float_map_shape(self):
        return self.get_float_map().shape

    def is_in_landing_zone(self):
        return self.landing_zone[self.position[1]][self.position[0]]

    def is_in_no_fly_zone(self):
        # Out of bounds is implicitly nfz
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            # NFZ or occupied
            return self.no_fly_zone[self.position[1], self.position[0]] or self.is_occupied()
        return True

    def is_occupied(self):
        for i, pos in enumerate(self.positions):
            if self.terminals[i]:
                continue
            if i == self.active_agent:
                continue
            if pos == self.position:
                return True
        return False

    def get_collection_ratio(self):
        return np.sum(self.collected) / self.initial_total_data

    def get_collected_data(self):
        return np.sum(self.collected)

    def reset_devices(self, device_list):
        self.device_map = device_list.get_data_map(self.no_fly_zone.shape)
        self.collected = np.zeros(self.no_fly_zone.shape, dtype=float)
        self.initial_total_data = device_list.get_total_data()
        self.device_list = device_list

    def get_agent_bool_maps(self):
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=bool)
        for agent in range(self.num_agents):
            # agent_map[self.positions[agent][1], self.positions[agent][0]][0] = self.landeds[agent]
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = not self.terminals[agent]
        return agent_map

    def get_agent_float_maps(self):
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=float)
        for agent in range(self.num_agents):
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = self.movement_budgets[agent]
        return agent_map
