from src.DDQN.Agent import DDQNAgentParams, DDQNAgent
from src.DDQN.Trainer import DDQNTrainerParams, DDQNTrainer
from src.DH.Display import DHDisplay
from src.DH.Grid import DHGridParams, DHGrid
from src.DH.Physics import DHPhysicsParams, DHPhysics
from src.DH.Rewards import DHRewardParams, DHRewards
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams


class DHEnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        # https://blog.csdn.net/a__int__/article/details/104600972
        # NOTE 覆盖父类的初始化，但是又通过super().__init__()继承回来
        self.grid_params = DHGridParams()
        self.reward_params = DHRewardParams()
        self.trainer_params = DDQNTrainerParams()
        self.agent_params = DDQNAgentParams()
        self.physics_params = DHPhysicsParams()


class DHEnvironment(BaseEnvironment):
    def __init__(self, params: DHEnvironmentParams):
        self.display = DHDisplay()
        # NOTE 可视化部分  先不看
        super().__init__(params, self.display)

        self.grid = DHGrid(params.grid_params, stats=self.stats)
        self.rewards = DHRewards(params.reward_params, stats=self.stats)
        self.physics = DHPhysics(params=params.physics_params, stats=self.stats)
        self.agent = DDQNAgent(params.agent_params, self.grid.get_example_state(), self.physics.get_example_action(),
                               stats=self.stats)
        self.trainer = DDQNTrainer(params.trainer_params, agent=self.agent)

        self.display.set_channel(self.physics.channel)

