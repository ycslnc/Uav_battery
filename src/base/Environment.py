import copy
import tqdm
import distutils.util

from src.ModelStats import ModelStatsParams, ModelStats
from src.base.BaseDisplay import BaseDisplay
from src.base.GridActions import GridActions


class BaseEnvironmentParams:
    def __init__(self):
        # 没参数传入的时候的默认参数
        self.model_stats_params = ModelStatsParams()


class BaseEnvironment:
    def __init__(self, params: BaseEnvironmentParams, display: BaseDisplay):
        self.stats = ModelStats(params.model_stats_params, display=display)
        self.trainer = None
        self.agent = None
        self.grid = None
        self.rewards = None
        self.physics = None
        self.display = display
        self.episode_count = 0
        self.step_count = 0

    # NOTE 已改完
    def fill_replay_memory(self):
        while self.trainer.should_fill_replay_memory():
            state = copy.deepcopy(self.init_episode())
            while not state.terminal:
                next_state = self.step(state, random=self.trainer.params.rm_pre_fill_random)
                state = copy.deepcopy(next_state)

    def train_episode(self):
        state = copy.deepcopy(self.init_episode())
        # √
        self.stats.on_episode_begin(self.episode_count)
        # 设置TensorBoard所要记录的召回函数
        while not state.is_terminal():
            state = self.step(state)
            self.trainer.train_agent()

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_training_data(step=self.step_count)

        self.episode_count += 1

    def run(self):
        # NOTE 这里不用改
        self.fill_replay_memory()

        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()

            if self.episode_count % self.trainer.params.eval_period == 0:
                self.test_episode()

            self.stats.save_if_best()

        self.stats.training_ended()

    def step(self, state, random=False):
        if random:
            action = self.agent.get_random_action()
        else:
            action = self.agent.act(state)
        next_state = self.physics.step(GridActions(action))
        # movement_step（下一步的agent的位置）→comm_step（收集数据等）
        # 每一次step之后都会检查是否结束，设置terminal标志
        reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
        # 计算累计reward
        # NOTE 此时添加的经验是四维的，即包含了充电桩信息
        self.trainer.add_experience(state, action, reward, next_state)
        # sars 四元组作为一条经验
        self.stats.add_experience((state, action, reward, copy.deepcopy(next_state)))

        self.step_count += 1
        return copy.deepcopy(next_state)

    def test_episode(self, scenario=None):
        state = copy.deepcopy(self.init_episode(scenario))
        print("初始电量为:", state.movement_budget)
        self.stats.on_episode_begin(self.episode_count)
        while not state.terminal:
            print("当前电量为:", state.movement_budget)
            action = self.agent.get_exploitation_action_target(state)
            next_state = self.physics.step(GridActions(action))
            reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
            self.stats.add_experience((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))
            state = copy.deepcopy(next_state)
        print("中途充电电量为:", self.physics.charge_time * 4)
        print("消耗电量为:", self.physics.num_step)
        print("最后剩余电量为:", state.movement_budget)
        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step=self.step_count)

    def test_scenario(self, scenario):
        state = copy.deepcopy(self.init_episode(scenario))
        while not state.terminal:
            action = self.agent.get_exploitation_action_target(state)
            state = self.physics.step(GridActions(action))

    def init_episode(self, init_state=None):
        if init_state:
            state = copy.deepcopy(self.grid.init_scenario(init_state))
            # 有了初始state，就返回带着设备列表的state
        else:
            state = copy.deepcopy(self.grid.init_episode())
            # grid.init_episode()

        self.rewards.reset()
        # TODO 添加一个充电桩reset
        self.physics.reset(state)
        return state

    def eval(self, episodes, show=False):
        for _ in tqdm.tqdm(range(episodes)):
            self.test_episode()
            self.step_count += 1  # Increase step count so that logging works properly

            if show:
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

                resp = input('Save run? [y/N]\n')
                try:
                    if distutils.util.strtobool(resp):
                        save_as = input('Save as: [run_' + str(self.step_count) + ']\n')
                        if save_as == '':
                            save_as = 'run_' + str(self.step_count)
                        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                                     save_path=save_as + '.png')
                        self.stats.save_episode(save_as)
                        print("Saved as run_" + str(self.step_count))
                except ValueError:
                    pass
                print("next then")

    def eval_scenario(self, init_state):
        self.test_scenario(init_state)

        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

        resp = input('Save run? [y/N]\n')
        try:
            if distutils.util.strtobool(resp):
                save_as = input('Save as: [scenario]\n')
                if save_as == '':
                    save_as = 'scenario'
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                             save_path=save_as + '.png')
                self.stats.save_episode(save_as)
                print("Saved as", save_as)
        except ValueError:
            pass
