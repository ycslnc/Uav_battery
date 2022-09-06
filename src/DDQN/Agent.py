import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D

import numpy as np


def print_node(x):
    print(x)
    return x


class DDQNAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # Training Params
        self.learning_rate = 3e-5
        self.alpha = 0.005
        self.gamma = 0.95

        # Exploration strategy
        self.soft_max_scaling = 0.1

        # Global-Local Map
        self.use_global_local = True
        self.global_map_scaling = 3
        self.local_map_size = 17

        # Printing
        self.print_summary = False


class DDQNAgent(object):

    def __init__(self, params: DDQNAgentParams, example_state, example_action, stats=None):

        self.params = params
        # 折扣因子
        gamma = tf.constant(self.params.gamma, dtype=float)
        self.align_counter = 0
        # 状态中的两个地图都已经中心化过了
        self.boolean_map_shape = example_state.get_boolean_map_shape()
        self.float_map_shape = example_state.get_float_map_shape()
        # 剩余电量
        self.scalars = example_state.get_num_scalars()
        # 动作数量
        self.num_actions = len(type(example_action))
        # 总通道数
        self.num_map_channels = self.boolean_map_shape[2] + self.float_map_shape[2]

        # Create shared inputs
        # 输入
        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        # NOTE 还没弄清楚这个用来干啥
        q_star_input = Input(shape=(), name='q_star_input', dtype=tf.float32)
        # 环境地图+设备地图+电量
        states = [boolean_map_input,
                  float_map_input,
                  scalars_input]

        # 将x的数据格式转化成dtype数据类型.例如，原来x的数据格式是bool，
        # 将其转化成float以后，就能够将其转化成0和1的序列。
        map_cast = tf.cast(boolean_map_input, dtype=tf.float32)

        # 中心化后的地图
        # 把环境地图 + 设备地图 在第三个维度拼接-     维度越高，括号越小
        padded_map = tf.concat([map_cast, float_map_input], axis=3)

        # Q网络
        self.q_network = self.build_model(padded_map, scalars_input, states)
        # 目标网络
        self.target_network = self.build_model(padded_map, scalars_input, states, 'target_')
        # build_model  卷积+隐藏
        self.hard_update()
        # 硬更新、复制参数

        # 如果用到了全局+局部地图
        if self.params.use_global_local:
            # 全局地图模型
            self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                          outputs=self.global_map)
            # 局部地图模型
            self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.local_map)
            # 总地图模型
            self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.total_map)

        # Q值
        q_values = self.q_network.output
        # 目标值
        q_target_values = self.target_network.output

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2

        # 从当前网络中选取使Q值最大的动作 NOTE 然后用这个动作去计算目标网络中此动作对应的Q值
        max_action = tf.argmax(q_values, axis=1, name='max_action', output_type=tf.int64)
        # 从目标网络中选取使得Q值最大的动作   NOTE 用到了吗？
        max_action_target = tf.argmax(q_target_values, axis=1, name='max_action', output_type=tf.int64)

        # 此动作对应的one_hot编码
        one_hot_max_action = tf.one_hot(max_action, depth=self.num_actions, dtype=float)

        # tf.reduce_sum()  按一定方式计算张量中元素之和，axis指定按哪个维度进行加和，默认将所有元素进行加和；默认保持原来维度
        # tf.multiply()    将两个矩阵中对应元素各自相乘
        # NOTE q_star 是啥？  是能使当前Q网络值最大的动作 乘 one_hot 编码？？？
        q_star = tf.reduce_sum(tf.multiply(one_hot_max_action, q_target_values, name='mul_hot_target'), axis=1,
                               name='q_star')
        self.q_star_model = Model(inputs=states, outputs=q_star)

        # Define Bellman loss 定义贝尔曼损失
        # NOTE one_hot编码 输入的动作
        one_hot_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=1.0, off_value=0.0, dtype=float)
        # NOTE one_cold编码 输入的动作
        one_cold_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=0.0, off_value=1.0, dtype=float)
        # NOTE 不知道为什么要用两种编码？   one_cold是one_hot反过来

        # 旧的Q值  而且停止追踪梯度，即不需要反向传播
        q_old = tf.stop_gradient(tf.multiply(q_values, one_cold_rm_action))

        # gamma是折扣因子 termination_input 是是否终止的标志
        # tf.math.logical_not 逻辑非 对termination_input 取反
        # tf.cast   将bool类型转为float32类型  然后再用折扣因子乘转换后的数据
        # NOTE 目前理解  如果当前结束了，那么termination_input就是True，转换后变成False，再变成0，再乘折扣因子还是0
        #  如果没结束，那么termination_input就是False，转换后变成True，再变成1，再乘折扣因子，那就是gamma
        gamma_terminated = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)

        q_update = tf.expand_dims(tf.add(reward_input, tf.multiply(q_star_input, gamma_terminated)), 1)
        q_update_hot = tf.multiply(q_update, one_hot_rm_action)
        q_new = tf.add(q_update_hot, q_old)
        q_loss = tf.losses.MeanSquaredError()(q_new, q_values)
        self.q_loss_model = Model(
            inputs=[boolean_map_input, float_map_input, scalars_input, action_input, reward_input,
                    termination_input, q_star_input],
            outputs=q_loss)

        # Exploit act model
        self.exploit_model = Model(inputs=states, outputs=max_action)
        self.exploit_model_target = Model(inputs=states, outputs=max_action_target)

        # Softmax explore model
        softmax_scaling = tf.divide(q_values, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')
        self.soft_explore_model = Model(inputs=states, outputs=softmax_action)

        self.q_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.q_loss_model.summary()

        if stats:
            stats.set_model(self.target_network)

    def build_model(self, map_proc, states_proc, inputs, name=''):

        flatten_map = self.create_map_proc(map_proc, name)
        # 对输入的图像进行局部和全局处理，通过所有卷积层，提取出特征向量，张成1维

        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        # 隐藏层
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)
        # 全连接层

        model = Model(inputs=inputs, outputs=output)

        return model

    def create_map_proc(self, conv_in, name):

        if self.params.use_global_local:
            # Forking for global and local map
            # 局部和全局地图的分支
            # Global Map
            # 局部变量
            global_map = tf.stop_gradient(
                AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))
            # 平均池化
            # 池化窗口、池化步长
            self.global_map = global_map
            # 全局变量
            self.total_map = conv_in

            for k in range(self.params.conv_layers):
                global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                    strides=(1, 1),
                                    name=name + 'global_conv_' + str(k + 1))(global_map)
            # 参数分别为卷积核个数、卷积核大小
            flatten_global = Flatten(name=name + 'global_flatten')(global_map)
            # 张成1维的特征向量

            # Local Map
            crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
            # 局部地图的大小
            local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
            # 图像裁剪  crop_frac是图像中心的百分之多少，裁剪出局部地图
            self.local_map = local_map

            for k in range(self.params.conv_layers):
                local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                   strides=(1, 1),
                                   name=name + 'local_conv_' + str(k + 1))(local_map)

            flatten_local = Flatten(name=name + 'local_flatten')(local_map)

            return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])
        else:
            conv_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu', strides=(1, 1),
                              name=name + 'map_conv_0')(conv_in)
            for k in range(self.params.conv_layers - 1):
                conv_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                  strides=(1, 1),
                                  name=name + 'map_conv_' + str(k + 1))(conv_map)

            flatten_map = Flatten(name=name + 'flatten')(conv_map)
            return flatten_map

    def act(self, state):
        return self.get_soft_max_exploration(state)

    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def get_exploitation_action(self, state):

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def get_soft_max_exploration(self, state):

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        p = self.soft_explore_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
        return np.random.choice(range(self.num_actions), size=1, p=p)

    def get_exploitation_action_target(self, state):

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model_target([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def hard_update(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def soft_update(self, alpha):
        weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        self.target_network.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])

    def train(self, experiences):
        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]
        #

        q_star = self.q_star_model(
            [next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape:
            q_loss = self.q_loss_model(
                [boolean_map, float_map, scalars, action, reward,
                 terminated, q_star])
        q_grads = tape.gradient(q_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_network.trainable_variables))

        self.soft_update(self.params.alpha)

    def save_weights(self, path_to_weights):
        self.target_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.target_network.save(path_to_model)

    def load_weights(self, path_to_weights):
        self.q_network.load_weights(path_to_weights)
        self.hard_update()

    def get_global_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.global_map_model([boolean_map_in, float_map_in]).numpy()

    def get_local_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.local_map_model([boolean_map_in, float_map_in]).numpy()

    def get_total_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.total_map_model([boolean_map_in, float_map_in]).numpy()
