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

        # 状态
        states = [boolean_map_input,
                  float_map_input,
                  scalars_input]

        # 将x的数据格式转化成dtype数据类型.例如，原来x的数据格式是bool，
        # 将其转化成float以后，就能够将其转化成0和1的序列。
        map_cast = tf.cast(boolean_map_input, dtype=tf.float32)

        # 中心化后的地图
        # 把环境地图 + 设备地图 在第三个维度拼接- 维度越高，括号越小
        # NOTE 网络输入
        padded_map = tf.concat([map_cast, float_map_input], axis=3)

        # NOTE Q网络
        #  输入是中心化后的总map、剩余电量、    状态（中心化的bool、float、电量）属于标识输入
        self.q_network = self.build_model(padded_map, scalars_input, states)
        # NOTE 目标网络
        self.target_network = self.build_model(padded_map, scalars_input, states, 'target_')

        # 硬更新、复制参数
        self.hard_update()

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

        # NOTE Q网络的输出，输出是在当前状态，每个动作的打分
        q_values = self.q_network.output
        # 目标网络的输出  每个动作的打分
        q_target_values = self.target_network.output

        # NOTE 最小化TD误差，Q*是Q网络中使得当前状态下Q值最大的动作
        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        # 从当前网络中选取使Q值最大的动作 NOTE 然后用这个动作去计算目标网络中此动作对应的Q值
        max_action = tf.argmax(q_values, axis=1, name='max_action', output_type=tf.int64)
        # 从目标网络中选取使得Q值最大的动作   NOTE 用到了吗？
        max_action_target = tf.argmax(q_target_values, axis=1, name='max_action', output_type=tf.int64)

        # 此动作对应的one_hot编码
        one_hot_max_action = tf.one_hot(max_action, depth=self.num_actions, dtype=float)
        # tf.reduce_sum()  按一定方式计算张量中元素之和，axis指定按哪个维度进行加和，默认将所有元素进行加和；默认保持原来维度
        # tf.multiply()    将两个矩阵中对应元素各自相乘
        # NOTE q_star 是啥？  是能使当前Q网络值最大的动作 乘 one_hot 编码
        #  作用是使 只有当前的动作才置为1，然后乘相应的Q值？
        # 即上面提到的Q*
        q_star = tf.reduce_sum(tf.multiply(one_hot_max_action, q_target_values, name='mul_hot_target'), axis=1,
                               name='q_star')
        # NOTE 用使得当前Q网络值最大的动作去计算目标网络中下一状态所有动作的Q值
        #  看懂了 没问题
        self.q_star_model = Model(inputs=states, outputs=q_star)

        # Define Bellman loss 定义贝尔曼损失
        # NOTE one_hot 编码  对输入的标签或者某些离散分开的动作进行编码 在序号处置1，其他位置置0
        # https://colab.research.google.com/drive/1U2tr_Sl3WpqK_0GnspsUfuMH_zqlGj8x?hl=zh-cn#scrollTo=_Dw2_KQbcV3c
        # NOTE one_hot编码 输入的动作
        one_hot_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=1.0, off_value=0.0, dtype=float)
        # NOTE one_cold编码 输入的动作
        one_cold_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=0.0, off_value=1.0, dtype=float)
        # NOTE 不知道为什么要用两种编码？   one_cold是one_hot反过来

        # 旧的Q值  而且停止追踪梯度，即不需要反向传播
        # NOTE 此动作对应的Q值变成0，其他动作对应Q值保留？
        # Q网络的输出是q_values
        # q_values是当前状态下每个动作的Q值
        # NOTE q_old中的值时Q网络下当前状态，其他没有选到的动作的Q值（选到的动作的Q值置为0）
        q_old = tf.stop_gradient(tf.multiply(q_values, one_cold_rm_action))
        # gamma是折扣因子 termination_input 是是否终止的标志
        # tf.math.logical_not 逻辑非 对termination_input 取反
        # tf.cast   将bool类型转为float32类型  然后再用折扣因子乘转换后的数据
        # NOTE 目前理解  如果当前结束了，那么termination_input就是True，转换后变成False，再变成0，再乘折扣因子还是0
        #  如果没结束，那么termination_input就是False，转换后变成True，再变成1，再乘折扣因子，那就是gamma
        gamma_terminated = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2

        # r + gamma * q_star   # tf.add 将两个参数相加
        q_update = tf.expand_dims(tf.add(reward_input, tf.multiply(q_star_input, gamma_terminated)), 1)
        # q_update_hot 选中的动作对应的Q值为正常值、而没有选中的动作对应的Q值为0
        q_update_hot = tf.multiply(q_update, one_hot_rm_action)
        # NOTE 所以现在q_new中的值 分为两部分   一部分是Q网络没有选到的动作的Q值，一部分是target网络选中的动作对应的Q值
        q_new = tf.add(q_update_hot, q_old)
        # q_values是Q网络下每个动作的Q值
        # 计算每个动作的TD error
        # 没有选到动作的Q值的td error为0，只有target网路中选中的动作对应的Q值才会有td error
        q_loss = tf.losses.MeanSquaredError()(q_new, q_values)
        self.q_loss_model = Model(
            inputs=[boolean_map_input, float_map_input, scalars_input, action_input, reward_input,
                    termination_input, q_star_input],
            outputs=q_loss)

        # Exploit act model
        # NOTE 探索   输入状态，Q网络输出使得Q值最大的动作
        self.exploit_model = Model(inputs=states, outputs=max_action)
        # NOTE 探索   输入状态，Q_target网络输出使得Q值最大的动作
        self.exploit_model_target = Model(inputs=states, outputs=max_action_target)

        # Softmax explore model
        # tf.divide 两个张量相除
        softmax_scaling = tf.divide(q_values, tf.constant(self.params.soft_max_scaling, dtype=float))
        # 转为概率，总和为1
        softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')

        # soft——max 动作
        self.soft_explore_model = Model(inputs=states, outputs=softmax_action)

        # q网络的优化器 adam优化器
        self.q_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.q_loss_model.summary()

        if stats:
            stats.set_model(self.target_network)

    def build_model(self, map_proc, states_proc, inputs, name=''):
        # NOTE 输入的map是中心化处理过后的
        #  状态也是中心化的
        #  对输入的图像进行局部和全局处理，通过所有卷积层，提取出特征，张成1维的特征向量并且拼接
        flatten_map = self.create_map_proc(map_proc, name)
        # NOTE 一维的特征向量和电量拼接到一起  合并输入
        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        # NOTE 分别通过全连接层，输出维度是动作的个数
        # 隐藏层
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)

        # TODO 添加价值网络
        # 价值网络输出
        V_s_value = Dense(1, name=name+'prediction_v_value')(layer)
        # TODO 添加优势网络
        # 优势网络输出
        A_s_a = Dense(self.num_actions, name=name+'prediction_A_s_a')(layer)

        # NOTE 对优势网络求平均，再相减,再加上价值网络，得到最后的输出
        output = V_s_value + A_s_a - tf.reduce_mean(A_s_a, axis=1, keepdims=True)
        # output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)
        # 全连接层

        model = Model(inputs=inputs, outputs=output)
        return model

    def build_dueling_model(self, map_proc, states_proc, inputs, name=''):
        # NOTE 输入的map是中心化处理过后的
        #  状态也是中心化的
        #  对输入的图像进行局部和全局处理，通过所有卷积层，提取出特征，张成1维的特征向量并且拼接
        flatten_map = self.create_map_proc(map_proc, name)
        # NOTE 一维的特征向量和电量拼接到一起  合并输入
        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        # NOTE 分别通过全连接层，输出维度是动作的个数
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)

        # 添加另外两层特殊的结构
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)
        # 全连接层

        model = Model(inputs=inputs, outputs=output)
        return model

    def create_map_proc(self, conv_in, name):
        # 如果用到了全局和局部地图
        if self.params.use_global_local:
            # Forking for global and local map
            # Global Map
            # NOTE 停止追踪梯度、平均池化
            # 池化窗口、池化步长
            # NOTE 对输入进行平均池化得到 global_map
            #  total_map是原来的输入
            global_map = tf.stop_gradient(
                AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))

            self.global_map = global_map
            # 全局变量
            self.total_map = conv_in

            # NOTE 对平均池化后的global_map进行两次卷积，得到新的global_map
            #  参数分别为卷积核个数、卷积核大小
            for k in range(self.params.conv_layers):
                global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                    strides=(1, 1),
                                    name=name + 'global_conv_' + str(k + 1))(global_map)

            # 将卷积后的全局特征图张成1维的特征向量
            flatten_global = Flatten(name=name + 'global_flatten')(global_map)

            # Local Map
            # 局部地图的大小
            crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
            # 图像裁剪  crop_frac是图像中心的百分之多少，裁剪出局部地图
            local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
            self.local_map = local_map

            # NOTE local_map没有平均池化，从输入的图像中裁剪局部地图
            for k in range(self.params.conv_layers):
                local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                   strides=(1, 1),
                                   name=name + 'local_conv_' + str(k + 1))(local_map)

            # 将local_map卷积后得到的特征图张成1维的特征向量
            flatten_local = Flatten(name=name + 'local_flatten')(local_map)
            # NOTE 将整体和局部的特征拼接到一起
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

        # min(Q - (r + gamma_terminated * Q*))^2
        # NOTE q_star_model 输入状态，输出打分  Q*是一维的，是能使当前Q网络输出Q值最大的动作放到targetQ网络中计算的Q值
        q_star = self.q_star_model(
            [next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape:
            q_loss = self.q_loss_model(
                [boolean_map, float_map, scalars, action, reward,
                 terminated, q_star])
        # 计算梯度
        q_grads = tape.gradient(q_loss, self.q_network.trainable_variables)
        # 更新梯度
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_network.trainable_variables))
        # 软更新
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
