import numpy as np
# import matplotlib.pyplot as plt
import io
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from src.Map.Map import Map
from src.base.BaseDisplay import BaseDisplay


class DHDisplay(BaseDisplay):

    def __init__(self):
        super().__init__()
        self.channel = None

    def set_channel(self, channel):
        self.channel = channel

    def draw_bar_plots(self, final_state, ax_bar):
        # Add bar plots
        device_list = final_state.device_list
        devices = device_list.get_devices()
        colors = [device.color for device in devices]
        names = ["total"] + colors
        colors = ["black"] + colors
        datas = [device_list.get_total_data()] + [device.data for device in devices]
        collected_datas = [device_list.get_collected_data()] + [device.collected_data for device in devices]
        y_pos = np.arange(len(colors))

        plt.sca(ax_bar)
        ax_bar.barh(y_pos, datas)
        ax_bar.barh(y_pos, collected_datas)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(names)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Data")
        ax_bar.set_aspect(- np.diff(ax_bar.get_xlim())[0] / np.diff(ax_bar.get_ylim())[0])

    def display_episode(self, env_map: Map, trajectory, plot=False, save_path=None):
        # get
        first_state = trajectory[0][0]
        final_state = trajectory[-1][3]

        fig_size = 5.5
        fig, ax = plt.subplots(1, 2, figsize=[2 * fig_size, fig_size])
        ax_traj = ax[0]
        ax_bar = ax[1]

        value_step = 0.4 / first_state.device_list.num_devices
        # Start with value of 200
        value_map = np.ones(env_map.get_size(), dtype=float)
        for device in first_state.device_list.get_devices():
            value_map -= value_step * self.channel.total_shadow_map[device.position[1], device.position[0]]

        self.create_grid_image(ax=ax_traj, env_map=env_map, value_map=value_map)

        for device in first_state.device_list.get_devices():
            ax_traj.add_patch(
                patches.Circle(np.array(device.position) + np.array((0.5, 0.5)), 0.4, facecolor=device.color,
                               edgecolor="black"))

        for battery in first_state.battery_list.get_batterys():
            ax_traj.add_patch(
                patches.Ellipse(np.array(battery.position) + np.array((0.5, 0.5)), 0.4, 0.3, facecolor=battery.color,
                                edgecolor="black"))

        self.draw_start_and_end(trajectory)

        for exp in trajectory:
            idx = exp[3].device_com
            if idx == -1:
                color = "black"
            else:
                color = exp[0].device_list.devices[idx].color

            self.draw_movement(exp[0].position, exp[3].position, color=color)

        self.draw_bar_plots(final_state, ax_bar)

        # save image and return
        if save_path is not None:
            # save just the trajectory subplot 0
            extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent.x0 -= 0.3
            extent.y0 -= 0.1
            fig.savefig(save_path, bbox_inches=extent,
                        format='png', dpi=300, pad_inches=1)
        if plot:
            plt.show()

        return self.create_tf_image()

    def display_state(self, env_map, initial_state, state, plot=False):
        fig_size = 5.5
        fig, ax = plt.subplots(1, 2, figsize=[2 * fig_size, fig_size])
        ax_traj = ax[0]
        ax_bar = ax[1]

        value_step = 0.4 / initial_state.device_list.num_devices
        # Start with value of 200
        value_map = np.ones(env_map.get_size(), dtype=float)
        for device in initial_state.device_list.get_devices():
            value_map -= value_step * self.channel.total_shadow_map[device.position[1], device.position[0]]

        self.create_grid_image(ax=ax_traj, env_map=env_map, value_map=value_map)

        for device in initial_state.device_list.get_devices():
            ax_traj.add_patch(
                patches.Circle(np.array(device.position) + np.array((0.5, 0.5)), 0.4, facecolor=device.color,
                               edgecolor="black"))

        for battery in initial_state.battery_list.get_batterys():
            ax_traj.add_patch(
                patches.Ellipse(np.array(battery.position) + np.array((0.5, 0.5)), 0.4, 0.3, facecolor=battery.color,
                                edgecolor="black"))

        color = "green" if state.landed else "r"
        plt.scatter(state.position[0] + 0.5, state.position[1] + 0.5,
                    s=self.marker_size, marker="D", color=color, zorder=10)

        self.draw_bar_plots(state, ax_bar)

        if plot:
            plt.show()

        return self.create_tf_image()
