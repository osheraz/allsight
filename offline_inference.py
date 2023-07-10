import torch
import matplotlib

matplotlib.use('TkAgg')  # Use the 'TkAgg' backend
import matplotlib.pyplot as plt

plt.ion()
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0

import pandas as pd
import cv2
import numpy as np
from train.utils.misc import unnormalize, unnormalize_max_min
import json
from train.utils.models import get_model
from train.utils.vis_utils import data_for_finger_parametrized
from train.utils.surface import create_finger_geometry
from train.utils.transforms import get_transforms
from train.utils.geometry import convert_quat_xyzw_to_wxyz
from transformations import quaternion_matrix
from scipy import spatial
import os
import random
from train.utils.datasets import output_map, get_inputs_and_targets

np.set_printoptions(precision=3)  # to widen the printed array
pc_name = os.getlogin()


class TactileOfflineInferenceFinger:

    def __init__(self, model=None, model_params=None, transform=None, statistics=None):

        self.model = model
        self.transform = transform
        self.statistics = statistics
        self.output_type = model_params['output']
        self.norm_method = model_params['norm_method']
        self.finger_geometry = create_finger_geometry()
        self.tree = spatial.KDTree(self.finger_geometry[0])
        self.model_params = model_params

    def config_display(self, blit):

        plt.close('all')

        self.fig = plt.figure(figsize=(8, 4.4))
        self.ax1 = self.fig.add_subplot(1, 1, 1, projection='3d')

        self.ax1.autoscale(enable=True, axis='both', tight=True)
        self.ax1.set_xlim3d(self.statistics['min'][0], self.statistics['max'][0])
        self.ax1.set_ylim3d(self.statistics['min'][1], self.statistics['max'][1])
        self.ax1.set_zlim3d(self.statistics['min'][2], self.statistics['max'][2])

        self.ax1.tick_params(color='white')
        self.ax1.grid(False)
        mpl.rcParams['grid.color'] = 'white'
        self.ax1.set_facecolor('white')
        # First remove fill
        self.ax1.xaxis.pane.fill = False
        self.ax1.yaxis.pane.fill = False
        self.ax1.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        self.ax1.xaxis.pane.set_edgecolor('w')
        self.ax1.yaxis.pane.set_edgecolor('w')
        self.ax1.zaxis.pane.set_edgecolor('w')

        Xc, Yc, Zc = data_for_finger_parametrized(h=0.016, r=0.0128)

        self.ax1.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

        self.ax1.set_yticklabels([])
        self.ax1.set_xticklabels([])
        self.ax1.set_zticklabels([])

        self.pred_arrow, = self.ax1.plot3D([], [], [], color='black', linewidth=5, alpha=0.8)
        self.true_arrow, = self.ax1.plot3D([], [], [], color='red', linewidth=5, alpha=0.8)

        plt.tight_layout()

        self.fig.canvas.draw()
        if blit:
            # cache the background
            self.axbackground = self.fig.canvas.copy_from_bbox(self.ax1.bbox)

        plt.show(block=False)

    def update_display(self, y, target=None, blit=True):

        scale = 1500
        pred_pose = y[:3]
        pred_force = y[3:6]
        depth = round((5e-3 - y[-1]) * 1000, 2)
        torque = round(y[-2], 4)

        if 'torque' in self.output_type and 'depth' in self.output_type:
            self.fig.suptitle(f'\nForce: {y[3:6]} (N)'
                              f'\nPose: {y[:3] * 1000} (mm)'
                              f'\nTorsion: {torque} (Nm)'
                              f'\nDepth: {abs(depth)} (mm)',
                              fontsize=13)

        if target is not None:
            true_pose = target[:3]
            true_force = target[3:6]
            _, ind = self.tree.query(true_pose)
            cur_rot = self.finger_geometry[1][ind].copy()
            true_rot = quaternion_matrix(convert_quat_xyzw_to_wxyz(cur_rot))
            true_force_transformed = np.dot(true_rot[:3, :3], true_force)

            self.true_arrow.set_xdata(np.array([true_pose[0], true_pose[0] + true_force_transformed[0] / scale]))
            self.true_arrow.set_ydata(np.array([true_pose[1], true_pose[1] + true_force_transformed[1] / scale]))
            self.true_arrow.set_3d_properties(
                np.array([true_pose[2], true_pose[2] + true_force_transformed[2] / scale]))

        _, ind = self.tree.query(pred_pose)
        cur_rot = self.finger_geometry[1][ind].copy()
        pred_rot = quaternion_matrix(convert_quat_xyzw_to_wxyz(cur_rot))
        pred_force_transformed = np.dot(pred_rot[:3, :3], pred_force)

        self.pred_arrow.set_xdata(np.array([pred_pose[0], pred_pose[0] + pred_force_transformed[0] / scale]))
        self.pred_arrow.set_ydata(np.array([pred_pose[1], pred_pose[1] + pred_force_transformed[1] / scale]))
        self.pred_arrow.set_3d_properties(np.array([pred_pose[2], pred_pose[2] + pred_force_transformed[2] / scale]))

        if blit:
            self.fig.canvas.restore_region(self.axbackground)
            self.ax1.draw_artist(self.pred_arrow)
            self.ax1.draw_artist(self.true_arrow)
            self.fig.canvas.blit(self.ax1.bbox)
        else:
            self.fig.canvas.draw()

        self.fig.canvas.flush_events()

    def prepare_data(self, paths):

        paths = [p.replace('roblab20', 'osher') for p in paths]
        episodes, targets = [], []

        for idx, p in enumerate(paths):

            df_data = pd.read_json(p).transpose()

            # Group the dataset by angle
            grouped = df_data.groupby(['theta', 'num'])

            # Convert dataset to episodic format
            for angle, group in grouped:
                episode_frames, target_frames = get_inputs_and_targets(group, self.output_type)
                episodes.append(episode_frames)
                targets.append(target_frames)

        return episodes, targets

    def offline_inference(self, samples=0):

        blit = True
        is_touching = True

        episodes, targets = self.prepare_data(model_params['buffer_paths'])

        self.config_display(blit=blit)

        for s in range(samples):

            idx = random.randint(0, len(episodes))

            episode = episodes[idx]
            target = targets[idx]

            ref_frame = cv2.cvtColor(cv2.imread(episode[0]), cv2.COLOR_BGR2RGB)
            to_model_ref = torch.unsqueeze(self.transform(ref_frame).to(device), 0)

            for i in range(len(episode)):

                raw_image = cv2.imread(episode[i])

                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    to_model = torch.unsqueeze(self.transform(raw_image).to(device), 0)

                    if np.round(is_touching):
                        y = self.model(to_model, to_model_ref).cpu().detach().numpy()
                        if self.norm_method == 'meanstd':
                            y = unnormalize(y[0], self.statistics['mean'], self.statistics['std'])
                        elif self.norm_method == 'maxmin':
                            y = unnormalize_max_min(y[0], self.statistics['max'], self.statistics['min'])
                    else:
                        y = [0] * len(output_map[self.output_type])

                if 'pixel' in self.output_type:
                    IDX = 0 if self.output_type == 'pixel' else 6
                    px, py, pr = int(y[IDX]), int(y[IDX + 1]), max(0, int(y[IDX + 2]))
                    raw_image = cv2.circle(raw_image, (px, py), pr, (0, 0, 0), 1)
                    raw_image = cv2.putText(raw_image, f'Contact: {[px, py, pr]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8,
                                            (255, 0, 0), 2,
                                            cv2.LINE_AA)

                self.update_display(y, target[i], blit=True)

                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Inference", raw_image)

                if cv2.waitKey(1) == 27:
                    break

        cv2.destroyAllWindows()


def find_strings_with_substring(string_list, substring):
    result = []
    for string in string_list:
        if substring in string and 'fintune' not in string:
            result.append(string)
    return result


if __name__ == "__main__":

    # warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu

    list_dirs = []
    path_to_dir = '/src/allsight/train/train_history/'

    for path_to_dir, dirs, files in os.walk(path_to_dir):
        for subdir in dirs:
            list_dirs.append(os.path.join(path_to_dir, subdir))
    model_name = 'train_pose_force_pixel_torque_depth_resnet18_sim_pre_6c_aug_meanstd_08-07-2023_19-19-13'

    path_to_dir = find_strings_with_substring(list_dirs, model_name)[0] + '/'

    # Load data statistics and model params
    with open(path_to_dir + "data_statistic.json", 'rb') as handle:
        statistics = json.load(handle)

    with open(path_to_dir + 'model_params.json', 'rb') as json_file:
        model_params = json.load(json_file)

    model = get_model(model_params)

    print('loaded {} with output: {}'.format(model_params['input_type'], model_params['output']))
    path_to_model = path_to_dir + 'model.pth'
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    _, _, test_transform = get_transforms(int(model_params['image_size']))

    device_id = 1 if pc_name == 'roblab20' else 4

    tactile = TactileOfflineInferenceFinger(model=model,
                                            model_params=model_params,
                                            transform=test_transform,
                                            statistics=statistics)

    tactile.offline_inference(samples=15)
