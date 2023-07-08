import time
import torch
import matplotlib

matplotlib.use('TkAgg')  # Use the 'TkAgg' backend
import matplotlib.pyplot as plt

plt.ion()
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0

import cv2
import numpy as np
from src.allsight.train.utils.misc import unnormalize, unnormalize_max_min
import json
from src.allsight.train.utils.models import PreTrainedModel, get_model
from src.allsight.train.utils.datasets import output_map
from src.allsight.train.utils.vis_utils import data_for_finger_parametrized
from src.allsight.train.utils.surface import create_finger_geometry
from src.allsight.train.utils.transforms import get_transforms
from src.allsight.train.utils.geometry import T_inv, convert_quat_wxyz_to_xyzw, convert_quat_xyzw_to_wxyz
from transformations import quaternion_matrix
from scipy import spatial
import os
import subprocess

from src.allsight.tactile_finger.src.envs.finger import Finger

np.set_printoptions(precision=3)  # to widen the printed array
pc_name = os.getlogin()


class TactileInferenceFinger(Finger):

    def __init__(self,
                 serial=None,
                 dev_name=None,
                 model=None,
                 model_params=None,
                 classifier=None,
                 transform=None,
                 statistics=None):

        super().__init__(serial, dev_name)

        self.model = model
        self.touch_classifier = classifier
        self.transform = transform
        self.statistics = statistics
        self.output_type = model_params['output']
        self.norm_method = model_params['norm_method']
        self.finger_geometry = create_finger_geometry()
        self.tree = spatial.KDTree(self.finger_geometry[0])
        self.frame = np.zeros((480, 480, 3), dtype=np.uint8)

    def config_display(self, blit):

        plt.close('all')

        self.fig = plt.figure(figsize=(12, 6))

        self.ax1 = self.fig.add_subplot(1, 3, 2, projection='3d')

        self.ax1.autoscale(enable=True, axis='both', tight=True)
        self.ax1.set_xlim3d(self.statistics['min'][0], self.statistics['max'][0])
        self.ax1.set_ylim3d(self.statistics['min'][1], self.statistics['max'][1])
        self.ax1.set_zlim3d(self.statistics['min'][2], self.statistics['max'][2])

        self.ax1.tick_params(color='white')
        self.ax1.grid(False)
        mpl.rcParams['grid.color'] = 'white'
        self.ax1.set_facecolor('white')
        self.ax1.xaxis.pane.fill = False
        self.ax1.yaxis.pane.fill = False
        self.ax1.zaxis.pane.fill = False

        self.ax1.xaxis.pane.set_edgecolor('w')
        self.ax1.yaxis.pane.set_edgecolor('w')
        self.ax1.zaxis.pane.set_edgecolor('w')

        Xc, Yc, Zc = data_for_finger_parametrized(h=0.016, r=0.0125)

        self.ax1.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

        self.ax1.set_yticklabels([])
        self.ax1.set_xticklabels([])
        self.ax1.set_zticklabels([])

        self.arrow, = self.ax1.plot3D([], [], [], color='black', linewidth=5, alpha=0.8)

        plt.tight_layout()
        self.fig.canvas.draw()

        if blit:
            # cache the background
            self.axbackground = self.fig.canvas.copy_from_bbox(self.ax1.bbox)

        plt.show(block=False)

    def update_display(self, y, blit=True):
        scale = 1500

        if 'torque' in self.output_type and 'depth' in self.output_type:
            depth = round((5e-3 - y[-1]) * 1000, 2)
            torque = round(y[-2], 4)
            self.fig.suptitle(f'\nForce: {y[3:6]} (N)'
                              f'\nPose: {y[:3] * 1000} (mm)'
                              f'\nTorsion: {torque} (Nm)'
                              f'\nDepth: {abs(depth)} (mm)',
                              fontsize=13)

            pred_pose = y[:3]
            pred_force = y[3:6]
            _, ind = self.tree.query(pred_pose)
            cur_rot = self.finger_geometry[1][ind].copy()
            pred_rot = quaternion_matrix(convert_quat_xyzw_to_wxyz(cur_rot))
            pred_force_transformed = np.dot(pred_rot[:3, :3], pred_force)

            self.arrow.set_xdata(np.array([pred_pose[0], pred_pose[0] + pred_force_transformed[0] / scale]))
            self.arrow.set_ydata(np.array([pred_pose[1], pred_pose[1] + pred_force_transformed[1] / scale]))
            self.arrow.set_3d_properties(np.array([pred_pose[2], pred_pose[2] + pred_force_transformed[2] / scale]))

            if blit:
                self.fig.canvas.restore_region(self.axbackground)
                self.ax1.draw_artist(self.arrow)
                self.fig.canvas.blit(self.ax1.bbox)
            else:
                self.fig.canvas.draw()

            self.fig.canvas.flush_events()

    def inference(self, ref_frame=None, display_pixel=True):
        """
        :return: None
        """
        blit = True
        is_touching = True
        self.config_display(blit=blit)

        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        to_model_ref = torch.unsqueeze(self.transform(ref_frame).to(device), 0)

        while True:

            raw_image = self.get_frame()
            frame = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

            with torch.no_grad():

                to_model = torch.unsqueeze(self.transform(frame).to(device), 0)

                if self.touch_classifier is not None:
                    is_touching = (self.touch_classifier(to_model).squeeze()).cpu().detach().numpy()

                if np.round(is_touching):
                    y = self.model(to_model, to_model_ref).cpu().detach().numpy()
                    if self.norm_method == 'meanstd':
                        y = unnormalize(y[0], self.statistics['mean'], self.statistics['std'])
                    elif self.norm_method == 'maxmin':
                        y = unnormalize_max_min(y[0], self.statistics['max'], self.statistics['min'])
                else:
                    y = [0] * len(output_map[self.output_type])

            if display_pixel:
                IDX = 0 if self.output_type == 'pixel' else 6
                px, py, pr = int(y[IDX]), int(y[IDX + 1]), max(0, int(y[IDX + 2]))
                frame = cv2.circle(frame, (px, py), pr, (0, 0, 0), 2)
                frame = cv2.putText(frame, f'Contact: {[px, py, pr]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 0, 0), 2,
                                    cv2.LINE_AA)

            self.update_display(y)

            cv2.imshow("Inference", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":

    # warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu

    with_classifier = False
    leds = 'rrrgggbbb'
    gel = 'markers'
    model_name = 'train_pose_force_pixel_torque_depth_resnet18_sim_pre_6c_aug_meanstd_04-07-2023_20-06-32'
    classifier_name = 'train_touch_detect_det_13-02-2023_16-39-47'

    path_to_dir = f'{os.path.dirname(__file__)}/train/train_history/{gel}/{leds}/'

    # Load data statistics and model params
    with open(path_to_dir + model_name + "/data_statistic.json", 'rb') as handle:
        statistics = json.load(handle)

    with open(path_to_dir + model_name + '/model_params.json', 'rb') as json_file:
        model_params = json.load(json_file)

    model = get_model(model_params)

    print('loaded {} with output: {}'.format(model_params['input_type'], model_params['output']))
    model.load_state_dict(torch.load(path_to_dir + model_name + '/model.pth'))
    model.eval()

    if with_classifier:
        with open(path_to_dir + classifier_name + '/model_params.json', 'rb') as json_file:
            model_params_classify = json.load(json_file)

        touch_classifier = PreTrainedModel(model_name=model_params_classify['model_name'],
                                           num_output=output_map[model_params_classify['output']],
                                           classifier=True).to(device)
        touch_classifier.load_state_dict(torch.load(path_to_dir + classifier_name + '/model.pth'))
        touch_classifier.eval()
    else:
        touch_classifier = None

    _, _, test_transform = get_transforms(int(model_params['image_size']))

    device_id = 0 if pc_name == 'roblab20' else 4

    tactile = TactileInferenceFinger(serial='/dev/video',
                                     dev_name=device_id,
                                     model=model,
                                     model_params=model_params,
                                     classifier=touch_classifier,
                                     transform=test_transform,
                                     statistics=statistics)

    tactile.connect()
    tactile.inference(ref_frame=tactile.get_frame())
