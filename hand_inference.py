import torch
import cv2
import numpy as np
import json
import os

from train.utils.misc import unnormalize
from train.utils.models import get_model
from train.utils.transforms import get_transforms
from tactile_finger.src.envs.finger import Finger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')
mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0


class Hand:
    def __init__(self, dev_names, model, model_params, test_transform, statistics):
        """
        Initializes the Hand object.

        Parameters:
        - dev_names (list): List of device names corresponding to the fingers.
        - model (torch.nn.Module): Pre-trained model for inference.
        - model_params (dict): Parameters of the pre-trained model.
        - test_transform (torchvision.transforms): Image transformation for testing.
        - statistics (dict): Statistics for normalizing the predicted outputs.
        """
        self.transform = test_transform
        self.model = model
        self.model_params = model_params
        self.statistics = statistics

        self.fingers = []

        for dev_name, fix in zip(dev_names, [(-7, 0), (0, -15), (8, 3)]):
            finger = Finger(dev_name=dev_name, serial='/dev/video', fix=fix)
            finger.connect()
            self.fingers.append(finger)

    def get_frames(self):
        """
        Retrieves frames captured by the fingers.

        Returns:
        - frames (tuple): Tuple of captured frames.
        """
        frames = []
        for finger in self.fingers:
            frame = finger.get_frame()
            frames.append(frame)
        return tuple(frames)

    def show_fingers_view(self):
        """
        Displays the concatenated frames of all fingers as a single image.
        """
        while True:
            frames = self.get_frames()
            cv2.imshow("Hand View", np.concatenate(frames, axis=1))
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def hand_inference(self):
        """
        Performs inference on the captured frames using the pre-trained model.
        """
        left_ref, right_ref, bottom_ref = self.get_frames()

        while True:
            left_raw, right_raw, bottom_raw = self.get_frames()

            with torch.no_grad():
                y_right = self.model(right_raw, right_ref).cpu().detach().numpy()
                y_right = unnormalize(y_right[0], self.statistics['mean'], self.statistics['std'])

                y_left = self.model(left_raw, left_ref).cpu().detach().numpy()
                y_left = unnormalize(y_left[0], self.statistics['mean'], self.statistics['std'])

                y_bottom = self.model(bottom_raw, bottom_ref).cpu().detach().numpy()
                y_bottom = unnormalize(y_bottom[0], self.statistics['mean'], self.statistics['std'])

            px_right, py_right, pr_right = int(y_right[6]), int(y_right[7]), max(0, int(y_right[8]))
            px_left, py_left, pr_left = int(y_left[6]), int(y_left[7]), max(0, int(y_left[8]))
            px_bottom, py_bottom, pr_bottom = int(y_bottom[6]), int(y_bottom[7]), max(0, int(y_bottom[8]))

            right_w_circ = cv2.circle(right_raw, (px_right, py_right), pr_right, (0, 0, 0), 2)
            right_w_circ = cv2.putText(right_w_circ, f'Contact: {[px_right, py_right, pr_right]}', (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            left_w_circ = cv2.circle(left_raw, (px_left, py_left), pr_left, (0, 0, 0), 2)
            left_w_circ = cv2.putText(left_w_circ, f'Contact: {[px_left, py_left, pr_left]}', (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            bottom_w_circ = cv2.circle(bottom_raw, (px_bottom, py_bottom), pr_bottom, (0, 0, 0), 2)
            bottom_w_circ = cv2.putText(bottom_w_circ, f'Contact: {[px_bottom, py_bottom, pr_bottom]}', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("Hand View", np.concatenate((left_w_circ, right_w_circ, bottom_w_circ), axis=1))

            if cv2.waitKey(1) == 27:
                break


if __name__ == "__main__":

    pc_name = os.getlogin()
    leds = 'rrrgggbbb'
    gel = 'markers'

    path_to_dir = f'{os.path.dirname(__file__)}/train/train_history/{gel}/{leds}/'
    model_name = 'train_pose_force_pixel_torque_depth_resnet18_sim_pre_6c_aug_meanstd_08-07-2023_19-19-13'
    path_to_dir += model_name + '/'

    with open(path_to_dir + "data_statistic.json", 'rb') as handle:
        statistics = json.load(handle)

    with open(path_to_dir + 'model_params.json', 'rb') as json_file:
        model_params = json.load(json_file)

    model = get_model(model_params)
    print(f"Loaded {model_params['input_type']} with output: {model_params['output']}")

    path_to_model = path_to_dir + 'model.pth'
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    _, _, test_transform = get_transforms(int(model_params['image_size']))

    tactile = Hand(dev_names=[4, 6, 8], model=model, model_params=model_params,
                   test_transform=test_transform, statistics=statistics)

    tactile.hand_inference()
