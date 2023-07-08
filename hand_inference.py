import time
import torch
import cv2
import numpy as np
from train.utils.misc import unnormalize
import json
from train.utils.models import get_model
from train.utils.transforms import get_transforms
from tactile_finger.src.envs.finger import Finger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.ion()
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0


class Hand:

    def __init__(self, dev_names, model, model_params, test_transform, statistics):

        self.transform = test_transform
        self.model = model
        self.model_params = model_params
        self.statistics = statistics

        self.finger_right = Finger(dev_name=dev_names[0], serial='/dev/video', fix=(-7, 0))  # (-7, 0)
        self.finger_left = Finger(dev_name=dev_names[1], serial='/dev/video', fix=(0, -15))  # (10,5)
        self.finger_bottom = Finger(dev_name=dev_names[2], serial='/dev/video', fix=(8, 3))  # (-5,-10)

        self.frames_right = []
        self.frames_left = []
        self.frames_bottom = []
        self.timesteps = []
        self.outputs_right = []
        self.outputs_left = []
        self.outputs_bottom = []
        self.ext_frames = []

    def init_hand(self):
        """
        Sets stream resolution based on supported streams in Finger.STREAMS
        :param resolution: QVGA or VGA from Finger.STREAMS
        :return: None
        """
        self.finger_right.connect()
        self.finger_left.connect()
        self.finger_bottom.connect()

    def get_frames(self):
        """
        Returns a single image frame for the device
        :param transpose: Show direct output from the image sensor, WxH instead of HxW
        :return: Image frame array
        """
        frame_left = self.finger_left.get_frame()
        frame_right = self.finger_right.get_frame()
        frame_bottom = self.finger_bottom.get_frame()

        return frame_left, frame_right, frame_bottom

    def show_fingers_view(self):
        """
        Creates OpenCV named window with live view of Finger device, ESC to close window
        :param ref_frame: Specify reference frame to show image difference
        :return: None
        """

        while True:

            left, right, bottom = self.get_frames()

            cv2.imshow("Hand View", np.concatenate((left, right, bottom), axis=1))

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    def init_external_cam(self):
        """
        Sets stream resolution based on supported streams in Finger.STREAMS
        :param resolution: QVGA or VGA from Finger.STREAMS
        :return: None
        """
        self.ext_cam = cv2.VideoCapture(10)
        if not self.ext_cam.isOpened():
            print("Cannot open video capture device ext cam")
            raise Exception("Error opening video stream: ext_cam")

    def wait_env_ready(self):

        import time
        import sys
        for i in range(5):
            print("WAITING..." + str(i))
            sys.stdout.flush()
            time.sleep(1.0)

        print("WAITING...DONE")

    def hand_inference(self):

        self.init_external_cam()

        how_long = 30
        start_time = time.time()
        save_every_sec = 0.05
        cur_time = time.time() - start_time
        save_time = time.time() - start_time

        left, right, bottom = self.get_frames()
        left_clear_ref, right_clear_ref, bottom_clear_ref = left.copy(), right.copy(), bottom.copy()

        left_ref = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        left_ref = torch.unsqueeze(self.transform(left_ref).to(device), 0)

        right_ref = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        right_ref = torch.unsqueeze(self.transform(right_ref).to(device), 0)

        bottom_ref = cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB)
        bottom_ref = torch.unsqueeze(self.transform(bottom_ref).to(device), 0)
        self.wait_env_ready()

        while cur_time < how_long:

            left_raw, right_raw, bottom_raw = self.get_frames()
            left_clear, right_clear, bottom_clear = left_raw.copy(), right_raw.copy(), bottom_raw.copy()

            cur_time = time.time() - start_time
            print(time.time() - start_time)

            left = cv2.cvtColor(left_raw, cv2.COLOR_BGR2RGB)
            left = torch.unsqueeze(self.transform(left).to(device), 0)

            right = cv2.cvtColor(right_raw, cv2.COLOR_BGR2RGB)
            right = torch.unsqueeze(self.transform(right).to(device), 0)

            bottom = cv2.cvtColor(bottom_raw, cv2.COLOR_BGR2RGB)
            bottom = torch.unsqueeze(self.transform(bottom).to(device), 0)

            with torch.no_grad():

                y_right = self.model(right, right_ref).cpu().detach().numpy()
                y_right = unnormalize(y_right[0], self.statistics['mean'], self.statistics['std'])

                y_left = self.model(left, left_ref).cpu().detach().numpy()
                y_left = unnormalize(y_left[0], self.statistics['mean'], self.statistics['std'])

                y_bottom = self.model(bottom, bottom_ref).cpu().detach().numpy()
                y_bottom = unnormalize(y_bottom[0], self.statistics['mean'], self.statistics['std'])

            px_right, py_right, pr_right = int(y_right[6]), int(y_right[6 + 1]), max(0, int(y_right[6 + 2]))
            px_left, py_left, pr_left = int(y_left[6]), int(y_left[6 + 1]), max(0, int(y_left[6 + 2]))
            px_bottom, py_bottom, pr_bottom = int(y_bottom[6]), int(y_bottom[6 + 1]), max(0, int(y_bottom[6 + 2]))

            right_w_circ = cv2.circle(right_raw, (px_right, py_right), pr_right, (0, 0, 0), 2)
            right_w_circ = cv2.putText(right_w_circ, f'Contact: {[px_right, py_right, pr_right]}', (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                       (255, 0, 0), 2,
                                       cv2.LINE_AA)

            left_w_circ = cv2.circle(left_raw, (px_left, py_left), pr_left, (0, 0, 0), 2)
            left_w_circ = cv2.putText(left_w_circ, f'Contact: {[px_left, py_left, pr_left]}', (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                      (255, 0, 0), 2,
                                      cv2.LINE_AA)

            bottom_w_circ = cv2.circle(bottom_raw, (px_bottom, py_bottom), pr_bottom, (0, 0, 0), 2)
            bottom_w_circ = cv2.putText(bottom_w_circ, f'Contact: {[px_bottom, py_bottom, pr_bottom]}', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (255, 0, 0), 2,
                                        cv2.LINE_AA)

            if cur_time - save_time > save_every_sec:
                save_time = cur_time
                _, ext_img = self.ext_cam.read()
                cv2.imshow("Ext Cam View", ext_img)
                cv2.imshow("Hand View", np.concatenate((left_w_circ,
                                                        right_w_circ,
                                                        bottom_w_circ),
                                                       axis=1))

                self.frames_right.append(right_clear)
                self.outputs_right.append(y_right)
                self.frames_left.append(left_clear)
                self.outputs_left.append(y_left)
                self.frames_bottom.append(bottom_clear)
                self.outputs_bottom.append(y_bottom)
                self.timesteps.append(time.time() - start_time)
                self.ext_frames.append(ext_img)

                if cv2.waitKey(1) == 27:
                    break

        save = input("Save output ? [true/false]\n").lower().strip()
        if save == "true":
            import pickle

            dict_to_save = {'frames_right': self.frames_right,
                            'frames_left': self.frames_left,
                            'frames_bottom': self.frames_bottom,
                            'ext_frames': self.ext_frames,
                            'time': self.timesteps,
                            'outputs_right': self.outputs_right,
                            'outputs_left': self.outputs_left,
                            'outputs_bottom': self.outputs_bottom,
                            'left_ref': left_clear_ref,
                            'right_ref': right_clear_ref,
                            'bottom_ref': bottom_clear_ref,
                            }

            file_name = input("enter file_name \n").lower().strip()

            with open(f'/home/osher/catkin_ws/src/allsight/train/records/{file_name}.pickle', 'wb') as handle:
                pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import os

    pc_name = os.getlogin()

    with_classifer = False
    leds = 'rrrgggbbb'
    gel = 'markers'

    path_to_dir = f'/home/{pc_name}/catkin_ws/src/allsight/train/train_history/{gel}/{leds}/'
    model_name = 'train_pose_force_pixel_torque_depth_resnet18_sim_pre_6c_aug_meanstd_04-07-2023_20-06-32'
    path_to_dir += model_name + '/'

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

    tactile = Hand(dev_names=[4, 6, 8],
                   model=model,
                   model_params=model_params,
                   test_transform=test_transform,
                   statistics=statistics)

    tactile.init_hand()

    # tactile.show_fingers_view()
    tactile.hand_inference()
