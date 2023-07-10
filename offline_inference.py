import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import cv2
import numpy as np
import json
import os
import random

from train.utils.misc import unnormalize, unnormalize_max_min, find_strings_with_substring
from train.utils.models import get_model
from train.utils.transforms import get_transforms
from train.utils.datasets import output_map, get_inputs_and_targets
from train.utils.vis_utils import Display

np.set_printoptions(precision=3)
pc_name = os.getlogin()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0


class TactileOfflineInferenceFinger:
    def __init__(self, model=None, model_params=None, transform=None, statistics=None):
        """
        Initialize the TactileOfflineInferenceFinger object.
        :param model: The trained model for inference.
        :param model_params: The parameters of the trained model.
        :param transform: The data transformation function.
        :param statistics: The statistics used for normalization.
        """
        self.model = model
        self.transform = transform
        self.statistics = statistics
        self.output_type = model_params['output']
        self.norm_method = model_params['norm_method']
        self.model_params = model_params
        self.display = Display(statistics=statistics, output_type=model_params['output'])

    def prepare_data(self, paths):
        """
        Prepare the data for offline inference.
        :param paths: The paths to the data files.
        :return: The prepared episodes and targets.
        """
        paths = [p.replace('roblab20', 'osher') for p in paths]
        episodes, targets = [], []

        for idx, p in enumerate(paths[:2]):
            df_data = pd.read_json(p).transpose()
            grouped = df_data.groupby(['theta', 'num'])

            for angle, group in grouped:
                episode_frames, _, target_frames = get_inputs_and_targets(group, self.output_type)
                episodes.append(episode_frames)
                targets.append(target_frames)

        return episodes, targets

    def offline_inference(self, samples=0):
        """
        Perform offline inference.
        :param samples: The number of samples to process.
        """
        blit = True
        is_touching = True

        episodes, targets = self.prepare_data(self.model_params['buffer_paths'])
        self.display.config_display(blit=blit)

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
                                            0.8, (255, 0, 0), 2, cv2.LINE_AA)

                self.display.update_display(y, target[i], blit=True)

                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Inference", raw_image)

                if cv2.waitKey(1) == 27:
                    break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    list_dirs = []
    path_to_dir = f'{os.path.dirname(__file__)}/train/train_history/'
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

    print('Loaded {} with output: {}'.format(model_params['input_type'], model_params['output']))
    path_to_model = path_to_dir + 'model.pth'
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    _, _, test_transform = get_transforms(int(model_params['image_size']))

    tactile = TactileOfflineInferenceFinger(model=model,
                                            model_params=model_params,
                                            transform=test_transform,
                                            statistics=statistics)

    tactile.offline_inference(samples=15)
