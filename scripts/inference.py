import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import cv2
import numpy as np
import json

from train.utils.misc import unnormalize, unnormalize_max_min
from train.utils.models import PreTrainedModel, get_model
from train.utils.datasets import output_map
from train.utils.transforms import get_transforms
from train.utils.vis_utils import Display
from tactile_finger.src.envs.finger import Finger

np.set_printoptions(precision=3)
pc_name = os.getlogin()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0


class TactileInferenceFinger(Finger):
    """
    Class for performing tactile inference using a trained model.

    Args:
        serial (str): Serial port for communication.
        dev_name (int): Device name.
        model (nn.Module): Trained model for tactile inference.
        model_params (dict): Parameters of the trained model.
        classifier (nn.Module, optional): Touch classifier model. Defaults to None.
        transform (nn.Module, optional): Image transformation module. Defaults to None.
        statistics (dict, optional): Data statistics for normalization. Defaults to None.
    """

    def __init__(self, serial=None, dev_name=None, model=None, model_params=None,
                 classifier=None, transform=None, statistics=None):
        super().__init__(serial, dev_name)
        self.model = model
        self.touch_classifier = classifier
        self.transform = transform
        self.statistics = statistics
        self.output_type = model_params['output']
        self.norm_method = model_params['norm_method']
        self.display = Display(statistics=statistics, output_type=model_params['output'])

    def inference(self, ref_frame=None, display_pixel=True):
        """
        Perform tactile inference using the trained model.

        Args:
            ref_frame (numpy.ndarray): Reference frame for comparison.
            display_pixel (bool): Whether to display the pixel information. Defaults to True.
        """

        blit = True
        is_touching = True
        self.display.config_display(blit=blit)

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
                                    (255, 0, 0), 2, cv2.LINE_AA)

            self.display.update_display(y)

            cv2.imshow("Inference", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    with_classifier = False
    leds = 'rrrgggbbb'
    gel = 'markers'
    model_name = 'train_pose_force_pixel_torque_depth_resnet18_sim_pre_6c_aug_meanstd_08-07-2023_19-19-13'
    classifier_name = 'train_touch_detect_det_13-02-2023_16-39-47'

    path_to_dir = f'{os.path.dirname(__file__)}/train/train_history/{gel}/{leds}/'

    # Load data statistics and model params
    with open(path_to_dir + model_name + "/data_statistic.json", 'rb') as handle:
        statistics = json.load(handle)

    with open(path_to_dir + model_name + '/model_params.json', 'rb') as json_file:
        model_params = json.load(json_file)

    model = get_model(model_params)

    print('Loaded {} with output: {}'.format(model_params['input_type'], model_params['output']))
    model.load_state_dict(torch.load(path_to_dir + model_name + '/model.pth'))
    model.eval()

    touch_classifier = None
    if with_classifier:
        with open(path_to_dir + classifier_name + '/model_params.json', 'rb') as json_file:
            model_params_classify = json.load(json_file)

        touch_classifier = PreTrainedModel(model_name=model_params_classify['model_name'],
                                           num_output=output_map[model_params_classify['output']],
                                           classifier=True).to(device)
        touch_classifier.load_state_dict(torch.load(path_to_dir + classifier_name + '/model.pth'))
        touch_classifier.eval()

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
