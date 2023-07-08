import time
import os
import json
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from utils.misc import normalize, unnormalize, normalize_max_min, unnormalize_max_min, save_df_as_json
from utils.vis_utils import Arrow3D
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR  # Learning rate schedulers
from utils.models import PreTrainedModel, PreTrainedModelWithRef, PreTrainedModelWithMask, PreTrainedModelWithOnlineMask
from utils.vis_utils import data_for_cylinder_along_z, data_for_sphere_along_z, set_axes_equal
from utils.datasets import TactileDataset, output_map, sensor_dict, get_buffer_paths
import utils.contrib.resnets as srn
from glob import glob
from utils.surface import create_finger_geometry
from utils.geometry import convert_quat_wxyz_to_xyzw, convert_quat_xyzw_to_wxyz
from transformations import quaternion_matrix
from scipy import spatial
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
np.set_printoptions(suppress=True, linewidth=np.inf)  # to widen the printed array

pc_name = os.getlogin()


class Trainer(object):

    def __init__(self, params):

        # Get params, create logger
        self.params = params

        # Set random seeds
        # seed = self.params['seed']
        # torch.manual_seed(seed)
        # np.random.seed(seed)

        leds = params['leds']
        gel = params['gel']
        sensor_id = params['sensor_id']
        indenter = ['sphere3', 'sphere4', 'sphere5']

        buffer_paths_to_train, buffer_paths_to_test, sensor_id, _ = get_buffer_paths(leds, gel, indenter, sensor_id)

        #####################
        ## SET AGENT PARAMS
        #####################

        self.model_params = {
            'learning_rate': params['learning_rate'],
            'device': params['device'].type,
            'deterministic': params['deterministic'],
            'batch_size': params['batch_size'],
            'image_size': params['image_size'],
            'display_step': params['display_step'],
            'epoch': params['epoch'],
            'optimizer': "Adam",
            'dropout': params['dropout'],
            'portion': params['portion'],
            'logdir': params['logdir'],
            'scheduler': params['scheduler'],
            'model_name': params['model_name'],
            'buffer_paths': buffer_paths_to_train,
            'output': params['output'],
            'norm_method': params['norm_method'],
            'aug': params['aug'],
            'input_type': params['input_type'],
            'train_size': params['train_size'],
            'leds': leds,
            'gel': gel,
            'sensor_id': sensor_id,
            'indenter': indenter,
        }

        self.finger_geometry = create_finger_geometry()
        self.tree = spatial.KDTree(self.finger_geometry[0])
        self.prepare_data(buffer_paths_to_train, buffer_paths_to_test, params['output'])

        if params['input_type'] == 'single':
            self.model = PreTrainedModel(params['model_name'], output_map[params['output']]).to(device)
        elif params['input_type'] == 'with_ref_6c':
            self.model = PreTrainedModelWithRef(params['model_name'], output_map[params['output']]).to(device)
        elif params['input_type'] == 'with_mask_7c':
            self.model = PreTrainedModelWithMask(params['model_name'], output_map[params['output']]).to(device)
        elif params['input_type'] == 'with_mask_8c':
            self.model = PreTrainedModelWithOnlineMask(params['model_name'], output_map[params['output']]).to(device)
        elif params['input_type'] == 'with_ref_rnn':
            self.model = srn.resnet18(False, False, num_classes=output_map[params['output']]).to(device)
        else:
            assert 'which model you want to use?'

        with open(self.params['logdir'] + '/model_params.json', 'w') as fp:
            dic_items = self.model_params.items()
            new_dict = {key: value for key, value in dic_items}
            json.dump(new_dict, fp, indent=3)

        with open(self.params['logdir'] + '/data_statistic.json', 'w') as fp:
            dic_items = self.originalset.data_statistics.items()
            new_dict = {key: value.tolist() for key, value in dic_items}
            json.dump(new_dict, fp, indent=3)

        # self.optimizer = getattr(torch.optim, self.model_params['optimizer'])(self.model.parameters(),
        #                                                                       params['learning_rate'])
        # weight_decay=0.0 )
        self.optimizer = getattr(torch.optim, self.model_params['optimizer'])(self.model.parameters(),
                                                                              lr=params['learning_rate'])
        # betas=(0.94, 0.96),
        # weight_decay=0.008)

        # torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

        if params['scheduler'] == 'reduce':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        elif params['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=4, eta_min=1e-8)
        elif params['scheduler'] == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.8)
        else:
            self.scheduler = None

        self.fig = plt.figure(figsize=(20, 15))

    def prepare_data(self, paths, test_paths, output_type):

        for idx, p in enumerate(paths):
            if idx == 0:
                df_data = pd.read_json(p).transpose()
            else:
                df_data = pd.concat([df_data, pd.read_json(p).transpose()], axis=0)

        df_data = df_data[df_data.time > 2.0]  # train only over touching samples!

        # for idx, p in enumerate(test_paths):
        #     if idx == 0:
        #         df_data_val = pd.read_json(p).transpose()
        #     else:
        #         df_data_val = pd.concat([df_data_val, pd.read_json(p).transpose()], axis=0)
        #
        # df_data_val = df_data_val[df_data_val.time > 1.0]  # train only over touching samples!

        train_df, remain_df = train_test_split(df_data, test_size=0.1, shuffle=True)
        valid_df, test_df = train_test_split(remain_df, test_size=0.5, shuffle=True)

        # Sample data portion to train
        train_df = train_df.sample(n=int(3000 * self.model_params['portion']))
        # valid_df = valid_df.sample(n=int(self.model_params['train_size'] * 0.3))
        # test_df = test_df.sample(n=int(self.model_params['train_size'] * 0.3))

        # save_df_as_json(train_df, self.params['logdir'] + '/', 'train_df')
        # save_df_as_json(valid_df, self.params['logdir'] + '/', 'valid_df')
        # save_df_as_json(test_df, self.params['logdir'] + '/', 'test_df')

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.params['image_size'], self.params['image_size'])),
            # transforms.RandomChoice([
            #     # transforms.RandomRotation(3),  # rotate +/- 10 degrees
            #     transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            #     # transforms.ColorJitter(brightness=1.0, contrast=0.1, saturation=0.5, hue=0.1),
            # ]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.params['image_size'], self.params['image_size'])),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            # transforms.RandomChoice([
            #     transforms.RandomAdjustSharpness(2),
            #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            #
            # ]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.params['image_size'], self.params['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.originalset = TactileDataset(self.model_params, train_df, output_type, self.train_transform,
                                          apply_mask=False)

        if self.params['aug']:
            self.augset = TactileDataset(self.model_params, train_df, output_type, self.aug_transform, apply_mask=False)
            self.trainset = torch.utils.data.ConcatDataset([self.originalset, self.augset])
        else:
            self.trainset = self.originalset

        self.validset = TactileDataset(self.model_params, valid_df, output_type, self.test_transform, apply_mask=False)
        self.testset = TactileDataset(self.model_params, test_df, output_type, self.test_transform, apply_mask=False)

        self.trainloader = DataLoader(self.trainset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        self.validloader = DataLoader(self.validset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        self.testloader = DataLoader(self.testset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)

        print(f'Train set length is {len(self.trainset)}')
        print(f'Valid set length is {len(self.validset)}')
        print(f'Test set length is {len(self.testset)}')

    def run_training_loop(self):

        epochs = self.model_params['epoch']
        # init vars at beginning of training
        self.start_time = time.time()
        self.min_valid_loss = np.inf
        mean_train_loss = np.inf

        COSTS, EVAL_COSTS, epoch_cost, eval_cost = [], [], [], []
        BATCH_SIZE = self.model_params['batch_size']

        for epoch in range(epochs):

            self.model.train()
            with tqdm(self.trainloader, unit="batch") as tepoch:
                for (batch_x, batch_x_ref, batch_masked_img, batch_masked_ref, batch_y, batch_y_ref) in tepoch:
                    tepoch.set_description(f"Epoch [{epoch}/{epochs}]")

                    # for b, (batch_x, batch_x_ref, batch_diff, batch_y, batch_y_ref) in enumerate(self.trainloader):

                    loss = nn.functional.mse_loss(self.model(batch_x, batch_x_ref, batch_masked_img, batch_masked_ref).to(device),
                                                  batch_y.to(device))

                    self.optimizer.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_norm_(list(self.model.parameters()), 0.5)
                    self.optimizer.step()
                    cost = loss.item()
                    COSTS.append(cost)

                    torch.cuda.empty_cache()
                    # log/save
                    # if b % 10 == 0:
                    #     print(f'epoch: {epoch:2}  batch: {b:4} [{BATCH_SIZE * b:6}/{len(self.trainset)}]  train loss: {cost:5.5f}')
                    tepoch.set_postfix(loss=cost, last_train_loss=mean_train_loss)

            mean_train_loss = np.mean(COSTS[-len(self.trainloader):])
            # print('Epoch train loss : ' + str(mean_train_loss))

            self.log_model_predictions(batch_x, batch_x_ref,batch_masked_img, batch_masked_ref, batch_y, 'train')

            EVAL_COSTS = self.run_validation_loop(EVAL_COSTS)
            mean_val_loss = np.mean(EVAL_COSTS[-len(self.validloader):])

            epoch_cost.append(mean_train_loss)
            eval_cost.append(mean_val_loss)

            # apply LR scheduler after each epoch
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(mean_val_loss)

            elif isinstance(self.scheduler, CosineAnnealingLR) or isinstance(self.scheduler, StepLR):
                self.scheduler.step()

            # Lets log a bit
            self.fig.clf()
            plt.plot(epoch_cost, '-ro', linewidth=3, label='train loss')
            plt.plot(eval_cost, '-ko', linewidth=3, label='val loss')
            plt.legend()
            self.fig.savefig(self.params['logdir'] + '/train_val_comp.png', dpi=200, bbox_inches='tight')

        # print("Optimization Finished!")

        np.save(self.params['logdir'] + '/train_val_comp.npy', [epoch_cost, eval_cost])
        self.fig.clf()
        plt.plot(epoch_cost, '-ro', linewidth=3, label='train loss')
        plt.plot(eval_cost, '-ko', linewidth=3, label='val loss')
        plt.legend()
        self.fig.savefig(self.params['logdir'] + '/train_val_comp.png', dpi=200, bbox_inches='tight')

    def run_validation_loop(self, EVAL_COSTS):

        self.model.eval()
        # BATCH_SIZE = self.model_params['batch_size']
        # print("\nLets validate")

        with tqdm(self.validloader, unit="batch") as tepoch:
            for (batch_x, batch_x_ref, batch_masked_img, batch_masked_ref, batch_y, batch_y_ref) in tepoch:
                tepoch.set_description("Validate")

                with torch.no_grad():
                    pred_px = self.model(batch_x, batch_x_ref, batch_masked_img, batch_masked_ref).to(device)
                    true_px = batch_y.to(device)
                    cost = nn.functional.mse_loss(pred_px, true_px)

                # for b, (batch_x, batch_x_ref, batch_diff, batch_y, batch_y_ref) in enumerate(self.validloader):
                #     with torch.no_grad():
                #         pred_px = self.model(batch_x, batch_x_ref).to(device)
                #         true_px = batch_y.to(device)
                #         cost = nn.functional.mse_loss(pred_px, true_px)

                EVAL_COSTS.append(cost.item())
                tepoch.set_postfix(loss=cost.item(), min_valid_loss=self.min_valid_loss)

                # print(f'epoch: {b:2}  batch: {b:4} [{BATCH_SIZE * b:6}/{len(self.validset)}]  test loss: {cost.item():10.8f}')

        mean_curr_valid_loss = np.mean(EVAL_COSTS[-len(self.validloader):])
        # print('\nValidation loss : ' + str(mean_curr_valid_loss))

        if self.min_valid_loss > mean_curr_valid_loss:
            print(f'Validation Loss Decreased {self.min_valid_loss} ---> {mean_curr_valid_loss} \t Saving The Model')
            self.min_valid_loss = mean_curr_valid_loss
            torch.save(self.model.state_dict(), '%s/%s.pth' % (self.params['logdir'] + '/', 'model'))

        self.log_model_predictions(batch_x, batch_x_ref,batch_masked_img, batch_masked_ref, batch_y, 'valid')

        return EVAL_COSTS

    def run_test_loop(self):

        TEST_COSTS = []
        self.model.eval()
        # BATCH_SIZE = self.model_params['batch_size']
        # print("\nLets test")

        for b, (batch_x, batch_x_ref, batch_masked_img, batch_masked_ref, batch_y, batch_y_ref) in enumerate(self.testloader):
            with torch.no_grad():
                pred_px = self.model(batch_x, batch_x_ref, batch_masked_img, batch_masked_ref).to(device)
                true_px = batch_y.to(device)
                cost = nn.functional.mse_loss(pred_px, true_px)

            TEST_COSTS.append(cost.item())
            # print(
            #     f'epoch: {b:2}  batch: {b:4} [{BATCH_SIZE * b:6}/{len(self.testset)}]  test loss: {cost.item():10.8f}')

        mean_curr_test_loss = np.mean(TEST_COSTS)
        print('\nTest loss : ' + str(mean_curr_test_loss))

        self.log_model_predictions(batch_x, batch_x_ref,batch_masked_img, batch_masked_ref, batch_y, 'test')
        # plt.close("all")

    def log_model_predictions(self, batch_x, batch_x_ref, batch_masked_img, batch_masked_ref,  batch_y, status):
        # model predictions

        self.model.eval()

        with torch.no_grad():

            pred = self.model(batch_x, batch_x_ref, batch_masked_img, batch_masked_ref).to(device).cpu().detach().numpy()
            true = batch_y.to(device).cpu().detach().numpy()

        if self.model_params['norm_method'] == 'meanstd':
            pred = unnormalize(pred, self.originalset.data_statistics['mean'], self.originalset.data_statistics['std'])
            true = unnormalize(true, self.originalset.data_statistics['mean'], self.originalset.data_statistics['std'])
        elif self.model_params['norm_method'] == 'maxmin':
            pred = unnormalize_max_min(pred, self.originalset.data_statistics['max'],
                                       self.originalset.data_statistics['min'])
            true = unnormalize_max_min(true, self.originalset.data_statistics['max'],
                                       self.originalset.data_statistics['min'])

        log_path = self.params['logdir'] + '/' + f'{status}_eval.txt'
        mode = 'a' if os.path.exists(log_path) else 'w'
        with open(log_path, mode) as f:
            f.write(f'rmse: {np.sqrt(np.mean((true - pred) ** 2, axis=0))}\n')

        # display visual model inputs
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        im_inv = inv_normalize(batch_x)  # back to [0,1]

        self.fig.clf()
        im = make_grid(im_inv, nrow=4)
        plt.imshow(im.permute(1, 2, 0).cpu().detach().numpy())
        self.fig.savefig(self.params['logdir'] + '/' + 'visual_input_{}.png'.format(status), bbox_inches='tight')
        self.fig.clf()

        # if self.model_params['output'] == 'pixel' or self.model_params['output'] == 'all':
        if 'pixel' in self.model_params['output']:
            # Inverse normalize the images
            IDX = 0 if self.model_params['output'] == 'pixel' else 6

            inv_normalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            im_inv = inv_normalize(batch_x)

            im_inv_resized = [torch.nn.functional.interpolate(
                img_i.unsqueeze(1),
                size=[self.originalset.h, self.originalset.w],  # 480, 640
                mode="bicubic",
                align_corners=False,
            ).squeeze() for img_i in im_inv]

            im_list = [transforms.ToPILImage()(b) for b in im_inv_resized]
            im_list_cv2 = [cv2.cvtColor(np.array(b), cv2.COLOR_RGB2BGR) for b in im_list]

            im_list_cv2_with_gt = [cv2.circle(b, (int(px[0]), int(px[1])), int(px[2]), (0, 0, 0), 2)
                                   for b, px in zip(im_list_cv2, true[:, IDX:IDX + 3])]

            im_list_cv2_with_gt_and_pres = [
                cv2.circle(b, (int(px[0]), int(px[1])), int(max(1, px[2])), (255, 255, 255), 2)
                for b, px in zip(im_list_cv2_with_gt, pred[:, IDX:IDX + 3])]

            im_list_with_gt = [cv2.cvtColor(np.array(b), cv2.COLOR_BGR2RGB) for b in im_list_cv2_with_gt_and_pres]

            im_list_with_gt = [transforms.ToTensor()(b) for b in im_list_with_gt]

            im = make_grid(im_list_with_gt, nrow=4)

            plt.imshow(im.permute(1, 2, 0).numpy())

            self.fig.savefig(self.params['logdir'] + '/' + 'pixel_output_{}.png'.format(status), bbox_inches='tight')
            self.fig.clf()

        if self.model_params['output'] == 'pose':

            from mpl_toolkits.mplot3d import Axes3D
            # self.fig = plt.figure(figsize=(20, 15))

            ax = self.fig.add_subplot(111, projection='3d')
            ax.autoscale(enable=True, axis='both', tight=True)

            # # # Setting the axes properties
            ax.set_xlim3d(self.originalset.y_min[0], self.originalset.y_max[0])
            ax.set_ylim3d(self.originalset.y_min[1], self.originalset.y_max[1])
            ax.set_zlim3d(self.originalset.y_min[2], self.originalset.y_max[2])

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 0.012, 0.016)
            ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')
            Xc, Yc, Zc = data_for_sphere_along_z(0., 0., 0.012, 0.016)
            ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

            # ax = self.fig.add_subplot(111, projection='3d')
            for i in range(len(true)):
                true_pose = true[i]
                ax.scatter(true_pose[0], true_pose[1], true_pose[2], c='black')
                pred_pose = pred[i]
                ax.scatter(pred_pose[0], pred_pose[1], pred_pose[2], c='red')

            ax.autoscale(enable=True, axis='both', tight=True)
            set_axes_equal(ax)
            self.fig.savefig(self.params['logdir'] + '/' + 'visual_output_{}.png'.format(status), bbox_inches='tight')

        if self.model_params['output'] == 'force':

            ax = self.fig.add_subplot(111, projection='3d')
            for i in range(len(true)):
                true_force = true[i]
                ax.scatter(true_force[0], true_force[1], true_force[2], c='black')
                pred_force = pred[i]
                ax.scatter(pred_force[0], pred_force[1], pred_force[2], c='red')

            ax.autoscale(enable=True, axis='both', tight=True)
            set_axes_equal(ax)
            self.fig.savefig(self.params['logdir'] + '/' + 'visual_output_{}.png'.format(status), bbox_inches='tight')

        if 'pose_force' in self.model_params['output']:

            ax = self.fig.add_subplot(111, projection='3d')
            ax.autoscale(enable=True, axis='both', tight=True)

            ax.set_xlim3d(self.originalset.y_min[0], self.originalset.y_max[0])
            ax.set_ylim3d(self.originalset.y_min[1], self.originalset.y_max[1])
            ax.set_zlim3d(self.originalset.y_min[2], self.originalset.y_max[2])

            Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 0.012, 0.016)
            ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')
            Xc, Yc, Zc = data_for_sphere_along_z(0., 0., 0.012, 0.016)
            ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

            for i in range(len(true)):
                scale = 1500
                true_pose = true[i][:3]
                true_force = true[i][3:6]
                _, ind = self.tree.query(true_pose)
                true_rot = quaternion_matrix(convert_quat_xyzw_to_wxyz(self.finger_geometry[1][ind]))
                true_force_transformed = np.dot(true_rot[:3, :3], true_force)

                ax.scatter(true_pose[0], true_pose[1], true_pose[2], c='black')
                a = Arrow3D([true_pose[0], true_pose[0] + true_force_transformed[0] / scale],
                            [true_pose[1], true_pose[1] + true_force_transformed[1] / scale],
                            [true_pose[2], true_pose[2] + true_force_transformed[2] / scale],
                            mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
                ax.add_artist(a)

                pred_pose = pred[i][:3]
                pred_force = pred[i][3:6]
                _, ind = self.tree.query(pred_pose)
                pred_rot = quaternion_matrix(convert_quat_xyzw_to_wxyz(self.finger_geometry[1][ind]))
                pred_force_transformed = np.dot(pred_rot[:3, :3], pred_force)

                ax.scatter(pred_pose[0], pred_pose[1], pred_pose[2], c='red')
                a = Arrow3D([pred_pose[0], pred_pose[0] + pred_force_transformed[0] / scale],
                            [pred_pose[1], pred_pose[1] + pred_force_transformed[1] / scale],
                            [pred_pose[2], pred_pose[2] + pred_force_transformed[2] / scale],
                            mutation_scale=20, lw=1, arrowstyle="-|>", color="red")
                ax.add_artist(a)

            ax.autoscale(enable=True, axis='both', tight=True)
            set_axes_equal(ax)
            # ax.view_init(90, 90)

            self.fig.savefig(self.params['logdir'] + '/' + 'visual_output_{}.png'.format(status), bbox_inches='tight')


def main():
    warnings.filterwarnings("ignore")

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', '-ep', type=int, default=10)

    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--portion', '-pr', type=float, default=1.0)
    parser.add_argument('--model_name', '-mn', type=str, default='resnet18')
    parser.add_argument('--input_type', '-it', type=str, default='with_ref_6c')  # 'single' , 'with_ref_6c', 'with_ref_rnn'
    parser.add_argument('--leds', '-ld', type=str, default='rrrgggbbb')  #
    parser.add_argument('--gel', '-gl', type=str, default='markers')  #
    parser.add_argument('--sensor_id', '-sid', type=int, default=0)

    parser.add_argument('--norm_method', '-im', type=str, default='meanstd')  # 'maxmin' , 'meanstd', 'none'
    parser.add_argument('--aug', '-aug', default=False)

    parser.add_argument('--output', '-op', type=str, default='pose')
    parser.add_argument('--scheduler', '-sch', type=str, default='none')

    parser.add_argument('--image_size', '-iz', type=int, default=224)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--train_size', '-ts', type=int, default=10000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--dropout', '-dp', type=float, default=0.)

    parser.add_argument('--display_step', '-d', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-gpu', default=True)
    parser.add_argument('--which_gpu', type=int, default=0)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    if torch.cuda.is_available() and params["use_gpu"]:
        which_gpu = "cuda:" + str(params["which_gpu"])
        params["device"] = torch.device(which_gpu)
        print("Pytorch is running on GPU", params["which_gpu"])
    else:
        params["device"] = torch.device("cpu")
        print("Pytorch is running on the CPU")

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    options = [
        ['rrrgggbbb', 'clear', False, 9, 'with_ref_6c'],
        # ['rrrgggbbb', 'markers', True, 2, 'with_mask_8c'],
        # ['rrrgggbbb', 'markers', False, 2, 'with_mask_7c'],
        # ['rrrgggbbb', 'markers', True, 2, 'with_mask_7c'],

        # ['rrrgggbbb', 'markers', True, 2, 'with_ref_rnn'],
        # ['rrrgggbbb', 'clear', False, 3],
        # ['white', 'markers', True, 5, 'single'],
        # ['white', 'markers', True, 5, 'with_ref_6c'],
        # ['white', 'markers', True, 5, 'with_ref_rnn'],
        # ['white', 'clear', False, 4],
        # ['rgbrgbrgb', 'markers', False, 7],
        # ['rgbrgbrgb', 'clear', False, 9]
    ]

    for conf in options:
        params['leds'] = conf[0]
        params['gel'] = conf[1]
        params['aug'] = conf[2]
        params['sensor_id'] = conf[3]
        params['input_type'] = conf[4]

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'train_history/{}/{}/'.format(params['gel'],
                                                               params['leds']))

        if not (os.path.exists(data_path)):
            os.makedirs(data_path)

        logdir = 'train' + '_' + params['output']
        logdir += '_' + params['model_name']
        logdir += '_det' if params['deterministic'] else '_prob'
        logdir += '_' + params['input_type']
        logdir += '_aug' if params['aug'] else ''
        logdir += '_' + params['norm_method']

        logdir += '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

        logdir = os.path.join(data_path, logdir)
        params['logdir'] = logdir
        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        print("\nLOGGING TO: ", logdir, "\n")

        ###################
        ### RUN TRAINING & TESTING
        ###################

        # trainer = Trainer(params)
        # trainer.run_test_loop()  # zero-shot

        for i in reversed(range(10)):
            i += 1
            params['portion'] = (11 - i) * 0.1
            trainer = Trainer(params)
            trainer.run_training_loop()
            trainer.run_test_loop()


if __name__ == "__main__":
    main()
