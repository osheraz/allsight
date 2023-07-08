import time
from collections import OrderedDict
import pickle
import os
import sys
import json
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import timm
from utils.misc import normalize, unnormalize
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau  # Learning rate schedulers
from utils.models import PreTrainedModel, output_map
from utils.img_utils import circle_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
np.set_printoptions(formatter=dict(int=lambda x: f'{x:3}'))  # to widen the printed array

pc_name = 'osher'

class TouchDataset(torch.utils.data.Dataset):

    def __init__(self, params, df, transform=None, apply_mask=True, remove_ref=False):
        self.df = df
        self.transform = transform
        self.apply_mask = apply_mask
        self.remove_ref = remove_ref

        if self.apply_mask:
            self.mask = circle_mask((640, 480))

        self.X = np.array([cv2.imread(df.iloc[idx].frame) for idx in range(self.df.shape[0])])

        self.Y = np.array([df.iloc[idx].touch for idx in range(self.df.shape[0])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img = self.X[idx]

        if self.remove_ref:
            img = img - self.ref_frame

        if self.apply_mask:
            img = (img * self.mask).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(np.array(img))

        y = torch.Tensor([self.Y[idx]])

        return img, y


class Trainer(object):

    def __init__(self, params):

        # Get params, create logger
        self.params = params

        # Set random seeds
        seed = self.params['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model_path = params['model_path']
        #####################
        ## SET AGENT PARAMS
        #####################

        with open(self.model_path + '/model_params.json', 'rb') as handle:
            self.model_params = json.load(handle)

        self.model = PreTrainedModel(self.model_params['model_name'],
                                     output_map[self.model_params['output']],
                                     classifer=True).to(device)

        self.model.load_state_dict(torch.load('%s.pth' % (self.model_path + '/model'), map_location=device))

        path_to_name_wo_touch = f"/home/{pc_name}/catkin_ws/src/tactile_finger/allsight/data_classify/" \
                                f"{params['buffer_name_wo_touch']}/{params['buffer_name_wo_touch']}.json"

        path_to_name_w_touch = f"/home/{pc_name}/catkin_ws/src/tactile_finger/allsight/data/" \
                               f"{params['buffer_name_w_touch']}/{params['buffer_name_w_touch']}.json"

        self.prepare_data(path_to_name_wo_touch, path_to_name_w_touch)

        with open(self.params['logdir'] + '/model_params.json', 'w') as fp:
            dic_items = self.model_params.items()
            new_dict = {key: value for key, value in dic_items}
            json.dump(new_dict, fp)

        self.optimizer = getattr(torch.optim, self.model_params['optimizer'])(self.model.parameters(),
                                                                              params['learning_rate'],
                                                                              weight_decay=0.0)

        plateau_factor = 0.5
        plateau_patience = 3
        cosine_T_max = 4
        cosine_eta_min = 1e-8
        self.max_val = 0

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=plateau_factor,
                                      patience=plateau_patience, verbose=True)

        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_T_max, eta_min=cosine_eta_min)
        # self.scheduler = None
        self.fig = plt.figure(figsize=(15, 10))

    def accuracy(self, outputs, labels):
        preds = torch.round(outputs.data)  # torch.round(outputs)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def prepare_data(self, path_wo, path_w):

        df_data_wo_touch = pd.read_json(path_wo).transpose()
        df_data_wo_touch['touch'] = 0
        df_data_w_touch = pd.read_json(path_w).transpose()
        df_data_w_touch['touch'] = 1
        df_data = pd.concat([df_data_wo_touch, df_data_w_touch.sample(df_data_wo_touch.shape[0])])

        train_df, valid_df = train_test_split(df_data, test_size=0.1, shuffle=True)

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.params['image_size'], self.params['image_size'])),
            # transforms.RandomRotation(10),  # rotate +/- 10 degrees
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            # transforms.RandomHorizontalFlip(p=0.5),  # reverse 50% of images
            # transforms.CenterCrop(IMG_SIZE),  # crop longest side to IMG_SIZE pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.params['image_size'], self.params['image_size'])),
            # resize shortest side to IMG_SIZE pixels
            # transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


        self.trainset = TouchDataset(self.params, train_df, self.train_transform, apply_mask=True)
        self.validset = TouchDataset(self.params, valid_df, self.test_transform, apply_mask=True)

        self.trainloader = DataLoader(self.trainset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        self.validloader = DataLoader(self.validset, batch_size=self.params['batch_size'], shuffle=False, drop_last=True)


        print(f'Train set length is {len(self.trainset)}')

    def run_training_loop(self):

        epochs = self.model_params['epoch']
        # init vars at beginning of training
        self.start_time = time.time()
        self.min_valid_loss = np.inf

        COSTS, EVAL_COSTS, EVAL_ACC, epoch_cost, eval_cost, eval_acc, ACCS, epoch_acc = [], [], [], [], [], [], [], []
        BATCH_SIZE = self.model_params['batch_size']

        count = 0

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            self.model.train()

            for b, (batch_x, batch_y) in enumerate(self.trainloader):

                count += 1

                loss = nn.BCELoss()(self.model(batch_x).to(self.model_params['device']),
                                              batch_y.to(self.model_params['device']))
                acc = self.accuracy(self.model(batch_x).to(self.model_params['device']),
                                                         batch_y.to(self.model_params['device']))

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(list(self.model.parameters()), 0.5)
                self.optimizer.step()
                cost = loss.item()
                COSTS.append(cost)
                ACCS.append(acc.item())

                # log/save
                if b % 10 == 0:
                    print(
                        f'epoch: {epoch:2}  batch: {b:4} [{BATCH_SIZE * b:6}/{len(self.trainset)}]  train loss: {cost:5.5f}, train acc: {acc:5.5f}')

            mean_train_loss = np.mean(COSTS[-len(self.trainloader):])
            mean_train_acc = np.mean(ACCS[-len(self.trainloader):])

            print('Epoch train loss : ' + str(mean_train_loss))

            EVAL_COSTS, EVAL_ACC = self.run_validation_loop(EVAL_COSTS, EVAL_ACC)
            mean_val_loss = np.mean(EVAL_COSTS[-len(self.validloader):])
            mean_val_acc = np.mean(EVAL_ACC[-len(self.validloader):])

            epoch_acc.append(mean_train_acc)
            epoch_cost.append(mean_train_loss)
            eval_cost.append(mean_val_loss)
            eval_acc.append(mean_val_acc)

            # apply LR scheduler after each epoch
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(mean_val_loss)

            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()

        print("Optimization Finished!")

        np.save(self.params['logdir'] + '/train_val_comp.npy', [epoch_cost, eval_cost])
        self.fig.clf()
        plt.plot(epoch_cost, '-ro', linewidth=3, label='train loss')
        plt.plot(eval_cost, '-ko', linewidth=3, label='val loss')
        plt.legend()
        self.fig.savefig(self.params['logdir'] + '/train_val_comp.png', dpi=200, bbox_inches='tight')
        # plt.close("all")

        np.save(self.params['logdir'] + '/train_val_acc.npy', [epoch_acc, eval_acc])
        self.fig.clf()
        plt.plot(epoch_acc, '-ro', linewidth=3, label='train acc')
        plt.plot(eval_acc, '-ko', linewidth=3, label='val acc')
        plt.legend()
        self.fig.savefig(self.params['logdir'] + '/train_val_acc.png', dpi=200, bbox_inches='tight')
        plt.close("all")

    def run_validation_loop(self, EVAL_COSTS, EVAL_ACC):

        self.model.eval()
        BATCH_SIZE = self.model_params['batch_size']
        print("\nLets validate")

        for b, (batch_x, batch_y) in enumerate(self.validloader):
            with torch.no_grad():
                pred_px = self.model(batch_x).to(self.model_params['device'])
                true_px = batch_y.to(self.model_params['device'])
                cost = nn.BCELoss()(pred_px, true_px)
                acc = self.accuracy(pred_px, true_px)

            EVAL_COSTS.append(cost.item())
            EVAL_ACC.append(acc.item())

            print(
                f'epoch: {b:2}  batch: {b:4} [{BATCH_SIZE * b:6}/{len(self.validset)}]  test loss: {cost.item():10.8f}, test acc: {acc.item():10.8f}')

        mean_curr_valid_loss = np.mean(EVAL_COSTS[-len(self.validset):])
        mean_curr_valid_acc = np.mean(EVAL_ACC[-len(self.validset):])

        print('\nValidation loss : ' + str(mean_curr_valid_loss))
        self.max_val = self.max_val if self.max_val > mean_curr_valid_acc else mean_curr_valid_acc
        print('Max acc : ' + str(self.max_val))

        if self.min_valid_loss > mean_curr_valid_loss:
            print(f'Validation Loss Decreased {self.min_valid_loss} ---> {mean_curr_valid_loss} \t Saving The Model')
            self.min_valid_loss = mean_curr_valid_loss
            torch.save(self.model.state_dict(), '%s/%s.pth' % (self.params['logdir'] + '/', 'model'))

        self.log_model_predictions(batch_x, batch_y)

        # if len(EVAL_COSTS) > 5:
        #     self.fig.clf()
        #     plt.plot(EVAL_COSTS, 'ko', markersize=2, alpha=0.3)
        #     self.fig.savefig(self.params['logdir'] + '/losses.png', dpi=200, bbox_inches='tight')

        return EVAL_COSTS, EVAL_ACC

    def log_model_predictions(self, batch_x, batch_y):
        # model predictions
        self.fig = plt.figure(figsize=(15, 10))

        self.model.eval()

        with torch.no_grad():

            pred = self.model(batch_x).to(self.model_params['device']).cpu().detach().numpy()
            true = batch_y.to(self.model_params['device']).cpu().detach().numpy()


        # Inverse normalize the images
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        im_inv = inv_normalize(batch_x)

        im_inv_resized = [torch.nn.functional.interpolate(
            img_i.unsqueeze(1),
            size=[480, 640],
            mode="bicubic",
            align_corners=False,
        ).squeeze() for img_i in im_inv]

        im_list = [transforms.ToPILImage()(b) for b in im_inv_resized]
        im_list_cv2 = [cv2.cvtColor(np.array(b), cv2.COLOR_RGB2BGR) for b in im_list]

        im_list_cv2_with_gt = [cv2.putText(b, str(px), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                               for b, px in zip(im_list_cv2, true)]

        im_list_cv2_with_gt_and_pres = [cv2.putText(b, str(px), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                                        for b, px in zip(im_list_cv2_with_gt, pred)]

        im_list_with_gt = [cv2.cvtColor(np.array(b), cv2.COLOR_BGR2RGB) for b in im_list_cv2_with_gt_and_pres]

        im_list_with_gt = [transforms.ToTensor()(b) for b in im_list_with_gt]

        im = make_grid(im_list_with_gt, nrow=4)

        # Print the images
        # # plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
        # im_inv = transforms.ToPILImage()(im_inv)
        plt.imshow(im.permute(1, 2, 0).numpy())


        # plot the predictions
        self.fig.savefig(self.params['logdir'] + '/' + 'visual_output.png', bbox_inches='tight')


def main():
    warnings.filterwarnings("ignore")

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', '-ep', type=int, default=5)

    parser.add_argument('--buffer_name_wo_touch', '-wobuf', type=str, default='data_2023_01_11-09:45:34')
    parser.add_argument('--buffer_name_w_touch', '-wbuf', type=str, default='data_2023_01_01-06:36:43')
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--portion', '-pr', type=float, default=1.0)
    parser.add_argument('--model_name', '-mn', type=str, default='resnet18')  # efficientnet_b0

    parser.add_argument('--model_path', '-mp', type=str, default='train_touch_detect_det_07-02-2023_11-02-32')

    parser.add_argument('--output', '-op', type=str, default='touch_detect')
    parser.add_argument('--image_size', '-iz', type=int, default=224)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--dropout', '-dp', type=float, default=0.001)

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

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_history/')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = 'finetuned_train' + '_' + params['model_path']
    logdir += '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    params['model_path'] = os.path.join(data_path, params['model_path'])

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING & TESTING
    ###################

    for i in range(1):
        i += 1
        params['portion'] = (11 - i) * 0.1
        trainer = Trainer(params)
        trainer.run_training_loop()


if __name__ == "__main__":
    main()
