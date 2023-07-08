import time
import os
import sys
import json
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau  # Learning rate schedulers
from utils.models import PreTrainedModel, PreTrainedModelWithRef
from utils.datasets import TactileTouchDataset, output_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
np.set_printoptions(formatter=dict(int=lambda x: f'{x:3}'))  # to widen the printed array

pc_name = os.getlogin()



class Trainer(object):

    def __init__(self, params):

        # Get params, create logger
        self.params = params

        # Set random seeds
        seed = self.params['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)

        #####################
        ## SET AGENT PARAMS
        #####################

        buffer_names = ['data_2023_02_02-05:21:12',
                        'data_2023_02_02-10:34:06',
                        'data_2023_02_02-11:46:32',
                        'data_2023_02_02-12:36:56']

        buffer_paths = [f"/home/{pc_name}/catkin_ws/src/allsight/dataset/data/" \
                        f"{bf}/{bf}_transformed_annotated.json" for bf in buffer_names]

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
            'model_name': params['model_name'],
            'buffer_name': buffer_names,
            'output': params['output']
        }

        self.prepare_data(buffer_paths, params['output'], finetune=True)

        self.model = PreTrainedModel(params['model_name'], output_map[params['output']], classifer=True)
        self.model.to(device)

        with open(self.params['logdir'] + '/model_params.json', 'w') as fp:
            dic_items = self.model_params.items()
            new_dict = {key: value for key, value in dic_items}
            json.dump(new_dict, fp, indent=3)

        self.optimizer = getattr(torch.optim, self.model_params['optimizer'])(list(self.model.parameters()),  # +
                                                                              # list(self.touch_detect.parameters())),
                                                                              params['learning_rate'])
        # weight_decay=0.0 )


        plateau_factor = 0.5
        plateau_patience = 3
        cosine_T_max = 4
        cosine_eta_min = 1e-8
        self.max_val = 0

        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=plateau_factor,
        #                                    patience=plateau_patience, verbose=True)

        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_T_max, eta_min=cosine_eta_min)
        self.scheduler = None
        # self.fig = plt.figure(figsize=(15, 10))

    # def accuracy(self, outputs, labels):
    #     preds = torch.round(outputs.data)
    #     return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def prepare_data(self, paths, output_type, finetune=False):

        for idx, p in enumerate(paths):
            if idx == 0:
                df_data = pd.read_json(p).transpose()
            else:
                df_data = pd.concat([df_data, pd.read_json(p).transpose()], axis=0)

        data_no_touch = df_data[df_data.ref_frame == df_data.frame]
        data_no_touch['touch'] = 0

        if finetune:
            path_to_finetuned = f'/home/{pc_name}/catkin_ws/src/allsight/dataset/data_classify/no_touch/' \
                                'data_2023_02_07-12:12:16/data_2023_02_07-12:12:16.json'
            finetuned_no_touch = pd.read_json(path_to_finetuned).transpose()
            finetuned_no_touch['touch'] = 0
            data_no_touch = pd.concat([data_no_touch, finetuned_no_touch])

        data_touch = df_data[df_data.ref_frame != df_data.frame].sample(data_no_touch.shape[0])
        data_touch['touch'] = 1

        df_data = pd.concat([data_no_touch, data_touch])

        train_df, valid_df = train_test_split(df_data, test_size=0.2, shuffle=True)

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.params['image_size'], self.params['image_size'])),
            transforms.RandomRotation(3),  # rotate +/- 10 degrees
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # transforms.ColorJitter(brightness=0, contrast=0, saturation=0.05, hue=0.05),
            # transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            # transforms.RandomHorizontalFlip(p=0.5),  # reverse 50% of images
            # transforms.CenterCrop(IMG_SIZE),  # crop the longest side to IMG_SIZE pixels at center
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

        self.trainset = TactileTouchDataset(self.model_params, train_df, output_type, self.train_transform, apply_mask=True)
        self.validset = TactileTouchDataset(self.model_params, valid_df, output_type, self.test_transform, apply_mask=True)

        self.trainloader = DataLoader(self.trainset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        self.validloader = DataLoader(self.validset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)

        print(f'Train set length is {len(self.trainset)}')

    def accuracy(self, outputs, labels):
        preds = torch.round(outputs.data) # torch.round(outputs) #
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

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

                output = self.model(batch_x).to(device)
                targets = batch_y.to(device)
                loss = nn.BCELoss()(output, targets)
                acc = self.accuracy(output, targets)

                # acc = (y_pred.round() == y_batch).float().mean()

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
                        f'epoch: {epoch:2}  batch: {b:4} [{BATCH_SIZE * b:6}/{len(self.trainset)}]  '
                        f'train loss: {cost:5.5f}, train acc: {acc:5.5f}')


            mean_train_loss = np.mean(COSTS[-len(self.trainloader):])
            mean_train_acc = np.mean(ACCS[-len(self.trainloader):])

            print('Epoch train loss : ' + str(mean_train_loss))
            print('Epoch train acc : ' + str(mean_train_acc))

            self.log_model_predictions(batch_x, batch_y, 'train')

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

        print("Optimization Finished!")


    def run_validation_loop(self, EVAL_COSTS, EVAL_ACC):

        self.model.eval()
        BATCH_SIZE = self.model_params['batch_size']
        print("\nLets validate")

        for b, (batch_x, batch_y) in enumerate(self.validloader):
            with torch.no_grad():
                output = self.model(batch_x).to(device)
                targets = batch_y.to(device)
                cost = nn.BCELoss()(output, targets)
                acc = self.accuracy(output, targets)

            EVAL_COSTS.append(cost.item())
            EVAL_ACC.append(acc.item())

            print(
                f'epoch: {b:2}  batch: {b:4} [{BATCH_SIZE * b:6}/{len(self.validset)}]'
                f'  test loss: {cost.item():10.8f}, test acc: {acc.item():10.8f}')

        mean_curr_valid_loss = np.mean(EVAL_COSTS[-len(self.validset):])
        mean_curr_valid_acc = np.mean(EVAL_ACC[-len(self.validset):])

        print('\nValidation loss : ' + str(mean_curr_valid_loss))
        print('Validation Acc : ' + str(mean_curr_valid_acc))

        if mean_curr_valid_acc > self.max_val:
            print(f'Accuracy Increased {self.max_val} ---> {mean_curr_valid_acc} \t Saving The Model')
            self.max_val = mean_curr_valid_acc
            torch.save(self.model.state_dict(), '%s/%s.pth' % (self.params['logdir'] + '/', 'model'))

        self.log_model_predictions(batch_x, batch_y, 'valid')

        return EVAL_COSTS, EVAL_ACC

    def log_model_predictions(self, batch_x, batch_y, status='train'):
        # model predictions
        self.fig = plt.figure(figsize=(15, 10))

        self.model.eval()

        with torch.no_grad():

            pred = self.model(batch_x).to(device).cpu().detach().numpy()
            pred = pred.round()
            true = batch_y.to(device).cpu().detach().numpy()


        # Inverse normalize the images
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        im_inv = inv_normalize(batch_x)

        im_inv_resized = [torch.nn.functional.interpolate(
            img_i.unsqueeze(1),
            size=[self.trainset.h, self.trainset.w],
            mode="bicubic",
            align_corners=False,
        ).squeeze() for img_i in im_inv]

        im_list = [transforms.ToPILImage()(b) for b in im_inv_resized]
        im_list_cv2 = [cv2.cvtColor(np.array(b), cv2.COLOR_RGB2BGR) for b in im_list]

        im_list_cv2_with_gt = [cv2.putText(b, 'T' + str(px), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
                               for b, px in zip(im_list_cv2, true)]

        im_list_cv2_with_gt_and_pres = [cv2.putText(b, 'P' + str(px), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)
                                        for b, px in zip(im_list_cv2_with_gt, pred)]

        im_list_with_gt = [cv2.cvtColor(np.array(b), cv2.COLOR_BGR2RGB) for b in im_list_cv2_with_gt_and_pres]

        im_list_with_gt = [transforms.ToTensor()(b) for b in im_list_with_gt]

        im = make_grid(im_list_with_gt, nrow=4)

        # Print the images
        # # plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
        # im_inv = transforms.ToPILImage()(im_inv)
        plt.imshow(im.permute(1, 2, 0).numpy())

        # plot the predictions
        self.fig.savefig(self.params['logdir'] + '/' + 'visual_output_{}.png'.format(status), bbox_inches='tight')


def main():
    warnings.filterwarnings("ignore")

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', '-ep', type=int, default=40)

    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--portion', '-pr', type=float, default=1.0)
    parser.add_argument('--model_name', '-mn', type=str, default='resnet18')

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

    logdir = 'train' + '_' + params['output']
    logdir += '_det' if params['deterministic'] else '_prob'
    logdir += '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

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
