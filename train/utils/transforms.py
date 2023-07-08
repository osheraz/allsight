from torchvision import transforms
import torch
from PIL import ImageFilter
import random

# display visual model inputs
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def get_transforms(im_size):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.RandomChoice([
            # transforms.RandomRotation(3),  # rotate +/- 10 degrees
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            # transforms.ColorJitter(brightness=1.0, contrast=0.1, saturation=0.5, hue=0.1),
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    aug_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        # transforms.RandomChoice([
        #     transforms.RandomRotation(3),  # rotate +/- 10 degrees
        #     transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))
        # ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, aug_transform, test_transform


import torchvision.transforms.functional as TF

class ToGrayscale(object):
    """Convert image to grayscale version of image."""

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        return TF.to_grayscale(img, self.num_output_channels)


class AdjustGamma(object):
    """Perform gamma correction on an image."""

    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img):
        return TF.adjust_gamma(img, self.gamma, self.gain)



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return torch.cat([q, k], dim=0)