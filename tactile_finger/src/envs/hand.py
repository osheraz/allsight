import concurrent.futures
import time
import torch
from torchvision import transforms
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


from src.allsight.tactile_finger.src.envs.finger import Finger


class Hand():

    def __init__(self, dev_names=None):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """

        if dev_names is None:
            dev_names = [4, 6, 8]

        #     tactile = Finger(dev_name=device_id, serial='/dev/video')
        #
        #     tactile.connect()
        #
        #     tactile.show_view(ref_frame=tactile.get_frame())

        self.finger_left = Finger(dev_name=dev_names[1], serial='/dev/video', fix=(0, -18))  # (10,5)
        self.finger_right = Finger(dev_name=dev_names[0], serial='/dev/video', fix=(-7, 10))  # (-7, 0)
        self.finger_bottom = Finger(dev_name=dev_names[2], serial='/dev/video', fix=(8, 3))  # (-5,-10)

        # self.finger_bottom = Finger(dev_name=dev_names[0], serial='/dev/video')

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
            # cv2.imshow("Finger View Right", right)
            # cv2.imshow("Finger View Bottom", bottom)

            # diff_abs = raw_image_2_height_map(raw_image, ref_frame)
            # cv2.imshow('diff abs', diff_abs)
            #
            # diff = _diff(raw_image, ref_frame)
            # cv2.imshow('diff', diff)

            # cv2.imshow('red', raw_image[:, :, 2])
            # cv2.imshow('green', raw_image[:, :, 1])
            # cv2.imshow('blue', raw_image[:, :, 0])

            # cv2.imshow("2 View {}".format(self.serial), raw_image)

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()


    def hand_inference(self, model, model_params, transform, statistics):

        while True:


            left, right, bottom = self.get_frames()
            # _, ext_img = self.ext_cam.read()

            cv2.imshow("Hand View", np.concatenate((left, right, bottom), axis=1))


            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    import os

    pc_name = os.getlogin()

    device_id = 0 if pc_name == 'roblab20' else 4

    tactile = Hand()

    tactile.init_hand()

    tactile.show_fingers_view()
