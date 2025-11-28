import cv2
import numpy as np

from allsight.finger import Finger
from allsight.utils.img_utils import ContactArea, circle_mask, align_center


class Hand:

    def __init__(
        self,
    ):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """
        self.init_success = False
        self.mask_resized = None
        self.finger_left = Finger(dev_name=0, serial="/dev/video", fix=(0, 0))
        self.finger_right = Finger(dev_name=1, serial="/dev/video", fix=(0, 0))

        self.init_hand()
        self.left_bg, self.right_bg = self.get_frames(diff=False)

    def init_hand(self):
        """
        Sets stream resolution based on supported streams in Finger.STREAMS
        :param resolution: QVGA or VGA from Finger.STREAMS
        :return: None
        """
        self.finger_right.connect()
        self.finger_left.connect()
        self.init_success = True

    def get_frames(self, diff=True):
        """
        Returns a single image frame for the device
        :param transpose: Show direct output from the image sensor, WxH instead of HxW
        :return: Image frame array
        """

        left = self.finger_left.get_frame()
        right = self.finger_right.get_frame()

        min_width = min(left.shape[1], right.shape[1])
        min_height = min(left.shape[0], right.shape[0])

        if self.mask_resized is None:
            self.mask_resized = circle_mask((min_width, min_height))

        left = cv2.resize(left, (min_width, min_height))
        right = cv2.resize(right, (min_width, min_height))

        if diff:
            left = self._subtract_bg(left, self.left_bg) * self.mask_resized
            right = self._subtract_bg(right, self.right_bg) * self.mask_resized

        return left, right

    def show_fingers_view(self, display_diff=True):
        """
        Creates OpenCV named window with live view of Finger device, ESC to close window
        :param ref_frame: Specify reference frame to show image difference
        :return: None
        """
        left_bg, right_bg = self.get_frames(diff=False)

        from time import time

        while True:

            start_time = time()

            left, right = self.get_frames(diff=False)

            if display_diff:

                diff_left = self._subtract_bg(left, left_bg) * self.mask_resized
                diff_right = self._subtract_bg(right, right_bg) * self.mask_resized

                cv2.imshow(
                    "Hand View\tLeft\tRight",
                    np.concatenate((diff_left, diff_right), axis=1),
                )
            else:
                cv2.imshow("Hand View", np.concatenate((left, right), axis=1))

            if cv2.waitKey(1) == 27:
                break

            print(
                "FPS: ", 1.0 / (time() - start_time)
            )  # FPS = 1 / time to process loop

        cv2.destroyAllWindows()

    def _subtract_bg(self, img1, img2, offset=0.5):
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff


if __name__ == "__main__":
    import os

    pc_name = os.getlogin()

    tactile = Hand()

    tactile.show_fingers_view()
