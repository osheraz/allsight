import numpy as np
import cv2
from env_utils.img_utils import ContactArea, circle_mask, align_center
from env_utils.img_utils import center_mask, _diff, ring_mask
from env_utils.img_utils import _mask, square_cut

# src.allsight.tactile_finger.src.envs.
# src.allsight.tactile_finger.src.envs.
# src.allsight.tactile_finger.src.envs.

class Finger:

    def __init__(self, serial=None, dev_name=None, fix=(0, 0)): # 10,6

        self.serial = serial
        self.name = 'AllSight'

        self.__dev = None
        self.contact = ContactArea()
        self.dev_name = dev_name

        self.resolution = {"width": 640, "height": 480}
        self.fps = 30

        self.mask = circle_mask(fix=fix)

        if self.serial is not None:
            print("Finger object constructed with serial {}".format(self.serial))

    def connect(self):
        print("{}:Connecting to Finger".format(self.serial))
        self.__dev = cv2.VideoCapture(self.dev_name)
        if not self.__dev.isOpened():
            print("Cannot open video capture device {} - {}".format(self.serial, self.dev_name))
            raise Exception("Error opening video stream: {}".format(self.dev_name))
        self.init_sensor()

    def init_sensor(self):
        """
        Sets stream resolution based on supported streams in Finger.STREAMS
        :param resolution: QVGA or VGA from Finger.STREAMS
        :return: None
        """
        width = self.resolution["width"]
        height = self.resolution["height"]
        print("{}:Stream resolution set to {}w x {}h".format(self.serial, height, width))
        print("{}:Stream FPS set to {}".format(self.serial, self.fps))
        self.__dev.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__dev.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.__dev.set(cv2.CAP_PROP_FPS, self.fps)

    def get_frame(self, transpose=True):
        """
        Returns a single image frame for the device
        :param transpose: Show direct output from the image sensor, WxH instead of HxW
        :return: Image frame array
        """
        ret, frame = self.__dev.read()
        if not ret:
            print(
                "Cannot retrieve frame data from {}, is Finger device open?".format(self.serial)
            )
            raise Exception(
                "Unable to grab frame from {} - {}!".format(self.serial, self.dev_name)
            )
        if not transpose:
            frame = cv2.transpose(frame, frame)
            frame = cv2.flip(frame, 0)

        if self.mask is None:
            self.find_center(frame)

        frame = (frame * self.mask).astype(np.uint8)
        # img_all_blurred = cv2.medianBlur(frame, 9)
        # frame = np.where(ring_mask() > 0, img_all_blurred, frame)
        frame = align_center(frame, self.mask)
        frame = square_cut(frame)

        return frame

    def find_center(self, clear_image):

        depth_image = clear_image.copy()
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)

        # Apply the Hough Circle Transform
        circles = cv2.HoughCircles(depth_image, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=10, minRadius=3,
                                   maxRadius=80)

        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :][0]).astype("int")
            fix = (int(clear_image.shape[0] // 2 - circles[1]), int(clear_image.shape[1] // 2 - circles[0]))
            print('Fix Values: {}'.format(fix))
            self.mask = circle_mask(fix=fix)


    def find_contact(self, raw_image, ref_frame):

        C = self.contact(raw_image, ref_frame)

        if C is not None:
            poly, major_axis, major_axis_end, minor_axis, minor_axis_end = C

            cv2.polylines(raw_image, [poly], True, (255, 255, 255), 2)
            cv2.line(
                raw_image,
                (int(major_axis_end[0]), int(major_axis_end[1])),
                (int(major_axis[0]), int(major_axis[1])),
                (0, 0, 255),
                2,
            )
            cv2.line(
                raw_image,
                (int(minor_axis_end[0]), int(minor_axis_end[1])),
                (int(minor_axis[0]), int(minor_axis[1])),
                (0, 255, 0),
                2,
            )

        return raw_image

    def show_view(self, ref_frame=None):
        """
        Creates OpenCV named window with live view of Finger device, ESC to close window
        :param ref_frame: Specify reference frame to show image difference
        :return: None
        """

        while True:

            raw_image = self.get_frame()

            cv2.imshow("Finger View {}".format(self.serial), raw_image)

            # # Mask
            # marker_img = _mask(raw_image)
            # cv2.imshow('markers', marker_img)
            #
            # # diff
            # diff = _diff(raw_image, ref_frame)
            # cv2.imshow('diff', diff)

            # By channel
            # cv2.imshow('red', raw_image[:, :, 2])
            # cv2.imshow('green', raw_image[:, :, 1])
            # cv2.imshow('blue', raw_image[:, :, 0])

            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def to_polar(self, image):

        margin = 1.0  # Cut off the outer 10% of the image
        # Do the polar rotation along 1024 angular steps with a radius of 256 pixels.
        polar_img = cv2.warpPolar(image, (256, 1024), (image.shape[0] / 2, image.shape[1] / 2),
                                  image.shape[1] * margin * 0.5, cv2.WARP_POLAR_LINEAR)
        # Rotate it sideways to be more visually pleasing
        polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return polar_img

    def disconnect(self):
        print("{}:Closing Finger device".format(self.serial))
        self.__dev.release()

    def __repr__(self):
        return "Finger(serial={}, name={})".format(self.serial, self.name)


if __name__ == "__main__":
    import os

    pc_name = os.getlogin()

    device_id = 0 if pc_name == 'roblab20' else 4

    tactile = Finger(dev_name=device_id, serial='/dev/video')

    tactile.connect()

    tactile.show_view(ref_frame=tactile.get_frame())
