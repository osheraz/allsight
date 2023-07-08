#!/usr/bin/env python

import numpy as np
import rospy
import cv2
import os
import sys
from envs.finger_ros import TactileSubscriberFinger

# TODO: fix timm python 2.7 issues , move to ros to python3 / ROS2

# sys.path.insert(0, '/home/osher/catkin_ws/src/allsight/experiments/utils/')
# from surface import create_finger_geometry
# from geometry import convert_quat_xyzw_to_wxyz
# from transforms import get_transforms
# from models import get_model
from transformations import quaternion_matrix
from scipy import spatial
import torch
import json
from std_msgs.msg import Float32MultiArray


np.set_printoptions(precision=3)  # to widen the printed array
pc_name = os.getlogin()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu


class ContactPublisher(TactileSubscriberFinger):

    def __init__(self, model_params, classifier_params, statistics):
        """
        ContactPublisher Device class for a single Finger
        """
        TactileSubscriberFinger.__init__(self)

        self.init_success &= self.load_model(model_params, classifier_params)
        _, _, self.transform = get_transforms(int(model_params['image_size']))

        self.finger_geometry = create_finger_geometry()
        self.tree = spatial.KDTree(self.finger_geometry[0])
        self.statistics = statistics
        self.output_type = model_params['output']
        self.norm_method = model_params['norm_method']
    #         frame = self.finger.last_frame


        self._topic_name = rospy.get_param('~topic_name', 'allsight/contact/')
        rospy.loginfo("[%s] (topic_name) Publishing Contact to topic  %s", self._topic_name)

        self._contact_publisher = rospy.Publisher(self._topic_name, Float32MultiArray, queue_size=1)

        self.fps = 60
        self._rate = rospy.get_param('~publish_rate', self.fps)
        rospy.loginfo("[%s] (publish_rate) Publish rate set to %s hz", self.name, self._rate)

        self._frame_id = rospy.get_param('~frame_id', 'fix')
        rospy.loginfo("[%s] (frame_id) Frame ID set to  %s", self.name, self._frame_id)


    def load_model(self, model_params, classifier_params):

        self.model = get_model(model_params)

        print('loaded {} with output: {}'.format(model_params['input_type'], model_params['output']))
        self.model.load_state_dict(torch.load(model_params['logdir'] + '/model.pth'))
        self.model.eval()

        if classifier_params is not None:
            self.classifier = get_model(classifier_params)
        else:
            self.classifier = None

    def process_image(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.unsqueeze(self.transform(frame).to(device), 0)

        return frame

    def run(self):

        msg = Float32MultiArray()
        ros_rate = rospy.Rate(self._rate)

        ref_frame = self.last_frame
        ref_frame = self.process_image(ref_frame)
        while not rospy.is_shutdown():

            try:
                frame = self.process_image(self.last_frame)
                pred = self.model(frame, ref_frame).cpu().detach().numpy()

                msg.data = pred.reshape(-1, 1)

                # if frame is not None:
            #         ros_msg = self._cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
            #         ros_msg.header.frame_id = self._frame_id
            #         ros_msg.header.stamp = rospy.Time.now()
            #         self._image_publisher.publish(ros_msg)
            #     else:
            #         rospy.loginfo("[%s] Invalid image file", self.name)
            #     ros_rate.sleep()
            #
            except Exception as e:
                rospy.logerr(e)


            self._contact_publisher.publish(msg)

        self._rate.sleep()


if __name__ == "__main__":

    with_classifier = False

    leds = 'rrrgggbbb'
    gel = 'markers'

    path_to_dir = '/home/{}/catkin_ws/src/allsight/experiments/train_history/{}/{}/'.format(pc_name, gel, leds)

    model_name = 'train_pose_force_pixel_torque_depth_resnet18_det_single_meanstd_13-05-2023_00-08-26'
    path_to_model = path_to_dir + model_name + '/'

    # Load data statistics and model params
    with open(path_to_dir + "data_statistic.json", 'rb') as handle:
        statistics = json.load(handle)

    with open(path_to_dir + 'model_params.json', 'rb') as json_file:
        model_params = json.load(json_file)

    if with_classifier:

        model_name_classify = 'train_touch_detect_det_13-02-2023_16-39-47'
        path_to_dir_classify = path_to_dir + model_name_classify + '/'

        with open(path_to_dir_classify + 'model_params.json', 'rb') as json_file:
            classifier_params = json.load(json_file)
    else:
        classifier_params = None

    cp = ContactPublisher(model_params=model_params, classifier_params=classifier_params, statistics=statistics)
