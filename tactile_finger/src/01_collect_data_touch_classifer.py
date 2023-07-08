#!/usr/bin/env python

import rospy
import numpy as np
from envs.finger_ros import TactileSubscriberFinger
import cv2
import os
from datetime import datetime
import json


def main():

    save = True

    rospy.init_node('touch_exp')
    ros_rate = rospy.Rate(100)
    indenter = '30'
    leds = 'rrrgggbbb'  # 'rgbrgbrgb

    finger = TactileSubscriberFinger()

    success = finger.init_success
    # Init of the dataset dir paths with the current day and time
    data_dict = {}
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S")
    dir_path = os.path.dirname(os.path.realpath(__file__))[:-19]

    dataset_path_images = dir_path + "/dataset/{}/images_classify/no_touch/{}/img_{}".format(leds, indenter, date)
    dataset_path_data = dir_path + "/dataset/{}/data_classify/no_touch/{}/data_{}".format(leds, indenter, date)

    if save:
        if not os.path.exists(dataset_path_images): os.mkdir(dataset_path_images)
        if not os.path.exists(dataset_path_data): os.mkdir(dataset_path_data)

    max_time = 15.0  # record data for 'max_time' seconds
    save_every_sec = 0.025  # capture image every 'save_every_sec' seconds

    if success:

        rospy.sleep(2)

        ref_frame = finger.last_frame
        ref_img_color_path = os.path.join(dataset_path_images, 'ref_frame.jpg')

        if save:
            if not cv2.imwrite(ref_img_color_path, ref_frame):
                raise Exception("Could not write image")

        try:

            start_time = rospy.get_time()
            cur_time = rospy.get_time() - start_time
            save_time = cur_time

            while cur_time < max_time:

                cur_time = rospy.get_time() - start_time

                print(cur_time)

                if cur_time - save_time > save_every_sec:

                    save_time = cur_time

                    frame = finger.last_frame

                    img_id = 'image{:.2f}.jpg'.format(save_time)
                    img_path = os.path.join(dataset_path_images, img_id)

                    if save:
                        if not cv2.imwrite(img_path, frame):
                            raise Exception("Could not write image")

                    data_dict[img_id] = {'frame': img_path}

                ros_rate.sleep()

        except:
            rospy.logwarn('Failed saving data , verify.')
            success = False

        if save:
            # save the data dictionary to the relevant dir as a json file
            path = os.path.join(dataset_path_data, 'data_{}.json'.format(date))
            if save:
                with open(path, 'w') as json_file:
                    json.dump(data_dict, json_file, indent=3)

            rospy.loginfo('finished experiment, everything was OK, lets submit'.format(success))


if __name__ == '__main__':
    main()
