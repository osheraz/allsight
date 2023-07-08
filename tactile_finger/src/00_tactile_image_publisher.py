#!/usr/bin/env python

import rospy
from envs.finger_ros import TactileFingerROSPublisher

if __name__ == "__main__":
    rospy.init_node('tactile_finger_publisher')

    tactile = TactileFingerROSPublisher(dev_name=4, serial='/dev/video')

    tactile.connect()

    tactile.run()
