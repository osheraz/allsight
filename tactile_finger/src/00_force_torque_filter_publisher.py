#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import WrenchStamped
from robotiq_force_torque_sensor.srv import sensor_accessor
import robotiq_force_torque_sensor.srv as ft_srv


class ForceTorqueFilter:

    '''
    Simple F\t publisher
    '''
    def __init__(self):
        self.e = 0.95
        self.robotiq_wrench_filtered_state = np.array([0, 0, 0, 0, 0, 0])
        self.ft_zero = rospy.ServiceProxy('/robotiq_force_torque_sensor_acc', sensor_accessor)
        self.force_pub = rospy.Publisher('/robotiq_force_torque_wrench_filtered_exp', WrenchStamped, queue_size=10)
        rospy.Subscriber('/robotiq_force_torque_wrench', WrenchStamped, self._robotiq_wrench_filtered_states_callback)

    def _robotiq_wrench_filtered_states_callback(self, data):

        e = self.e
        self.robotiq_wrench_filtered_state = e * self.robotiq_wrench_filtered_state + (1 - e) * np.array(
            [data.wrench.force.x,
             data.wrench.force.y,
             data.wrench.force.z,
             data.wrench.torque.x,
             data.wrench.torque.y,
             data.wrench.torque.z])

        ft = WrenchStamped()
        ft.header = data.header
        ft.wrench.force.x = self.robotiq_wrench_filtered_state[0]
        ft.wrench.force.y = self.robotiq_wrench_filtered_state[1]
        ft.wrench.force.z = self.robotiq_wrench_filtered_state[2]
        ft.wrench.torque.x = self.robotiq_wrench_filtered_state[3]
        ft.wrench.torque.y = self.robotiq_wrench_filtered_state[4]
        ft.wrench.torque.z = self.robotiq_wrench_filtered_state[5]

        self.force_pub.publish(ft)

    def calib_robotiq(self):

        # self.ft_zero.call(8, "")
        rospy.sleep(0.5)
        msg = ft_srv.sensor_accessorRequest()
        msg.command = "SET ZRO"
        suc_zero = True
        self.robotiq_wrench_filtered_state *= 0
        for _ in range(5):
            result = self.ft_zero(msg)
            if 'Done' not in str(result):
                suc_zero &= False
                rospy.logerr('Failed calibrating the F\T')
            else:
                suc_zero &= True
        return suc_zero

if __name__ == '__main__':

    rospy.init_node('force_torque_filter')

    ftf = ForceTorqueFilter()
    rate = rospy.Rate(100.0)

    while not rospy.is_shutdown():
        rate.sleep()
