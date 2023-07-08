#!/usr/bin/env python

import rospy
import tf
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf.transformations import translation_matrix, rotation_matrix, translation_from_matrix, rotation_from_matrix, \
    concatenate_matrices

origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)


def radius(z, height, rad):
    if z < height:
        return rad
    else:
        return np.sqrt(rad ** 2 - (z - height) ** 2)


if __name__ == '__main__':

    '''
    Script for publishing the surface of the finger as TF,
    Due to limitation of the manipulator, we only in one axis, and rotate the finger
    with different motor to reach the whole surface
    '''

    rospy.init_node('target_finger_broadcaster_parametrized')
    #  self.__make_cylinder(name, pose, height, radius)
    # Finger properties -->  x,  y, z,  cyl height , cyl radius
    finger_props = [0.4 + 0.004, -0.002, 0.15, 0.016, 0.0125]
    start_h = 0.0

    br = tf.TransformBroadcaster()
    rate = rospy.Rate(100.0)

    N = 1
    M = 7
    G = 2.0
    K = 1.5

    while not rospy.is_shutdown():

        # Cylinder push points
        theta = np.linspace(0, 2 * np.pi, N)
        H = np.linspace(start_h, finger_props[3] + start_h, M)

        f = finger_props[4]

        counter = 0

        for j, h in enumerate(H):

            # base
            br.sendTransform((finger_props[0] - h, finger_props[1], finger_props[2]),
                             (0, -0.7071068, 0, 0.7071068),
                             rospy.Time.now(),
                             "fs_" + str(j),
                             "world")
            #

            for i, q in enumerate(theta):

                H = np.asarray([
                    f * np.cos(q),
                    f * np.sin(q),
                    0,
                ])

                H2 = np.asarray([
                    f * G * np.cos(q),
                    f * G * np.sin(q),
                    0,
                ])

                H3 = np.asarray([
                    f * K * G * np.cos(q),
                    f * K * G * np.sin(q),
                    0,
                ])

                Rz = rotation_matrix(q, zaxis)
                Rt = rotation_matrix(np.pi, zaxis)
                Rx = rotation_matrix(np.pi, xaxis)

                rot = concatenate_matrices(Rz, Rt, Rx)[:3, :3]

                rot = R.from_dcm(rot[:3, :3]).as_quat()

                br.sendTransform((H[0], H[1], H[2]),
                                 rot,
                                 rospy.Time.now(),
                                 "pp_" + str(counter),
                                 "fs_" + str(j))

                br.sendTransform((H2[0], H2[1], H2[2]),
                                 rot,
                                 rospy.Time.now(),
                                 "sp_" + str(counter),
                                 "fs_" + str(j))

                br.sendTransform((H3[0], H3[1], H3[2]),
                                 rot,
                                 rospy.Time.now(),
                                 "tp_" + str(counter),
                                 "fs_" + str(j))

                counter += 1

        # Sphere push points
        NN = 1
        MM = 5

        theta = np.linspace(0, 2 * np.pi, NN)
        H = np.linspace(0, finger_props[4], MM)

        f = np.sqrt(finger_props[4] ** 2 - H ** 2)

        counter = N * M

        for j, p in enumerate(H):

            for i, q in enumerate(theta):

                B = np.asarray([
                    f[j] * np.cos(q),
                    f[j] * np.sin(q),
                    p,
                ])

                B2 = np.asarray([
                    f[j] * G * np.cos(q),
                    f[j] * G * np.sin(q),
                    p * G,
                ])

                B3 = np.asarray([
                    f[j] * K * G * np.cos(q),
                    f[j] * K * G * np.sin(q),
                    p * K * G,
                ])

                Rz = rotation_matrix(q, zaxis)
                Rt = rotation_matrix(np.pi, zaxis)
                Rx = rotation_matrix(np.pi, xaxis)

                rot = concatenate_matrices(Rz, Rt, Rx)[:3, :3]

                rot = R.from_dcm(rot).as_quat()

                br.sendTransform((B[0], B[1], B[2]),
                                 rot,
                                 rospy.Time.now(),
                                 "pp_" + str(counter),
                                 "fs_" + str(M - 1))

                br.sendTransform((B2[0], B2[1], B2[2]),
                                 rot,
                                 rospy.Time.now(),
                                 "sp_" + str(counter),
                                 "fs_" + str(M - 1))

                br.sendTransform((B3[0], B3[1], B3[2]),
                                 rot,
                                 rospy.Time.now(),
                                 "tp_" + str(counter),
                                 "fs_" + str(M - 1))

                counter += 1

        rate.sleep()
