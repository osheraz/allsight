#!/usr/bin/env python

import rospy
import numpy as np
from envs.robots import ExperimentEnv, fix_
from envs.env_utils.logger import DataLogger
import cv2
import os

'''
    We will look over [0, 2pi] by N jumps, and press M times each positions.
    We save every 0.5 sec until reaching the maximum force.
    Should be moved to a config file.
'''


def main():
    rospy.init_node('data_collection')

    save = True
    start_from = 8
    up_to = 11
    max_press_time = 7.0
    max_pressure = 6.0
    save_every_sec = 0.003

    sensor_id = 11
    indenter = 'square'
    leds = 'rrrgggbbb'
    gel = 'markers'
    N = 10

    conf = {'method': 'tip_press',
            'save': save,
            'up_to': up_to,
            'start_from': start_from,
            'N': N,
            'max_press_time': max_press_time,
            'max_pressure': max_pressure,
            'leds': leds,
            'gel': gel,
            'sensor_id': sensor_id,
            'indenter': indenter,
            'save_every_sec': save_every_sec}

    log = DataLogger(conf)

    Q = np.interp(np.linspace(0, 2 * np.pi, N), [0, 2 * np.pi], [0, 1.0]).tolist()
    ros_rate = rospy.Rate(100)

    env = ExperimentEnv()
    success = env.ready

    if success:

        # Define the workspace around the arm with the finger

        env.arm.move_manipulator.define_workspace_at_init()
        env.arm.move_manipulator.reach_named_position('home')
        env.arm.calib_robotiq()
        # env.arm.move_manipulator.set_constraints()

        ref_frame = env.finger.last_frame
        ref_img_color_path = os.path.join(log.dataset_path_images, 'ref_frame.jpg')

        if save:
            if not cv2.imwrite(ref_img_color_path, ref_frame):
                raise Exception("Could not write image")

        for q in Q:

            env.finger_base_act.set_angle(q)

            for i in range(start_from, up_to, 1):

                fix = fix_[i]
                if i == 7: continue

                try:

                    # Move a bit higher
                    sx = 0.05 if i <= start_from else 0.01
                    env.arm.move_manipulator.define_workspace_back()
                    env.arm.move_manipulator.scale_vel(scale_vel=sx, scale_acc=0.01)

                    (trans, rot) = env.listener.lookupTransform('/world', '/tp_' + str(i), rospy.Time(0))
                    trans[1] += fix
                    success &= env.arm.set_ee_pose_from_trans_rot(trans, rot, wait=True)

                    (trans, rot) = env.listener.lookupTransform('/world', '/sp_' + str(i), rospy.Time(0))
                    trans[1] += fix
                    success &= env.arm.set_ee_pose_from_trans_rot(trans, rot, wait=True)

                    env.arm.calib_robotiq()
                    rospy.sleep(2.0)
                    env.arm.calib_robotiq()

                    # Start the press
                    env.arm.move_manipulator.define_workspace_press()
                    env.arm.move_manipulator.scale_vel(scale_vel=0.001, scale_acc=0.001)
                    (trans, rot) = env.listener.lookupTransform('/world', '/pp_' + str(i), rospy.Time(0))
                    trans[1] += fix
                    success &= env.arm.set_ee_pose_from_trans_rot(trans, rot, wait=False)

                    ft_init = env.arm.robotiq_wrench_filtered_state
                    start_press_time = rospy.get_time()
                    cur_press_time = rospy.get_time() - start_press_time
                    save_time = cur_press_time

                    # get ref image
                    ft, frame, (trans, rot), (trans_ee, rot_ee) = env.get_obs(i)
                    ft -= ft_init
                    trans_ee[1] -= fix
                    log.append(i, q, frame, trans, rot, trans_ee, rot_ee, ft)

                    # Start the press
                    while cur_press_time < max_press_time and abs(ft[2]) < max_pressure:

                        cur_press_time = rospy.get_time() - start_press_time

                        ft = env.arm.robotiq_wrench_filtered_state - ft_init

                        if abs(ft[2]) > max_pressure / 10 and cur_press_time - save_time > save_every_sec:
                            save_time = cur_press_time
                            ft, frame, (trans, rot), (trans_ee, rot_ee) = env.get_obs(i)
                            ft -= ft_init
                            trans_ee[1] -= fix
                            log.append(i, q, frame, trans, rot, trans_ee, rot_ee, ft, save_time)

                        ros_rate.sleep()

                    env.arm.move_manipulator.stop_motion()

                    rospy.sleep(0.2)

                    # get last image
                    ft, frame, (trans, rot), (trans_ee, rot_ee) = env.get_obs(i)
                    ft -= ft_init
                    trans_ee[1] -= fix
                    log.append(i, q, frame, trans, rot, trans_ee, rot_ee, ft, rospy.get_time() - start_press_time)

                    env.arm.move_manipulator.define_workspace_back()
                    env.arm.move_manipulator.scale_vel(scale_vel=sx, scale_acc=0.01)
                    (trans, rot) = env.listener.lookupTransform('/world', '/sp_' + str(i), rospy.Time(0))
                    trans[1] += fix
                    success &= env.arm.set_ee_pose_from_trans_rot(trans, rot, wait=True)

                    (trans, rot) = env.listener.lookupTransform('/world', '/tp_' + str(i), rospy.Time(0))
                    trans[1] += fix
                    success &= env.arm.set_ee_pose_from_trans_rot(trans, rot, wait=True)

                    if save: log.save_batch_images()

                except Exception as e:
                    print(e)
                    rospy.logwarn('Failed to reach push_point #{}, verify.'.format(i))
                    success = False
                    continue

            # env.arm.move_manipulator.clear_all_constraints()
            env.arm.move_manipulator.define_workspace_at_init()
            (trans, rot) = env.listener.lookupTransform('/world', '/tp_9', rospy.Time(0))
            trans[1] += fix
            success &= env.arm.set_ee_pose_from_trans_rot(trans, rot, wait=True)
            (trans, rot) = env.listener.lookupTransform('/world', '/tp_8', rospy.Time(0))
            trans[1] += fix
            success &= env.arm.set_ee_pose_from_trans_rot(trans, rot, wait=True)
            (trans, rot) = env.listener.lookupTransform('/world', '/tp_7', rospy.Time(0))
            trans[1] += fix
            success &= env.arm.set_ee_pose_from_trans_rot(trans, rot, wait=True)

            # env.arm.move_manipulator.scale_vel(scale_vel=0.05, scale_acc=0.01)
            # env.arm.move_manipulator.reach_named_position('start')
            # env.arm.move_manipulator.set_constraints()

            if save: log.save_data_dict()

        rospy.loginfo('finished experiment, everything was OK, lets submit'.format(success))

        for q in Q[::-1]:
            rospy.loginfo('Restarting env.')
            env.finger_base_act.set_angle(q)
            rospy.sleep(1.0)


if __name__ == '__main__':
    main()
