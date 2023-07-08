import numpy
import rospy
import tf
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.msg import JointLimits
import moveit_msgs.msg
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
import copy
from robotiq_force_torque_sensor.srv import sensor_accessor
import robotiq_force_torque_sensor.srv as ft_srv
from robotiq_force_torque_sensor.msg import *
from openhand_node.srv import MoveServos, ReadServos
from finger_ros import TactileSubscriberFinger

fix_ = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: -0.001, 9: -0.003, 10: -0.005, 11: -0.006}


class ExperimentEnv():
    """ Superclass for all Robots environments.
    """

    def __init__(self, ):
        rospy.logwarn('Setting up the environment')
        self.finger_base_act = SingleDynamixel()
        self.finger = TactileSubscriberFinger()
        self.arm = RobotWithFtEnv()
        self.listener = tf.TransformListener()
        rospy.sleep(2)
        self.ready = self.arm.init_success and self.finger.init_success

    def get_obs(self, i):

        rospy.sleep(0.03)
        ft = self.arm.robotiq_wrench_filtered_state
        frame = self.finger.last_frame
        (trans, rot) = self.listener.lookupTransform('/fs_0', '/pp_' + str(i), rospy.Time(0))
        (trans_ee, rot_ee) = self.listener.lookupTransform('/fs_0', 'end_effector_link', rospy.Time(0))

        return ft, frame, (trans, rot), (trans_ee, rot_ee)


class RobotWithFtEnv():
    """ Superclass for all Robots environments.
    """

    def __init__(self, robot='Manipulator'):

        self.wait_env_ready()
        self.robotiq_wrench_filtered_state = numpy.array([0, 0, 0, 0, 0, 0])
        self.init_success = self._check_all_systems_ready()

        # Define feedback callbacks
        self.ft_zero = rospy.ServiceProxy('/robotiq_force_torque_sensor_acc', sensor_accessor)
        rospy.Subscriber("/arm_controller/state", JointTrajectoryControllerState, self._joint_state_callback)
        rospy.Subscriber('/robotiq_force_torque_wrench_filtered_exp', WrenchStamped,
                         self._robotiq_wrench_states_callback)

        self.move_manipulator = MoveManipulator() if robot == 'Manipulator' else None

    def calib_robotiq(self):

        # self.ft_zero.call(8, "")
        rospy.sleep(0.5)
        msg = ft_srv.sensor_accessorRequest()
        msg.command = "SET ZRO"
        suc_zero = True
        self.robotiq_wrench_filtered_state *= 0
        for _ in range(5):
            result = self.ft_zero(msg)
            rospy.sleep(0.5)
            if 'Done' not in str(result):
                suc_zero &= False
                rospy.logerr('Failed calibrating the F\T')
            else:
                suc_zero &= True
        return suc_zero

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other robot systems are
        operational.
        """
        rospy.logdebug("ManipulatorOpenhandRealEnv check_all_systems_ready...")
        self._check_arm_joint_state_ready()
        self._check_robotiq_connection()
        rospy.logdebug("END ManipulatorOpenhandRealEnv _check_all_systems_ready...")
        return True

    def _check_arm_joint_state_ready(self):

        self.arm_joint_state = None
        rospy.logdebug(
            "Waiting for arm_controller/state to be READY...")
        while self.arm_joint_state is None and not rospy.is_shutdown():
            try:
                self.arm_joint_state = rospy.wait_for_message(
                    "arm_controller/state", JointTrajectoryControllerState, timeout=5.0)
                rospy.logdebug(
                    "Current arm_controller/state READY=>")

            except:
                rospy.logerr(
                    "Current arm_controller/state not ready yet, retrying for getting laser_scan")
        return self.arm_joint_state

    def _check_robotiq_connection(self):

        self.robotiq_wrench_filtered_state = numpy.array([0, 0, 0, 0, 0, 0])
        rospy.logdebug(
            "Waiting for robotiq_force_torque_wrench_filtered to be READY...")
        while not numpy.sum(self.robotiq_wrench_filtered_state) and not rospy.is_shutdown():
            try:
                self.robotiq_wrench_filtered_state = rospy.wait_for_message(
                    "robotiq_force_torque_wrench_filtered", WrenchStamped, timeout=5.0)
                self.robotiq_wrench_filtered_state = numpy.array([self.robotiq_wrench_filtered_state.wrench.force.x,
                                                                  self.robotiq_wrench_filtered_state.wrench.force.y,
                                                                  self.robotiq_wrench_filtered_state.wrench.force.z,
                                                                  self.robotiq_wrench_filtered_state.wrench.torque.x,
                                                                  self.robotiq_wrench_filtered_state.wrench.torque.y,
                                                                  self.robotiq_wrench_filtered_state.wrench.torque.z])
                rospy.logdebug(
                    "Current robotiq_force_torque_wrench_filtered READY=>")
            except:
                rospy.logerr(
                    "Current robotiq_force_torque_wrench_filtered not ready yet, retrying")

        return self.robotiq_wrench_filtered_state

    def wait_env_ready(self):

        import time

        for i in range(1):
            print("WAITING..." + str(i))
            sys.stdout.flush()
            time.sleep(1.0)

        print("WAITING...DONE")

    def _joint_state_callback(self, data):
        self.arm_joint_state = data

    def _robotiq_wrench_states_callback(self, data):

        self.robotiq_wrench_filtered_state = numpy.array([data.wrench.force.x,
                                                          data.wrench.force.y,
                                                          data.wrench.force.z,
                                                          data.wrench.torque.x,
                                                          data.wrench.torque.y,
                                                          data.wrench.torque.z])

    def rotate_pose_by_rpy(self, in_pose, roll, pitch, yaw, wait=True):
        """
        Apply an RPY rotation to a pose in its parent coordinate system.
        """
        try:
            if in_pose.header:  # = in_pose is a PoseStamped instead of a Pose.
                in_pose.pose = self.rotate_pose_by_rpy(in_pose.pose, roll, pitch, yaw, wait)
                return in_pose
        except:
            pass
        q_in = [in_pose.orientation.x, in_pose.orientation.y, in_pose.orientation.z, in_pose.orientation.w]
        q_rot = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        q_rotated = tf.transformations.quaternion_multiply(q_in, q_rot)

        rotated_pose = copy.deepcopy(in_pose)
        rotated_pose.orientation = geometry_msgs.msg.Quaternion(*q_rotated)
        result = self.move_manipulator.ee_traj_by_pose_target(rotated_pose, wait)
        return rotated_pose

    def set_trajectory_ee(self, action_xyz, action_theta=None, wait=True):
        """
        Sets the Pose of the EndEffector based on the action variable.
        The action variable contains the position and orientation of the EndEffector.
        See create_action
        """
        # Set up a trajectory message to publish.
        if action_theta is None:
            action_theta = [0.0, 0.0, 0.0]

        action_xyz = action_xyz if isinstance(action_xyz, list) else action_xyz.tolist()
        if len(action_xyz) < 3: action_xyz.append(0.0)  # xy -> xyz
        ee_target = self.get_ee_pose().pose
        cur_ee_pose = self.get_ee_pose()

        if numpy.any(action_theta):
            roll, pitch, yaw = tf.transformations.euler_from_quaternion((cur_ee_pose.pose.orientation.x,
                                                                         cur_ee_pose.pose.orientation.y,
                                                                         cur_ee_pose.pose.orientation.z,
                                                                         cur_ee_pose.pose.orientation.w))
            roll = roll + action_theta[0] if action_theta[0] else roll
            pitch = pitch + action_theta[1] if action_theta[1] else pitch
            yaw = yaw + action_theta[2] if action_theta[2] else yaw
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            ee_target.orientation.x, ee_target.orientation.y, ee_target.orientation.z, ee_target.orientation.w = quaternion

        if numpy.any(action_xyz):
            ee_target.position.x = cur_ee_pose.pose.position.x + action_xyz[0] if action_xyz[
                0] else cur_ee_pose.pose.position.x
            ee_target.position.y = cur_ee_pose.pose.position.y + action_xyz[1] if action_xyz[
                1] else cur_ee_pose.pose.position.y
            ee_target.position.z = cur_ee_pose.pose.position.z + action_xyz[2] if action_xyz[
                2] else cur_ee_pose.pose.position.z

        result = self.move_manipulator.ee_traj_by_pose_target(ee_target, wait)
        return result

    def set_ee_pose(self, pose, wait=True):
        # Set up a trajectory message to publish.
        ee_target = geometry_msgs.msg.Pose()
        if isinstance(pose, dict):
            ee_target.orientation.x = pose["qx"]
            ee_target.orientation.y = pose["qy"]
            ee_target.orientation.z = pose["qz"]
            ee_target.orientation.w = pose["qw"]

            ee_target.position.x = pose["x"]
            ee_target.position.y = pose["y"]
            ee_target.position.z = pose["z"]
        else:
            ee_target = pose

        result = self.move_manipulator.ee_traj_by_pose_target(ee_target, wait)
        return result

    def set_ee_pose_from_trans_rot(self, trans, rot, wait=True):
        # Set up a trajectory message to publish.

        req_pose = self.get_ee_pose()

        req_pose.pose.position.x = trans[0]
        req_pose.pose.position.y = trans[1]
        req_pose.pose.position.z = trans[2]
        req_pose.pose.orientation.x = rot[0]
        req_pose.pose.orientation.y = rot[1]
        req_pose.pose.orientation.z = rot[2]
        req_pose.pose.orientation.w = rot[3]

        result = self.set_ee_pose(req_pose, wait=wait)
        return result

    def set_trajectory_joints(self, initial_qpos):

        positions_array = [None] * 6
        positions_array[0] = initial_qpos["joint1"]
        positions_array[1] = initial_qpos["joint2"]
        positions_array[2] = initial_qpos["joint3"]
        positions_array[3] = initial_qpos["joint4"]
        positions_array[4] = initial_qpos["joint5"]
        positions_array[5] = initial_qpos["joint6"]

        self.move_manipulator.joint_traj(positions_array)

        return True

    def get_ee_pose(self):
        """
        Returns geometry_msgs/PoseStamped
            std_msgs/Header header
              uint32 seq
              time stamp
              string frame_id
            geometry_msgs/Pose pose
              geometry_msgs/Point position
                float64 x
                float64 y
                float64 z
              geometry_msgs/Quaternion orientation
                float64 x
                float64 y
                float64 z
                float64 w
        """
        gripper_pose = self.move_manipulator.ee_pose()
        return gripper_pose

    def get_ee_rpy(self):
        gripper_rpy = self.move_manipulator.ee_rpy()
        return gripper_rpy


class MoveManipulator():

    def __init__(self):
        rospy.logdebug("===== In MoveManipulator")
        moveit_commander.roscpp_initialize(sys.argv)
        arm_group_name = "arm"

        self.robot = moveit_commander.RobotCommander("robot_description")
        self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
        self.group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        rospy.logdebug("===== Out MoveManipulator")
        # self.get_planning_feedback()

    def scale_vel(self, scale_vel, scale_acc):
        self.group.set_max_velocity_scaling_factor(scale_vel)
        self.group.set_max_acceleration_scaling_factor(scale_acc)

    def set_constraints(self):
        joint_constraint_list = []
        above = below = 0.007
        # joint_constraint = moveit_msgs.msg.JointConstraint()
        # joint_constraint.joint_name = self.group.get_joints()[1]
        # joint_constraint.position = 0.0
        # joint_constraint.tolerance_above = above
        # joint_constraint.tolerance_below = below
        # joint_constraint.weight = 1.0
        # joint_constraint_list.append(joint_constraint)

        joint_constraint = moveit_msgs.msg.JointConstraint()
        joint_constraint.joint_name = self.group.get_joints()[4]
        joint_constraint.position = 0.0
        joint_constraint.tolerance_above = above
        joint_constraint.tolerance_below = below
        joint_constraint.weight = 1.0
        joint_constraint_list.append(joint_constraint)

        # joint_constraint = moveit_msgs.msg.JointConstraint()
        # joint_constraint.joint_name = self.group.get_joints()[6]
        # joint_constraint.position = 0.0
        # joint_constraint.tolerance_above = above
        # joint_constraint.tolerance_below = below
        # joint_constraint.weight = 1.0
        # joint_constraint_list.append(joint_constraint)

        constraint_list = moveit_msgs.msg.Constraints()
        constraint_list.name = 'todo'
        constraint_list.joint_constraints = joint_constraint_list

        self.group.set_path_constraints(constraint_list)

    def clear_all_constraints(self):

        self.group.clear_path_constraints()

    def get_planning_feedback(self):
        planning_frame = self.group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # print the name of the end-effector link for this group:
        eef_link = self.group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", group_names)

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())

    def define_workspace_at_init(self):
        # Walls are defined with respect to the coordinate frame of the robot base, with directions
        # corresponding to standing behind the robot and facing into the table.
        rospy.sleep(0.6)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'

        # self.robot.get_planning_frame()
        table_pose = PoseStamped()
        table_pose.header = header
        table_pose.pose.position.x = 0
        table_pose.pose.position.y = 0
        table_pose.pose.position.z = -0.0001
        self.scene.remove_world_object('bottom')
        self.scene.add_plane(name='bottom', pose=table_pose, normal=(0, 0, 1))

        upper_pose = PoseStamped()
        upper_pose.header = header
        upper_pose.pose.position.x = 0
        upper_pose.pose.position.y = 0
        upper_pose.pose.position.z = 0.6
        self.scene.remove_world_object('upper')
        self.scene.add_plane(name='upper', pose=upper_pose, normal=(0, 0, 1))

        back_pose = PoseStamped()
        back_pose.header = header
        back_pose.pose.position.x = 0
        back_pose.pose.position.y = -0.4  # -0.25
        back_pose.pose.position.z = 0
        self.scene.remove_world_object('rightWall')
        self.scene.add_plane(name='rightWall', pose=back_pose, normal=(0, 1, 0))

        front_pose = PoseStamped()
        front_pose.header = header
        front_pose.pose.position.x = -0.25
        front_pose.pose.position.y = 0.0  # 0.52 # Optimized (0.55 NG)
        front_pose.pose.position.z = 0
        self.scene.remove_world_object('backWall')
        # self.scene.add_plane(name='backWall', pose=front_pose, normal=(1, 0, 0))

        right_pose = PoseStamped()
        right_pose.header = header
        right_pose.pose.position.x = 0.45  # 0.2
        right_pose.pose.position.y = 0
        right_pose.pose.position.z = 0
        self.scene.remove_world_object('frontWall')
        self.scene.add_plane(name='frontWall', pose=right_pose, normal=(1, 0, 0))

        left_pose = PoseStamped()
        left_pose.header = header
        left_pose.pose.position.x = 0.0  # -0.54
        left_pose.pose.position.y = 0.4
        left_pose.pose.position.z = 0
        self.scene.remove_world_object('leftWall')
        self.scene.add_plane(name='leftWall', pose=left_pose, normal=(0, 1, 0))
        rospy.sleep(0.6)

        # Add finger to the scene
        # Finger properties -->  x,  y, z,  cyl height , cyl radius
        # finger_props = [0.4 + 0.004,  -0.002, 0.15, 0.02, 0.012 * 2]
        finger_props = [rospy.get_param('/finger/x'),
                        rospy.get_param('/finger/y'),
                        rospy.get_param('/finger/z'),
                        rospy.get_param('/finger/h') + 0.004,
                        rospy.get_param('/finger/r') * 2]

        #  self.__make_cylinder(name, pose, height, radius)
        h = finger_props[3]
        header = Header()
        header.stamp = rospy.Time(0)
        header.frame_id = 'world'

        self.scene.remove_world_object('finger_body')
        self.scene.remove_world_object('finger_cone')

        finger_body = geometry_msgs.msg.PoseStamped()
        finger_body.header = header
        finger_body.pose.position.x = finger_props[0] - h / 2
        finger_body.pose.position.y = finger_props[1]
        finger_body.pose.position.z = finger_props[2]
        finger_body.pose.orientation.x = 0
        finger_body.pose.orientation.y = -0.7071068
        finger_body.pose.orientation.z = 0
        finger_body.pose.orientation.w = 0.7071068
        self.scene.add_cylinder("finger_body", finger_body, finger_props[3], finger_props[4])

        finger_cone = geometry_msgs.msg.PoseStamped()
        finger_cone.header = header
        finger_cone.pose.position.x = finger_props[0] - h
        finger_cone.pose.position.y = finger_props[1]
        finger_cone.pose.position.z = finger_props[2]
        finger_body.pose.orientation.x = 0
        finger_body.pose.orientation.y = -0.7071068
        finger_body.pose.orientation.z = 0
        finger_body.pose.orientation.w = 0.7071068
        self.scene.add_sphere("finger_cone", finger_cone, finger_props[4])

    def define_workspace_press(self):
        # Walls are defined with respect to the coordinate frame of the robot base, with directions
        # corresponding to standing behind the robot and facing into the table.
        rospy.sleep(0.6)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'

        # Add finger to the scene
        # Finger properties -->  x,  y, z,  cyl height , cyl radius
        # finger_props = [0.4 + 0.004,  -0.002, 0.15, 0.016, 0.0125]
        finger_props = [rospy.get_param('/finger/x'),
                        rospy.get_param('/finger/y'),
                        rospy.get_param('/finger/z'),
                        rospy.get_param('/finger/h'),
                        rospy.get_param('/finger/r') - 0.003]

        self.scene.remove_world_object('finger_body')
        self.scene.remove_world_object('finger_cone')

        #  self.__make_cylinder(name, pose, height, radius)
        h = finger_props[3]
        header = Header()
        header.stamp = rospy.Time(0)
        header.frame_id = 'world'

        finger_body = geometry_msgs.msg.PoseStamped()
        finger_body.header = header
        finger_body.pose.position.x = finger_props[0] - h / 2
        finger_body.pose.position.y = finger_props[1]
        finger_body.pose.position.z = finger_props[2]
        finger_body.pose.orientation.x = 0
        finger_body.pose.orientation.y = -0.7071068
        finger_body.pose.orientation.z = 0
        finger_body.pose.orientation.w = 0.7071068
        self.scene.add_cylinder("finger_body", finger_body, finger_props[3], finger_props[4])

        finger_cone = geometry_msgs.msg.PoseStamped()
        finger_cone.header = header
        finger_cone.pose.position.x = finger_props[0] - h
        finger_cone.pose.position.y = finger_props[1]
        finger_cone.pose.position.z = finger_props[2]
        finger_body.pose.orientation.x = 0
        finger_body.pose.orientation.y = -0.7071068
        finger_body.pose.orientation.z = 0
        finger_body.pose.orientation.w = 0.7071068
        self.scene.add_sphere("finger_cone", finger_cone, finger_props[4])

    def define_workspace_back(self):
        # Walls are defined with respect to the coordinate frame of the robot base, with directions
        # corresponding to standing behind the robot and facing into the table.
        rospy.sleep(0.6)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'

        # Add finger to the scene
        # Finger properties -->  x,  y, z,  cyl height , cyl radius
        # finger_props = [0.4 + 0.004, -0.002, 0.15, 0.016, 0.0125]
        finger_props = [rospy.get_param('/finger/x'),
                        rospy.get_param('/finger/y'),
                        rospy.get_param('/finger/z'),
                        rospy.get_param('/finger/h'),
                        rospy.get_param('/finger/r')]

        self.scene.remove_world_object('finger_body')
        self.scene.remove_world_object('finger_cone')

        #  self.__make_cylinder(name, pose, height, radius)
        h = finger_props[3]
        header = Header()
        header.stamp = rospy.Time(0)
        header.frame_id = 'world'

        finger_body = geometry_msgs.msg.PoseStamped()
        finger_body.header = header
        finger_body.pose.position.x = finger_props[0] - h / 2
        finger_body.pose.position.y = finger_props[1]
        finger_body.pose.position.z = finger_props[2]
        finger_body.pose.orientation.x = 0
        finger_body.pose.orientation.y = -0.7071068
        finger_body.pose.orientation.z = 0
        finger_body.pose.orientation.w = 0.7071068
        self.scene.add_cylinder("finger_body", finger_body, finger_props[3], finger_props[4])

        finger_cone = geometry_msgs.msg.PoseStamped()
        finger_cone.header = header
        finger_cone.pose.position.x = finger_props[0] - h
        finger_cone.pose.position.y = finger_props[1]
        finger_cone.pose.position.z = finger_props[2]
        finger_body.pose.orientation.x = 0
        finger_body.pose.orientation.y = -0.7071068
        finger_body.pose.orientation.z = 0
        finger_body.pose.orientation.w = 0.7071068
        self.scene.add_sphere("finger_cone", finger_cone, finger_props[4])

    def all_close(self, goal, actual, tolerance):
        """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
        all_equal = True
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.all_close(goal.pose, actual.pose, tolerance)

        # elif type(goal) is geometry_msgs.msg.Pose:
        #     return self.all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

        return True

    def reach_named_position(self, target):
        arm_group = self.group

        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        arm_group.set_named_target(target)
        return arm_group.go(wait=True)

    def ee_traj_by_cartesian_path(self, pose, wait=True):
        # self.group.set_pose_target(pose)
        # result = self.execute_trajectory(wait)
        cartesian_plan, fraction = self.plan_cartesian_path(pose)
        result = self.execute_plan(cartesian_plan, wait)
        return result

    def ee_traj_by_pose_target(self, pose, wait=True, tolerance=0.0001):  # 0.0001

        self.group.set_goal_position_tolerance(tolerance)
        self.group.set_pose_target(pose)
        result = self.execute_trajectory(wait)
        return result

    def get_cartesian_pose(self, verbose=False):
        arm_group = self.group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()
        if verbose:
            rospy.loginfo("Actual cartesian pose is : ")
            rospy.loginfo(pose.pose)

        return pose.pose

    def joint_traj(self, positions_array):

        self.group_variable_values = self.group.get_current_joint_values()
        self.group_variable_values[0] = positions_array[0]
        self.group_variable_values[1] = positions_array[1]
        self.group_variable_values[2] = positions_array[2]
        self.group_variable_values[3] = positions_array[3]
        self.group_variable_values[4] = positions_array[4]
        self.group_variable_values[5] = positions_array[5]
        self.group.set_joint_value_target(self.group_variable_values)
        result = self.execute_trajectory()

        return result

    def execute_trajectory(self, wait=True):
        """
        Assuming that the trajecties has been set to the self objects appropriately
        Make a plan to the destination in Homogeneous Space(x,y,z,yaw,pitch,roll)
        and returns the result of execution
        """
        self.plan = self.group.plan()
        result = self.group.go(wait=wait)
        self.group.clear_pose_targets()

        return result

    def ee_pose(self):
        gripper_pose = self.group.get_current_pose()
        return gripper_pose

    def ee_rpy(self):
        gripper_rpy = self.group.get_current_rpy()
        return gripper_rpy

    def plan_cartesian_path(self, pose, eef_step=0.001):

        waypoints = []
        # start with the current pose
        # waypoints.append(self.arm_group.get_current_pose().pose)

        wpose = self.group.get_current_pose().pose  # geometry_msgs.msg.Pose()
        wpose.position.x = pose.position.x
        wpose.position.y = pose.position.y
        wpose.position.z = pose.position.z
        # wpose.orientation.x = pose.orientation.x
        # wpose.orientation.y = pose.orientation.y
        # wpose.orientation.z = pose.orientation.z
        # wpose.orientation.w = pose.orientation.w

        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints,  # waypoints to follow
            eef_step,  # eef_step
            2.0)  # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet
        return plan, fraction

    def execute_plan(self, plan, wait=True):
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        result = self.group.execute(plan, wait=wait)
        self.group.clear_pose_targets()
        return result

    def stop_motion(self):

        self.group.stop()
        self.group.clear_pose_targets()


class SingleDynamixel():
    """ angles are between 0 to 1."""

    def __init__(self):
        self.zero = 0
        self.move_servo_srv = rospy.ServiceProxy('/openhand_reset_node/move_servos', MoveServos)
        self.read_servo_srv = rospy.ServiceProxy('/openhand_reset_node/read_servos', ReadServos)
        self.move_servo_srv.call([0])

    def get_angle(self):
        return self.read_servo_srv() - self.zero

    def set_angle(self, angle):
        print('Setting new angle: {}'.format(angle))
        self.move_servo_srv.call([self.zero + angle])

    def set_zero(self, angle):
        print('setting new zero: '.format(angle))
        self.zero = angle
        self.move_servo_srv.call([self.zero])


# #####################################
# #####################################
#
# calibrate = True
# if calibrate:
#     from getkey import getkey
#     angle, dq = 0, 0.05
#     while calibrate:
#         key = getkey()
#         print(key)
#         if str(key) == 'q':
#             calibrate = False
#         if str(key) == 'w':
#             angle += dq
#         if str(key) == 'e':
#             angle -= dq
#
#         angle = max(0, min(angle, 0.3))
#         env.arm.finger_base_act.set_zero(angle)
#
# #####################################
# #####################################