#!/usr/bin/env python3

import sys
import rospy

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import shape_msgs.msg
import visualization_msgs.msg

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped

from interbotix_xs_msgs.msg import JointGroupCommand

from tf.transformations import quaternion_from_euler
import tf

import numpy as np
import time

import time

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("graspover", anonymous=True)
pub = rospy.Publisher('/rviz_visual_tools0', visualization_msgs.msg.MarkerArray , queue_size=10)
pub_camera = rospy.Publisher('/locobot/commands/joint_group', JointGroupCommand , queue_size=1)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
arm = moveit_commander.MoveGroupCommander("interbotix_arm")
gripper = moveit_commander.MoveGroupCommander("interbotix_gripper")

arm.set_max_velocity_scaling_factor(0.5)
arm.set_max_acceleration_scaling_factor(0.5)
gripper.set_max_velocity_scaling_factor(0.5)
gripper.set_max_acceleration_scaling_factor(0.5)

camera_state = JointGroupCommand()
camera_state.name = 'camera'
camera_state.cmd = [0, 0.9]

pub_camera.publish(camera_state)

time.sleep(1)

scene.remove_world_object()	
eef_link = arm.get_end_effector_link()
scene.remove_attached_object(eef_link)

pose_floor = PoseStamped()
pose_floor.header.frame_id= "locobot/base_footprint"
scene.add_box("floor", pose_floor, (2, 2, 0.001))

arm_handover_joints = [-1.6275537014007568, 0.2316311001777649, 1.0446408987045288, 1.512505054473877, 1.5800002813339233, -2.8163888454437256]

gripper_pose = rospy.wait_for_message('/customid/gripper_pose', PoseStamped)

gripper.set_joint_value_target([0.036,-0.036])
gripper.go()

gripper_pose.pose.position.z = gripper_pose.pose.position.z + 0.1
gripper_pose.pose.position.x = gripper_pose.pose.position.x + 0.01
gripper_pose.pose.position.y = gripper_pose.pose.position.y + 0.02

"""pose_point = PoseStamped()
pose_point.header.frame_id= "locobot/base_footprint"
pose_point.pose = goal1"""

arm.set_pose_target(gripper_pose.pose)
plan = arm.go(wait=True)
arm.stop()
arm.clear_pose_targets()

gripper_pose.pose.position.z = gripper_pose.pose.position.z - 0.1


arm.set_pose_target(gripper_pose.pose)
plan = arm.go(wait=True)
arm.stop()
arm.clear_pose_targets()


gripper.set_joint_value_target([0.017,-0.017])
gripper.go()

scene.add_box("box", gripper_pose, (0.02, 0.02, 0.22))
touch_links = robot.get_link_names(group='interbotix_gripper')
print(touch_links)
scene.attach_box(eef_link, "box", touch_links=touch_links)


arm.set_joint_value_target(arm_handover_joints)
arm.go()

camera_state.cmd = [0, 0.0]

pub_camera.publish(camera_state)

time.sleep(1)

ee_link = arm.get_end_effector_link()
print(ee_link)
current_pose = arm.get_current_pose(ee_link)
current = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z])
arm_base_pose = arm.get_current_pose("locobot/arm_base_link")
arm_base = np.array([arm_base_pose.pose.position.x, arm_base_pose.pose.position.y, arm_base_pose.pose.position.z])

listener = tf.TransformListener()

while not rospy.is_shutdown():
    try:
        hand = rospy.wait_for_message('/locobot/customid/hand', PointStamped)
        hand_trs = listener.transformPoint('locobot/base_footprint', hand)
        break
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        print(e)

sphere_world = np.array([hand_trs.point.x, hand_trs.point.y, hand_trs.point.z]) 
print('!!!')
print(sphere_world)
print('!!!')
sphere = sphere_world - arm_base
angleZ = np.arctan2(sphere[1], sphere[0])
abs = np.sqrt(sphere[0]**2 + sphere[1]**2 + sphere[2]**2)

factor = 0.2/abs
point = sphere - sphere*factor + arm_base

quaternion = quaternion_from_euler(0, 0, angleZ)

pose_sphere = geometry_msgs.msg.PoseStamped()
pose_sphere.header.frame_id = "locobot/base_footprint"
pose_sphere.pose.orientation.w = 1.0
pose_sphere.pose.position.x = sphere_world[0]
pose_sphere.pose.position.y = sphere_world[1]
pose_sphere.pose.position.z = sphere_world[2]

pose_point = geometry_msgs.msg.PoseStamped()
pose_point.header.frame_id= "locobot/base_footprint"
pose_point.pose.orientation.x = quaternion[0]
pose_point.pose.orientation.y = quaternion[1]
pose_point.pose.orientation.z = quaternion[2]
pose_point.pose.orientation.w = quaternion[3]
pose_point.pose.position.x = point[0]
pose_point.pose.position.y = point[1]
pose_point.pose.position.z = point[2]

pose_start = geometry_msgs.msg.PoseStamped()
pose_start.header.frame_id= "locobot/base_footprint"
pose_start.pose.orientation.x = quaternion[0]
pose_start.pose.orientation.y = quaternion[1]
pose_start.pose.orientation.z = quaternion[2]
pose_start.pose.orientation.w = quaternion[3]
pose_start.pose.position.x = current[0]
pose_start.pose.position.y = current[1]
pose_start.pose.position.z = current[2]
#current = np.array([0,0,0])
line = point - current
#line_angleZ = np.arctan2(line[1], line[0])
line_angleY = np.arctan2(line[2], line[0])
print(line_angleY)
component_x = line[0]/(np.sin(np.deg2rad(90)-line_angleY))
print(component_x)
print(line[0])
line_angleZ = np.arctan2(line[1], component_x)
#print(line_angleZ)
#line_angleX = np.arctan2(line[1], line[2])
line_quaternion = quaternion_from_euler(-line_angleY, line_angleZ, 0, 'ryzx')
print(line_quaternion)


pose_line = geometry_msgs.msg.PoseStamped()
pose_line.header.frame_id= "locobot/base_footprint"
pose_line.pose.orientation.x = line_quaternion[0]
pose_line.pose.orientation.y = line_quaternion[1]
pose_line.pose.orientation.z = line_quaternion[2]
pose_line.pose.orientation.w = line_quaternion[3]
pose_line.pose.position = current_pose.pose.position
"""pose_line.pose.position.x = 0
pose_line.pose.position.y = 0
pose_line.pose.position.z = 0"""

"""plane_quaternion = quaternion_from_euler(-line_angleY,0, 0, 'ryzx')
pose_plane = geometry_msgs.msg.PoseStamped()
pose_plane.header.frame_id= "locobot/base_footprint"
pose_plane.pose.orientation.x = plane_quaternion[0]
pose_plane.pose.orientation.y = plane_quaternion[1]
pose_plane.pose.orientation.z = plane_quaternion[2]
pose_plane.pose.orientation.w = plane_quaternion[3]
pose_plane.pose.position = current_pose.pose.position"""

mks = visualization_msgs.msg.MarkerArray()

hand = visualization_msgs.msg.Marker()
hand.header.frame_id = "locobot/base_footprint"
hand.header.stamp = rospy.Time.now()
hand.pose = pose_sphere.pose
hand.type = visualization_msgs.msg.Marker.SPHERE
hand.action = visualization_msgs.msg.Marker.ADD
hand.id = 0
hand.scale.x = 0.1
hand.scale.y = 0.1
hand.scale.z = 0.1
hand.color.a = 1
hand.color.r = 0
hand.color.g = 1
hand.color.b = 0

poin = visualization_msgs.msg.Marker()
poin.header.frame_id = "locobot/base_footprint"
poin.header.stamp = rospy.Time.now()
poin.pose = pose_point.pose
poin.type = visualization_msgs.msg.Marker.SPHERE
poin.action = visualization_msgs.msg.Marker.ADD
poin.id =1
poin.scale.x = 0.05
poin.scale.y = 0.05
poin.scale.z = 0.05
poin.color.a = 1
poin.color.r = 1
poin.color.g = 0
poin.color.b = 0

line = visualization_msgs.msg.Marker()
line.header.frame_id = "locobot/base_footprint"
line.header.stamp = rospy.Time.now()
line.pose = pose_line.pose
line.type = visualization_msgs.msg.Marker.CUBE
line.action = visualization_msgs.msg.Marker.ADD
line.id = 2
line.scale.x = 1
line.scale.y = 0.01
line.scale.z = 0.01
line.color.a = 1
line.color.r = 1
line.color.g = 1
line.color.b = 0

obj = visualization_msgs.msg.Marker()
obj.header.frame_id = "locobot/base_footprint"
obj.header.stamp = rospy.Time.now()
obj.pose = current_pose.pose
obj.type = visualization_msgs.msg.Marker.CUBE
obj.action = visualization_msgs.msg.Marker.ADD
obj.id = 3
obj.scale.x = 0.02
obj.scale.y = 0.02
obj.scale.z = 0.1
obj.color.a = 1
obj.color.r = 1
obj.color.g = 0
obj.color.b = 1

mks.markers.append(hand)
mks.markers.append(poin)
mks.markers.append(line)
mks.markers.append(obj)

pub.publish(mks)

"""
scene.remove_world_object()
scene.add_sphere("hand", pose_sphere, 0.1)
scene.add_sphere("point", pose_point, 0.05)
#scene.add_box("plane", pose_plane, (1, 3, 0.001))
#scene.add_sphere("gripper", pose_point, 0.02)#(0.05, 0.02, 0.02))
scene.add_box("constraint", pose_line, (1, 0.002, 0.002))"""

line_constraint = moveit_msgs.msg.PositionConstraint()
line_constraint.header.frame_id = "locobot/base_footprint"
line_constraint.link_name = arm.get_end_effector_link()
line_shape = shape_msgs.msg.SolidPrimitive()
line_shape.type = shape_msgs.msg.SolidPrimitive.BOX
line_shape.dimensions = [2, 0.05, 0.05]
line_constraint.constraint_region.primitives.append(line_shape)
line_constraint.constraint_region.primitive_poses.append(pose_line.pose)
line_constraint.weight = 1.0

constraints = moveit_msgs.msg.Constraints()
#constraints.name = "use_equality_constraints";
constraints.position_constraints.append(line_constraint)

orientation_constraint = moveit_msgs.msg.OrientationConstraint()
orientation_constraint.header.frame_id = "locobot/base_footprint"
orientation_constraint.link_name = arm.get_end_effector_link()
orientation_constraint.orientation = pose_point.pose.orientation
orientation_constraint.absolute_x_axis_tolerance = np.deg2rad(10)
orientation_constraint.absolute_y_axis_tolerance = np.deg2rad(10)
orientation_constraint.absolute_z_axis_tolerance = 1
orientation_constraint.weight = 1.0
constraints.orientation_constraints.append(orientation_constraint)

#group.set_pose_target(pose_start.pose)
arm.set_planning_time(10)
#group.set_goal_orientation_tolerance(np.deg2rad(5))
arm.set_num_planning_attempts(3)
#plan = group.go(wait=True)
# Calling stop() ensures that there is no residual movement
#group.stop()
# It is always good to clear your targets after planning with poses.
# Note: there is no equivalent function for clear_joint_value_targets()
#group.clear_pose_targets()

arm.set_path_constraints(constraints)
arm.set_pose_target(pose_point.pose)


plan = arm.go(wait=True)
# Calling stop() ensures that there is no residual movement
arm.stop()
# It is always good to clear your targets after planning with poses.
# Note: there is no equivalent function for clear_joint_value_targets()
arm.clear_pose_targets()
print("FATTO!")
time.sleep(2)
gripper.set_joint_value_target([0.036,-0.036])
gripper.go()

while not rospy.is_shutdown():
    pub.publish(mks)