#!/usr/bin/env python

import rospy

from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from std_msgs.msg import Header
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

from ultralytics import YOLO
import numpy as np
import cv2

import types

import tf
import sys
import moveit_commander

from tf.transformations import quaternion_from_euler

def points_to3d(points, depth, intr, scale):
    hand_joints = np.zeros((3,3))

    id = 0
    for i, landmark in enumerate(points):
        print(landmark)
        x = np.minimum(int(landmark[0]), intr.width-1)
        y = np.minimum(int(landmark[1]), intr.height-1)

        z = depth[y,x]
        hand_joints[id,0] = z
        hand_joints[id,1] = x
        hand_joints[id,2] = y

        id = id + 1
        
    h_z = hand_joints[:,0] / scale
    h_x = (hand_joints[:,1] - intr.ppx) * h_z / intr.fx
    h_y = (hand_joints[:,2] - intr.ppy) * h_z / intr.fy

    h_cloud = np.stack([h_x, h_y, h_z], axis=-1)
    return h_cloud

def publish_point(pub, center, frame_id):

    header = Header()
    header.stamp = rospy.Time(0)
    header.frame_id = frame_id

    point = Point()
    point.x = center[0]
    point.y = center[1]
    point.z = center[2]

    point_stamped = PointStamped()
    point_stamped.header = header
    point_stamped.point = point

    pub.publish(point_stamped)

    return point_stamped

def callback(rgb, depth_aligned, info):

    camera = info.K
    intr = types.SimpleNamespace()
    intr.fx = camera[0]
    intr.fy = camera[4]
    intr.ppx = camera[2]
    intr.ppy = camera[5]
    intr.height = 480
    intr.width = 640

    rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding='rgb8')
    depth_aligned.encoding = "mono16"
    depth = bridge.imgmsg_to_cv2(depth_aligned, desired_encoding='mono16')

    color_image = np.asanyarray(rgb)
    depth_image = np.asanyarray(depth)

    depth_scale = 0.001
    clipping_distance_in_meters = 2 
    clipping_distance = clipping_distance_in_meters / depth_scale

    bg_removed = np.where((depth_image > clipping_distance), clipping_distance, depth_image)

    results = model_final(color_image)

    for result in results:
        n_result = result.numpy()

        max = len(n_result.boxes.cls)
        print(max)
        for i in range(max):
            conf = n_result.boxes.conf[i]

            if (conf > 0.5):
                box = n_result.boxes.xyxy[i]
                cls = n_result.boxes.cls[i]
                name = names[int(cls)]
                color = colors[int(cls)]
                conf = n_result.boxes.conf[i]

                color_image = cv2.rectangle(color_image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), color, 2) 
                color_image = cv2.putText(color_image, str(name)+' '+str(conf) , (10,30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA) 

                if len(n_result.keypoints) > 0:
                    points = n_result.keypoints[i].xy[0]
                    color_image = cv2.circle(color_image, (int(points[0,0]), int(points[0,1])), 3, (0,255,0), 2)
                    color_image = cv2.circle(color_image, (int(points[1,0]), int(points[1,1])), 3, (255,255,0), 2)
                    color_image = cv2.circle(color_image, (int(points[2,0]), int(points[2,1])), 3, (255,0,0), 2)
                    print("---")
                    h_cloud = points_to3d(points, bg_removed, intr, 1000)
                    good = publish_point(pub_p0, h_cloud[0], 'locobot/camera_depth_link')
                    grasp = publish_point(pub_p1, h_cloud[1], 'locobot/camera_depth_link')
                    dang = publish_point(pub_p2, h_cloud[2], 'locobot/camera_depth_link')

                    while not rospy.is_shutdown():
                        try:
                            good_trs = listener.transformPoint('locobot/base_footprint', good)
                            dang_trs = listener.transformPoint('locobot/base_footprint', dang)
                            grasp_trs = listener.transformPoint('locobot/base_footprint', grasp)
                            break
                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                            print(e)
                    
                    x = grasp_trs.point.x - good_trs.point.x
                    y = grasp_trs.point.y - good_trs.point.y
                    z = grasp_trs.point.z - good_trs.point.z

                    angleZ = np.arctan2(y, x)

                    quaternion = quaternion_from_euler(0, np.deg2rad(90), angleZ)

                    pose_point = PoseStamped()
                    pose_point.header.frame_id= "locobot/base_footprint"
                    pose_point.pose.orientation.x = quaternion[0]
                    pose_point.pose.orientation.y = quaternion[1]
                    pose_point.pose.orientation.z = quaternion[2]
                    pose_point.pose.orientation.w = quaternion[3]
                    pose_point.pose.position.x = good_trs.point.x
                    pose_point.pose.position.y = good_trs.point.y
                    pose_point.pose.position.z = good_trs.point.z

                    pub_pose.publish(pose_point)
                    pub_goal.publish(pose_point)
                    rospy.spin()
                
        detection = bridge.cv2_to_imgmsg(color_image, "rgb8")
        pub_detection.publish(detection)


rospy.init_node('test_yolo_grasp', anonymous=True)

listener = tf.TransformListener()

names = ['paintbrush', 'pen', 'razor', 'screwdriver', 'hairbrush','knife','lighter']
colors = [(0,0,255), (255,0,0), (0,255,0), (255,255,0), (0,255,255), (255,0,255), (120,120,120)]

bridge = CvBridge()
model_final = YOLO('...')

pub_detection= rospy.Publisher('/customid/detection', Image, queue_size=1)
pub_p0 = rospy.Publisher('/customid/point0', PointStamped, queue_size=1)
pub_p1 = rospy.Publisher('/customid/point1', PointStamped, queue_size=1)
pub_p2 = rospy.Publisher('/customid/point2', PointStamped, queue_size=1)
pub_pose = rospy.Publisher('/customid/pose', PoseStamped, queue_size=1)
pub_goal = rospy.Publisher('/customid/gripper_pose', PoseStamped, queue_size=1)

rgb_sub = message_filters.Subscriber('/locobot/camera/color/image_raw', Image)
depth_aligned_sub = message_filters.Subscriber('/locobot/camera/aligned_depth_to_color/image_raw', Image)
depth_aligned_info_sub = message_filters.Subscriber('/locobot/camera/aligned_depth_to_color/camera_info', CameraInfo)

tss = message_filters.TimeSynchronizer([rgb_sub,depth_aligned_sub, depth_aligned_info_sub], 100)
tss.registerCallback(callback)

rospy.spin()