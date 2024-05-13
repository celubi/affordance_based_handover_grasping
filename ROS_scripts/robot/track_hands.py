#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import types

def get_joints(hand_landmarks, depth, intr, scale):

    hand_joints = np.zeros((6,3))

    id = 0
    for i, landmark in enumerate(hand_landmarks):
        if i == 0 or i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            x = np.minimum(round(landmark.x*intr.width), intr.width-1)
            y = np.minimum(round(landmark.y*intr.height), intr.height-1)

            z = depth[y,x]
            hand_joints[id,0] = z
            hand_joints[id,1] = x
            hand_joints[id,2] = y

            id = id + 1
        
    h_z = hand_joints[:,0] / scale
    h_x = (hand_joints[:,1] - intr.ppx) * h_z / intr.fx
    h_y = (hand_joints[:,2] - intr.ppy) * h_z / intr.fy

    h_cloud = np.stack([h_x, h_y, h_z], axis=-1)
    center = np.mean(h_cloud, axis=0)
    
    return center

def manage_hand(best_center, old_center, new_center, count):

    if np.linalg.norm(old_center - new_center) < 0.04:
        count = count +1
    else:
        count = 0

    if count == 10:
        if np.linalg.norm(best_center - new_center) > 0.1:
            best_center = new_center

        count = 0

    return best_center, count
    

def publish_hand(pub, center, frame_id):

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    point = Point()
    point.x = center[0]
    point.y = center[1]
    point.z = center[2]

    point_stamped = PointStamped()
    point_stamped.header = header
    point_stamped.point = point

    pub.publish(point_stamped)

def publish_pose(pub, center, frame_id):
    
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    pose_goal = Pose()
    pose_goal.orientation.w = 1.0
    pose_goal.orientation.y = -1.0
    pose_goal.position.x = center[0]
    pose_goal.position.y = center[1]
    pose_goal.position.z = center[2]

    pose_stamped = PoseStamped()
    pose_stamped.header = header
    pose_stamped.pose = pose_goal

    pub.publish(pose_stamped)

def callback(rgb, depth_aligned, info):
    global best_center, old_center, count, available

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
    #bg_removed = depth_image

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
    hand_landmarker_result = recognizer.recognize_for_video(mp_image, round(time.time() * 1000))
    #hand_landmarker_result = landmarker.detect_for_video(mp_image, round(time.time() * 1000))

    print(hand_landmarker_result.gestures)

    if len(hand_landmarker_result.hand_landmarks) > 0:
        if hand_landmarker_result.gestures[0][0].category_name == "Open_Palm":
            center = get_joints(hand_landmarker_result.hand_landmarks[0], bg_removed, intr, 1/depth_scale)
            #print(center)
            best_center, count = manage_hand(best_center,old_center,center,count)
            print(best_center)
            publish_hand(pub_sphere, center, 'locobot/camera_depth_link')
            publish_pose(pub_pose, center, 'locobot/camera_depth_link')
            available = True
            old_center = center

rospy.init_node('track_hands', anonymous=True)

topic_name1='/locobot/customid/hand'
pub_sphere= rospy.Publisher(topic_name1, PointStamped, queue_size=1)

topic_name2='/locobot/customid/handpose'
pub_pose = rospy.Publisher(topic_name2, PoseStamped, queue_size=1)

old_center = np.array([0, 0, 0])
best_center = np.array([0, 0, 0])
count = 0
available = False

bridge = CvBridge()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

# Create a gesture recognizer instance with the video mode:
options_gesture = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/home/locobot/celu/handover_ws/src/handover/scripts/gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO)
recognizer = GestureRecognizer.create_from_options(options_gesture)

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/home/locobot/celu/handover_ws/src/handover/scripts/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)
landmarker = HandLandmarker.create_from_options(options)

rgb_sub = message_filters.Subscriber('/locobot/camera/color/image_raw', Image)
depth_aligned_sub = message_filters.Subscriber('/locobot/camera/aligned_depth_to_color/image_raw', Image)
depth_aligned_info_sub = message_filters.Subscriber('/locobot/camera/aligned_depth_to_color/camera_info', CameraInfo)

tss = message_filters.TimeSynchronizer([rgb_sub,depth_aligned_sub, depth_aligned_info_sub], 10)
tss.registerCallback(callback)

rospy.spin()