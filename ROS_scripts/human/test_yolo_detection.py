#!/usr/bin/env python

import rospy

from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

from ultralytics import YOLO
import numpy as np
import cv2

def callback(rgb, depth_aligned, info):

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
                
        detection = bridge.cv2_to_imgmsg(color_image, "rgb8")
        pub_detection.publish(detection)


rospy.init_node('test_yolo_detection', anonymous=True)

names = ['paintbrush', 'pen', 'razor', 'screwdriver', 'hairbrush','knife','lighter']
colors = [(0,0,255), (255,0,0), (0,255,0), (255,255,0), (0,255,255), (255,0,255), (120,120,120)]

bridge = CvBridge()
model_final = YOLO('...')

pub_detection= rospy.Publisher('/customid/detection', Image, queue_size=1)

rgb_sub = message_filters.Subscriber('/locobot/camera/color/image_raw', Image)
depth_aligned_sub = message_filters.Subscriber('/locobot/camera/aligned_depth_to_color/image_raw', Image)
depth_aligned_info_sub = message_filters.Subscriber('/locobot/camera/aligned_depth_to_color/camera_info', CameraInfo)

tss = message_filters.TimeSynchronizer([rgb_sub,depth_aligned_sub, depth_aligned_info_sub], 10)
tss.registerCallback(callback)

rospy.spin()