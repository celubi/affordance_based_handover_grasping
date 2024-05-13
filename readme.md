# Overview
This repository contains some of the code developed for the master's thesis *Smart mobile manipulation for human-robot assistive applications, Politecnico di Torino, 2024*. 

## Introduction
In applications where robots assist humans, they need to be capable of handing objects to people. The goal is typically not just to transfer an object, but to enable the person to utilize it for a specific task. This problem is referred to as task-oriented robot-to-human handover.

The thesis introduced a framework designed to address each stage of a handover process, with a primary focus on ensuring safety. This framework was tested on a real low-cost mobile manipulator. The robot is equipped with a 6DOF arm, a parallel gripper, a mobile base and a RGB-D camera. The code was written using ROS1 Noetic.

## Contribution
Affordance estimation is used to extract information about objects' dangerous parts and handles. This information is used to plan grasp and deliver the object to the human with the correct orientation. 
The handover location is determined based on the current position of the human hand, and gestures are utilized to initiate the handover process.

Main components of the overall architecture:

 - Novel affordance representation based on keypoints
 - YOLOv8 Pose used to detect objects and affordance keypoints
 - MediaPipe used for hand tracking and gesture recognition
 - Safety policies for the manipulator path planning

## Repository's content
The repository contains ROS Python nodes that operate on both the robot and a remote machine. These nodes facilitate the robot in executing grasps and handovers. Additionally, the repository includes the script utilized for constructing and annotating the dataset employed to train YOLO.