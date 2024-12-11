# How to build the dataset?
The dataset is built from simple rgb pictures of objects to be grasped and manipulated by the robot. To make the annotation process easier an object mask is extracted from the picture. In this mask white pixels represent the object, while balck pixels represent the background. The object_mask was obatained using: https://github.com/renatoviolin/bg-remove-augment.

## Annotation
The **annotate_mask.py** script enables to annotate three object regions:
- **Danger**, hazardous object parts, like baldes and sharp points.
- **Grasp**, part suitable for robot grasping.
- **Handle**, part suitable for human grasping.

The script takes as input images in the **resources/sample_yolo_data** folder and outputs data in the **resources/annotated** folder. For each original picture the script outputs a copy of the rgb image and object_mask and an additional *segmask.png* file. The *segmask.png* classifies each pixel in the rgb image as one of the object regions + background, by givining the same numerical value to the pixel belonging to the same region.

## Keypoints extraction
The **extract_keypoints.py** script computes the correspondig keypoint for each object region. For each picture it produces the **keypoints.txt** file containing the coordintes of the keypoints expressed in pixels.

## Formatting
The **format_for_yolo.py** script reorganizes the dataset to be easily used to train YOLOv8. To mitigate overfitting several copies of the original image are crated performing background swap and random data augmentation.