import numpy as np
import os
import cv2

file = open('./machine_learning/dataset_builder/resources/formatted/yolo/0.txt','r')
img = cv2.imread('./machine_learning/dataset_builder/resources/formatted/img/0.jpg', cv2.IMREAD_COLOR)

line = file.readline()
info = line.split(' ')

width = img.shape[1]
height = img.shape[0]

yclass = int(float(info[0]))
x = int(float(info[1])*width)
y = int(float(info[2])*height)
w = int(float(info[3])*width)
h = int(float(info[4])*height)
c1 = (int(float(info[5])*width), int(float(info[6])*height))
c2 = (int(float(info[7])*width), int(float(info[8])*height))
c3 = (int(float(info[9])*width), int(float(info[10])*height))

img = cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color=(255,0,0), thickness=3)

#img = cv2.circle(img, (x,y), radius=0, color=(0, 0, 255), thickness=10)
img = cv2.circle(img, c1, radius=0, color=(0, 255, 0), thickness=15)
img = cv2.circle(img, c2, radius=0, color=(0, 255, 255), thickness=15)
img = cv2.circle(img, c3, radius=0, color=(0, 0, 255), thickness=15)

cv2.imshow('image',img)
cv2.waitKey(0)