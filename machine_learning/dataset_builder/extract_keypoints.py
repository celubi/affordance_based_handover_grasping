import numpy as np
import cv2 as cv
import pickle
import os
from pathlib import Path

FOLDER = './machine_learning/dataset_builder/resources/annotated'
WIDTH = 640
HEIGHT = 480

for dir in os.listdir(FOLDER):
    IN_FOLDER = os.path.join(FOLDER, dir)
    if os.path.isdir(IN_FOLDER):
        for in_dir in os.listdir(IN_FOLDER):
            END_FOLDER = os.path.join(IN_FOLDER, in_dir)
            print(END_FOLDER)
            path = Path(END_FOLDER)
            for fs in path.glob('segmask.png'):
                keypoints = open(os.path.join(END_FOLDER,'keypoints.txt'), 'w')
                print(fs)
                segmask = cv.imread(str(fs), cv.IMREAD_GRAYSCALE)
                x1 = []; y1 = []; x2 = []; y2 = []; x3 = []; y3 = []
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        if segmask[y,x] == 1:
                            x1.append(x)
                            y1.append(y)
                        if segmask[y,x] == 2:
                            x2.append(x)
                            y2.append(y)
                        if segmask[y,x] == 3:
                            x3.append(x)
                            y3.append(y)
                            
                c1 = (int(sum(x1) / len(x1)), int(sum(y1) / len(y1)))
                c2 = (int(sum(x2) / len(x2)), int(sum(y2) / len(y2)))
                c3 = (int(sum(x3) / len(x3)), int(sum(y3) / len(y3)))
                keypoints.write(str(c1[0])+' '+str(c1[1])+' '+str(c2[0])+' '+str(c2[1])+' '+str(c3[0])+' '+str(c3[1]))