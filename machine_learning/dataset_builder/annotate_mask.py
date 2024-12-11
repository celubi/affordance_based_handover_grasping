import numpy as np
import cv2 as cv
import pickle
import os
from pathlib import Path

BASE_INPUT = "./machine_learning/dataset_builder/resources/sample_yolo_data"
SELECTED = "/paintbrush"
INPUT_FOLDER = BASE_INPUT + SELECTED
OUTPUT_FOLDER = "machine_learning/dataset_builder/resources/annotated"
WIDTH = 640
HEIGHT = 480

def reset():
    global ix,iy,sx,sy, index, array1, array2, fase, stop, mask, good
    ix,iy,sx,sy = -1,-1,-1,-1
    array1 = np.zeros((4,2), dtype=int)
    array2 = np.zeros((4,2), dtype=int)
    index = 0
    fase = 0
    good = False

# mouse callback function
def draw_lines(event, x, y, flags, param):
    global ix,iy,sx,sy, index, array1, array2, fase, mask, good
    global name, out_imgs, out_sm, out_masks, out_xmls
    
    if event == cv.EVENT_LBUTTONDOWN and fase == 0:
        
        array1[index] = np.array([x,y])

        # draw circle of 2px
        cv.circle(img, (x, y), 3, (0, 0, 255), -1)

        if ix != -1: # if ix and iy are not first points, then draw a line
            cv.line(img, (ix, iy), (x, y), (0, 0, 255), 2, cv.LINE_AA)
        else: # if ix and iy are first points, store as starting points
            sx, sy = x, y
        ix,iy = x, y

        index += 1
        print(index)
        if index == 4:
            index = 0
            ix, iy = -1, -1 # reset ix and iy
            fase = 1
            print(array1)
            pts = array1.reshape((-1,1,2))
            cv.fillPoly(img,[pts],(0,0,255))


    elif event == cv.EVENT_LBUTTONDOWN and fase == 1:
        
        array2[index] = np.array([x,y])

        # draw circle of 2px
        cv.circle(img, (x, y), 3, (255, 0, 0), -1)

        if ix != -1: # if ix and iy are not first points, then draw a line
            cv.line(img, (ix, iy), (x, y), (255, 0, 0), 2, cv.LINE_AA)
        else: # if ix and iy are first points, store as starting points
            sx, sy = x, y
        ix,iy = x, y

        index += 1
        print(index)
        if index == 4:
            index = 0
            fase = 2
            print(array2)
            pts = array2.reshape((-1,1,2))
            cv.fillPoly(img,[pts],(255,0,0))

    elif event == cv.EVENT_LBUTTONDOWN and fase == 2:
        boh = mask.copy()
        cv.fillPoly(boh,[array1.reshape((-1,1,2))],(0,0,0))
        cv.fillPoly(boh,[array2.reshape((-1,1,2))],(0,0,0))
        cv.bitwise_and(img,mask, dst=img)
        cv.add(img,boh,dst=img)
        fase = 3

    elif event == cv.EVENT_LBUTTONDOWN and fase == 3:
        good = True
        seg = np.zeros((HEIGHT,WIDTH))
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if (img[i,j] == np.array([0,0,255])).all():
                    seg[i,j] = 3
                elif (img[i,j] == np.array([255,0,0])).all():
                    seg[i,j] = 2
                elif (img[i,j] == np.array([255,255,255])).all():
                    seg[i,j] = 1

        save_path = os.path.join(out, str(name))
        os.mkdir(save_path)
        cv.imwrite(os.path.join(save_path,'segmask.png'), seg)
        cv.imwrite(os.path.join(save_path,'mask.png'), cv.cvtColor(mask,cv.COLOR_BGR2GRAY))
        cv.imwrite(os.path.join(save_path,'img.jpg'), img_bk)

name = 0
out = OUTPUT_FOLDER+SELECTED
print(out)
os.mkdir(out)   

for dir in os.listdir(INPUT_FOLDER):
    if os.path.isdir(os.path.join(INPUT_FOLDER, dir)):
        print(dir)
        path = Path(os.path.join(INPUT_FOLDER, dir))
        for fm in path.glob('mask.png'):
            print(fm)
            mask_raw = cv.imread(str(fm))
            mask_gray = cv.cvtColor(mask_raw,cv.COLOR_BGR2GRAY)
            ret, mask_mono = cv.threshold(mask_gray, 1, 255, cv.THRESH_BINARY)
            mask = cv.merge([mask_mono,mask_mono,mask_mono])
        for fi in path.glob('rgb.jpg'):
            print(fi)
            img = cv.imread(str(fi))
        
        reset()

        cv.namedWindow('image') 
        cv.setMouseCallback('image',draw_lines)

        img_bk = img.copy()
        while(1):
            cv.imshow('image',img)
            if good:
                break
            elif cv.waitKey(20) & 0xFF == 27:
                img = img_bk.copy()
                reset()

        cv.destroyAllWindows()

        name+=1