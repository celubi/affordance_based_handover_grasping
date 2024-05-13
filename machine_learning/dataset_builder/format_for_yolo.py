from PIL import Image, ImageOps, ImageEnhance
import cv2
import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import random

INPUT_FOLDER = "./machine_learning/dataset_builder/resources/annotated"
OUTPUT_FOLDER = "./machine_learning/dataset_builder/resources/formatted"

WIDTH = 640
HEIGHT = 480

yclasses = {
    "paintbrush": 0,
    "pen": 1,
    "razor": 2,
    "screwdriver": 3,
    "hairbrush": 4,
    "knife": 5,
    "lighter": 6
}

def get_boundingbox(fm, resize):
    img = Image.open(str(fm))
    if resize:
        img = img.resize([WIDTH,HEIGHT],resample=Image.NEAREST)
    gray = np.array(img.convert('L'))
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        xmin = x
        ymin = y
        xmax = x+w
        ymax = y+h

    return xmin, ymin, xmax, ymax


def manage_img(fi, name, path_img):
    img = Image.fromarray(fi)
    img = img.resize([WIDTH,HEIGHT],resample=Image.NEAREST)
    img.save(os.path.join(path_img,str(name)+'.jpg'))

def manage_segmask(fs, name, path_segmask, path_sm):
    seg = Image.open(str(fs))
    seg = seg.resize([WIDTH,HEIGHT],resample=Image.NEAREST)
    seg.save(os.path.join(path_segmask,str(name)+'.png'))

    file = open(os.path.join(path_sm,str(name)+'_1_segmask.sm'), 'wb')
    pickle.dump(np.array(seg).astype(np.uint8), file)

def manage_yolo(fk, fm, name, path_yolo, dir):
    annotation_file = open(os.path.join(path_yolo,str(name)+'.txt'), 'w')

    yclass_name = str(dir).replace("B","").replace("C","").replace("D","").replace("I","").replace("F","")
    yclass = yclasses[str(yclass_name)]

    if yclass == 1:
        print(name)

    x, y, xmax, ymax = get_boundingbox(fm, True)
    w = xmax - x
    h = ymax - y
    xc = w/2 + x
    yc = h/2 + y

    keys = open(fk, 'r')
    line = keys.readline()
    ks = line.split(' ')

    annotation_file.write(str(yclass)+' '+str(xc/WIDTH)+' '+str(yc/HEIGHT)+' '+str(w/WIDTH)+' '+str(h/HEIGHT)+' '+str(int(ks[0])/640)+' '+str(int(ks[1])/480)+' '+str(int(ks[2])/640)+' '+str(int(ks[3])/480)+' '+str(int(ks[4])/640)+' '+str(int(ks[5])/480))

def augment(img):

    if random.randint(0,1) == 0:
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.6, 1.4) 
        img = enhancer.enhance(factor)
    
    if random.randint(0,1) == 0:
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.6, 1.4) 
        img = enhancer.enhance(factor)

    if random.randint(0,1) == 0:
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.6, 1.4) 
        img = enhancer.enhance(factor)

    if random.randint(0,1) == 0:
        enhancer = ImageEnhance.Sharpness(img)
        factor = random.uniform(0.6, 1.4) 
        img = enhancer.enhance(factor)
    
    aug_img = np.array(img)

    if random.randint(0,1) == 0:
        mean = 0
        stddev = random.randint(5,20)
        noise = np.zeros(aug_img.shape, np.uint8)
        cv2.randn(noise, mean, stddev)
        aug_img = cv2.add(aug_img, noise)
    
    return aug_img

path_segmask = os.path.join(OUTPUT_FOLDER,"segmask")
path_sm = os.path.join(OUTPUT_FOLDER,"sm")
path_img = os.path.join(OUTPUT_FOLDER,"img")
path_yolo = os.path.join(OUTPUT_FOLDER,"yolo")

os.mkdir(path_segmask)
os.mkdir(path_sm)
os.mkdir(path_img)
os.mkdir(path_yolo)

bk_list = [
    np.array(Image.open('./machine_learning/dataset_builder/resources/sample_backgrounds/bk0.png')),
    np.array(Image.open('./machine_learning/dataset_builder/resources/sample_backgrounds/bk1.png')),
]

train_file = open("/Users/celu/Documents/robe_tesi/data_lab/sets/yolo.txt", 'w')

name = 0

for dir in os.listdir(INPUT_FOLDER):
    path = os.path.join(INPUT_FOLDER,dir)
    if os.path.isdir(path):
        for in_dir in os.listdir(path):
            if os.path.isdir(os.path.join(path,in_dir)):
                in_path = Path(os.path.join(path,in_dir))

                print(in_path)

                fi = next(in_path.glob("img.jpg")) 
                fm = next(in_path.glob("mask.png")) 
                fs = next(in_path.glob("segmask.png"))
                fk = next(in_path.glob("keypoints.txt"))

                mask = np.array(Image.open(str(fm)))
                kernel = np.ones((4, 4), np.uint8) 
                mask_in = cv2.erode(mask, kernel)  
                mask_in = cv2.GaussianBlur(mask_in, (0,0), sigmaX=1, sigmaY=1, borderType = cv2.BORDER_DEFAULT)
                ret, mask_in = cv2.threshold(mask_in, 100, 255, cv2.THRESH_BINARY)
                mask = cv2.GaussianBlur(mask, (0,0), sigmaX=4, sigmaY=4, borderType = cv2.BORDER_DEFAULT)
                mask = cv2.add(mask, mask_in)

                
                mask_not = cv2.bitwise_not(mask)
                image = np.array(Image.open(str(fi)))
                b_channel, g_channel, r_channel = cv2.split(image)
                img_a = cv2.merge((b_channel, g_channel, r_channel, mask))

                manage_segmask(fs,name,path_segmask,path_sm)
                manage_img(image, name, path_img)
                manage_yolo(fk, fm, name, path_yolo, dir)

                name+=1

                image = augment(Image.fromarray(image))
                manage_segmask(fs,name,path_segmask,path_sm)
                manage_img(image, name, path_img)
                manage_yolo(fk, fm, name, path_yolo, dir)

                name+=1

                for bk in bk_list:

                    choice = random.randint(0,2)
                    print(choice)

                    if choice < 3:

                        #masked_bk = cv2.bitwise_and(bk, cv2.merge([mask_not,mask_not,mask_not]))
                        #aug_img = cv2.add(masked_bk, masked_img)

                        bk_a = cv2.cvtColor(bk, cv2.COLOR_RGB2RGBA)
                        
                        obj = Image.fromarray(img_a)
                        bkg = Image.fromarray(bk_a)
                        
                        #bkg.show()
                        bkg.alpha_composite(obj)
                        #bkg.save('/Users/celu/Documents/robe_tesi/data_lab/bubu2.png')

                        if choice == 1:
                            aug_img = np.asarray(augment(bkg.convert('RGB')))
                        else:
                            aug_img = np.asarray(bkg.convert('RGB'))

                        #print(fm)
                        manage_segmask(fs,name,path_segmask,path_sm)
                        manage_img(aug_img, name, path_img)
                        manage_yolo(fk, fm, name, path_yolo, dir)

                        name+=1