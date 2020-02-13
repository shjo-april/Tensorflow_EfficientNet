# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import time
import datetime
import numpy as np

def imread(image_path):
    image = cv2.imread(image_path)
    if image is None:
        if 'jpg' in image_path:
            image = cv2.imread(image_path.replace('.jpg', '.JPG'))
        elif 'JPG' in image_path:
            image = cv2.imread(image_path.replace('.JPG', '.jpg'))
    
    return image

def get_today():
    # dt = datetime.date.today()
    # return dt.strftime("%Y-%B-%d")
    
    now = time.localtime()

    s = "%04d-%02d-%02d-%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()
    
# + : all margin
# - : object 10%
def random_crop(image, bbox):
    h, w, c = image.shape
    xmin, ymin, xmax, ymax = bbox
    
    if xmin > 0:
        xmin = xmin - np.random.randint(0, xmin)
    if ymin > 0:
        ymin = ymin - np.random.randint(0, ymin)
    if (w - xmax) > 0:
        xmax = xmax + np.random.randint(0, w - xmax)
    if (h - ymax) > 0:
        ymax = ymax + np.random.randint(0, h - ymax)
    
    return image[ymin : ymax, xmin : xmax].copy()

def one_hot(labels, classes):
    v = np.zeros([len(labels), classes], dtype = np.float32)
    
    for i, label in enumerate(labels):
        v[i, int(label)] = 1.
    return v