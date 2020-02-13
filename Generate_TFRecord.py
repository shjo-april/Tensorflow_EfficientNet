# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import glob

import numpy as np
import tensorflow as tf

from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 1. define
root_dir = './flower_dataset/'
dataset_dir = './dataset/'

image_per_tfrecord = 500
label_dic = {name : label for label,name in enumerate(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])}

# 2. read image_paths
for name in ['train', 'test']:
    dataset = []
    src_dir = root_dir + name + '/'

    for label_name in os.listdir(src_dir):
        label = label_dic[label_name]
        image_paths = glob.glob(src_dir + label_name + '/*')

        dataset += [[image_path, label] for image_path in image_paths]

    print('# Total : {}'.format(len(dataset)))

    # 3. write tfrecords
    iteration = len(dataset) // image_per_tfrecord

    for index in range(iteration):
        tfrecord_path = dataset_dir + '{}_{}.tfrecord'.format(name, index + 1)
        print(tfrecord_path, image_per_tfrecord, len(dataset[index * image_per_tfrecord : (index + 1) * image_per_tfrecord]))

        with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
            for (image_path, label) in dataset[index * image_per_tfrecord : (index + 1) * image_per_tfrecord]:

                # with opencv
                image = cv2.imread(image_path)
                if image is None:
                    print(image_path)
                    continue

                h, w, c = image.shape
                image_raw = image.tostring()
                
                # with Pillow
                # image = np.array(Image.open(image_path))
                # h, w, c = image.shape
                # image_raw = image.tostring()
                
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_raw),
                    'label'    : _int64_feature(label),

                    'height'   : _int64_feature(h),
                    'width'    : _int64_feature(w),
                    'channel'  : _int64_feature(c),
                }))
                tfrecord_writer.write(example.SerializeToString())

    if len(dataset) % image_per_tfrecord > 0:
        tfrecord_path = dataset_dir + '{}_{}.tfrecord'.format(name, index + 2)
        print(tfrecord_path, len(dataset) % image_per_tfrecord)

        with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
            for (image_path, label) in dataset[-(len(dataset) % image_per_tfrecord) :]:
                image = cv2.imread(image_path)
                if image is None:
                    print(image_path)
                    continue

                h, w, c = image.shape
                image_raw = image.tostring()
                
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_raw),
                    'label'    : _int64_feature(label),

                    'width'    : _int64_feature(w),
                    'height'   : _int64_feature(h),
                    'channel'  : _int64_feature(c),
                }))
                tfrecord_writer.write(example.SerializeToString())
