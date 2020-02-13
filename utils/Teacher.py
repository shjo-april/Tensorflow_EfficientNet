# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import time

import numpy as np
import tensorflow as tf

from utils.Timer import *

from core.weakaugment import *
from core.randaugment import *

class TFRecord_Reader:
    def __init__(self, tfrecord_format, batch_size, image_size = [224, 224], is_training = False, use_prefetch = False):
        self.image_size = image_size
        
        dataset = tf.data.Dataset.list_files(tfrecord_format, shuffle = is_training)
        if is_training: 
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, buffer_size = 16 * 1024 * 1024), 
            cycle_length = 16, 
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.shuffle(1024)

        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                self.parser,
                batch_size = batch_size,
                num_parallel_calls = 2,
                drop_remainder = is_training,
            )
        )

        if use_prefetch:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.iterator = dataset.make_initializable_iterator()
        self.image_op, self.label_op = self.iterator.get_next()
        self.initializer_op = self.iterator.initializer

    def parser(self, record):
        parsed = tf.parse_single_example(
            record, 
            features = {
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label'    : tf.FixedLenFeature([], tf.int64),

                'height'   : tf.io.FixedLenFeature([], tf.int64),
                'width'    : tf.io.FixedLenFeature([], tf.int64),
                'channel'  : tf.io.FixedLenFeature([], tf.int64),
        })
        
        height = tf.cast(parsed['height'], tf.int64)
        width = tf.cast(parsed['width'], tf.int64)
        channel = tf.cast(parsed['channel'], tf.int64)

        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.reshape(image, [height, width, channel])
        image = tf.image.resize(image, self.image_size)

        image = tf.cast(image, tf.float32)
        label = tf.cast(parsed['label'], tf.float32)
        
        return image, label
