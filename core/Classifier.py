# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

import core.efficientnet.efficientnet_builder as efficientnet

from utils.Utils import *

def Visualize(conv, fc_w, classes):
    vis_h = conv.shape[1]
    vis_w = conv.shape[2]
    vis_c = conv.shape[3]
    
    heatmap_conv = tf.reshape(conv, [-1, vis_w * vis_h, vis_c])
    heatmap_fc_w = tf.reshape(fc_w, [-1, vis_c, classes])
    heatmap_flat = tf.matmul(heatmap_conv, heatmap_fc_w)
    
    heatmaps = tf.reshape(heatmap_flat, [-1, vis_h, vis_w, classes])
    
    # normalize
    min_value = tf.math.reduce_min(heatmaps, axis = [0, 1, 2])
    max_value = tf.math.reduce_max(heatmaps, axis = [0, 1, 2])
    heatmaps = (heatmaps - min_value) / (max_value - min_value) * 255.

    return tf.identity(heatmaps, name = 'heatmaps')

def EfficientNet(x, is_training, option):
    model_name = 'efficientnet-{}'.format(option['name'])

    log_print('# {}'.format(model_name), option['log_txt_path'])
    log_print('- mean = {}, std = {}'.format(efficientnet.MEAN_RGB, efficientnet.STDDEV_RGB), option['log_txt_path'])

    x = (x[..., ::-1] - efficientnet.MEAN_RGB) / efficientnet.STDDEV_RGB
    _, end_points = efficientnet.build_model_base(x, model_name, is_training, getter = option['getter'])
    
    for i in range(1, 5 + 1):
        log_print('- reduction_{} : {}'.format(i, end_points['reduction_{}'.format(i)]), option['log_txt_path'])

    with tf.variable_scope('Classifier', reuse = tf.AUTO_REUSE, custom_getter = option['getter']):
        feature_maps = end_points['reduction_5']
        
        x = tf.reduce_mean(feature_maps, axis = [1, 2], name = 'GAP')
        log_print('- GAP : {}'.format(x), option['log_txt_path'])

        logits = tf.layers.dense(x, option['classes'], use_bias = False, name = 'logits')
        predictions = tf.nn.softmax(logits, name = 'outputs')
    
    return {
        'logits' : logits,
        'predictions' : predictions,
        'feature_maps' : feature_maps
    }

