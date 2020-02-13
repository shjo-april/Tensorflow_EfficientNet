# Copyright (C) 2020 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import time
import glob
import json
import random
import argparse

import numpy as np
import tensorflow as tf

from queue import Queue

from core.Classifier import *
from core.efficientnet.utils import *

from utils.Utils import *
from utils.Teacher import *
from utils.ArgumentParser import *
from utils.Tensorflow_Utils import *

args = ArgumentParser().parse_args()
args['warmup_iteration'] = int(args['max_iteration'] * 0.05) # warmup iteration = 5%

model_name = '{}-{}-EfficientNet-{}'.format(args['experimenter'], get_today(), args['option'])

if not args['multi_scale']:
    width_coeff, depth_coeff, resolution, dropout_rate = efficientnet.efficientnet_params('efficientnet-{}'.format(args['option']))
    args['max_image_size'] = resolution

num_gpu = len(args['use_gpu'].split(','))
os.environ["CUDA_VISIBLE_DEVICES"] = args['use_gpu']

args['batch_size'] = args['batch_size_per_gpu'] * num_gpu
args['init_learning_rate'] = 0.016 * args['batch_size'] / 256
args['alpha_learning_rate'] = 0.002 * args['batch_size'] / 256

model_dir = './experiments/model/{}/'.format(model_name)
tensorboard_dir = './experiments/tensorboard/{}/'.format(model_name)

ckpt_format = model_dir + '{}.ckpt'
log_txt_path = model_dir + 'log.txt'

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

if os.path.isfile(log_txt_path):
    open(log_txt_path, 'w').close()

#######################################################################################
# 1. Dataset
#######################################################################################
log_print('# {}'.format(model_name), log_txt_path)
log_print('{}'.format(json.dumps(args, indent='\t')), log_txt_path)

train_dataset = TFRecord_Reader('./dataset/train_*.tfrecord', batch_size = args['batch_size'], is_training = True, use_prefetch = False)
test_dataset = TFRecord_Reader('./dataset/test_*.tfrecord', batch_size = args['batch_size'], is_training = False, use_prefetch = False)

#######################################################################################
# 2. Model
#######################################################################################
shape = [args['max_image_size'], args['max_image_size'], 3]

image_var = tf.placeholder(tf.float32, [None] + shape, name = 'images')
label_var = tf.placeholder(tf.float32, [None, 5], name = 'labels')
is_training = tf.placeholder(tf.bool)

option = {
    'name' : args['option'],
    'classes' : 5,
    'log_txt_path' : log_txt_path,
    'getter' : None,
}

if num_gpu == 1:
    output_dic = EfficientNet(image_var, is_training, option)
    logits_op = output_dic['logits']
else:
    logits_list = []

    image_vars = tf.split(image_var, num_gpu)
    for gpu_id in range(num_gpu):
        # with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = gpu_id)):
        with tf.device(tf.train.replica_device_setter(worker_device = '/gpu:%d'%gpu_id, ps_device = '/cpu:0', ps_tasks = 1)):
            with tf.variable_scope(tf.get_variable_scope(), reuse = gpu_id != 0):
                output_dic = EfficientNet(image_vars[gpu_id], is_training, option)
                logits_list.append(output_dic['logits'])
    
    logits_op = tf.concat(logits_list, axis = 0)

#######################################################################################
# 2.1. Loss
#######################################################################################
log_print('[i] softmax_cross_entropy_with_logits', log_txt_path)

class_loss_op = tf.nn.softmax_cross_entropy_with_logits(logits = logits_op, labels = label_var)
class_loss_op = tf.reduce_mean(class_loss_op)

train_vars = tf.trainable_variables()
l2_vars = [var for var in train_vars if 'kernel' in var.name or 'weights' in var.name]
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in l2_vars]) * args['weight_decay']

loss_op = class_loss_op + l2_reg_loss_op

#######################################################################################
# 2.2. Accuracy
#######################################################################################
log_print('[i] generate accuracy operation', log_txt_path)

if args['ema_decay'] != -1:
    ema = tf.train.ExponentialMovingAverage(decay = args['ema_decay'])
    ema_op = ema.apply(get_ema_vars())

    # option['getter'] = get_getter(ema)
    # predictions_op = EfficientNet(test_image_var, False, option)['predictions']

correct_op = tf.equal(tf.argmax(logits_op, axis = -1), tf.argmax(label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

#######################################################################################
# 3. optimizer
#######################################################################################
global_step = tf.placeholder(dtype = tf.int32)

warmup_lr_op = tf.to_float(global_step) / tf.to_float(args['warmup_iteration']) * args['init_learning_rate']
decay_lr_op = tf.train.cosine_decay(
    args['init_learning_rate'],
    global_step = global_step - args['warmup_iteration'],
    decay_steps = args['max_iteration'] - args['warmup_iteration'],
    alpha = args['alpha_learning_rate']
)

learning_rate = tf.where(global_step < args['warmup_iteration'], warmup_lr_op, decay_lr_op)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True).minimize(loss_op, colocate_gradients_with_ops = True)

    if args['ema_decay'] != -1:
        train_op = tf.group(train_op, ema_op)

#######################################################################################
# 4. tensorboard
#######################################################################################
train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/Clasification_Loss' : class_loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op, 
    'Accuracy/Train_Accuracy' : accuracy_op,
    'Learning_rate' : learning_rate,
}
train_summary_op = tf.summary.merge([tf.summary.scalar(name, train_summary_dic[name]) for name in train_summary_dic.keys()])

test_summary_dic = {
    'Accuracy/Test_Accuracy' : tf.placeholder(tf.float32),
}
test_summary_op = tf.summary.merge([tf.summary.scalar(name, test_summary_dic[name]) for name in test_summary_dic.keys()])

train_writer = tf.summary.FileWriter(tensorboard_dir)

#######################################################################################
# 5. Session, Saver
#######################################################################################
sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(train_dataset.initializer_op)

saver = tf.train.Saver(max_to_keep = 2)

# pretrained model
pretrained_model_name = 'efficientnet-{}'.format(args['option'])
pretrained_model_path = './pretrained_model/{}/model.ckpt'.format(pretrained_model_name)

imagenet_saver = tf.train.Saver(var_list = [var for var in train_vars if pretrained_model_name in var.name])
imagenet_saver.restore(sess, pretrained_model_path)

log_print('[i] restore pretrained model ({}) -> {}'.format(pretrained_model_name, pretrained_model_path), log_txt_path)

#######################################################################################
# 6. Train
#######################################################################################
loss_list = []
class_loss_list = []
l2_reg_loss_list = []
accuracy_list = []
train_time = time.time()

best_test_accuracy = 0
train_ops = [train_op, loss_op, class_loss_op, l2_reg_loss_op, accuracy_op, train_summary_op]

for iter in range(1, args['max_iteration'] + 1):
    batch_image_data, batch_label_data = sess.run([train_dataset.image_op, train_dataset.label_op])
    batch_label_data = one_hot(batch_label_data, 5)
    
    _feed_dict = {
        image_var : batch_image_data, 
        label_var : batch_label_data, 
        is_training : True,
        global_step : iter,
    }
    _, loss, class_loss, l2_reg_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
    train_writer.add_summary(summary, iter)
    
    loss_list.append(loss)
    class_loss_list.append(class_loss)
    l2_reg_loss_list.append(l2_reg_loss)
    accuracy_list.append(accuracy)
    
    if iter % args['log_iteration'] == 0:
        loss = np.mean(loss_list)
        class_loss = np.mean(class_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        accuracy = np.mean(accuracy_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter = {}, loss = {:.4f}, class_loss = {:.4f}, l2_reg_loss = {:.4f}, accuracy = {:.2f}, train_time = {}sec'.format(iter, loss, class_loss, l2_reg_loss, accuracy, train_time), log_txt_path)
        
        loss_list = []
        class_loss_list = []
        l2_reg_loss_list = []
        accuracy_list = []
        train_time = time.time()

    #######################################################################################
    # 8. Test
    #######################################################################################
    if iter % args['val_iteration'] == 0:
        test_time = time.time()
        test_accuracy_list = []
        
        sess.run(test_dataset.initializer_op)

        iteration = 0
        while True:
            try:
                batch_image_data, batch_label_data = sess.run([test_dataset.image_op, test_dataset.label_op])
                batch_label_data = one_hot(batch_label_data, 5)

                iteration += 1
                sys.stdout.write('\r# Test = [{}]'.format(iteration))
                sys.stdout.flush()
            except tf.errors.OutOfRangeError:
                print()
                break

            _feed_dict = {
                image_var : batch_image_data,
                label_var : batch_label_data,
                is_training : False
            }
            accuracy = sess.run(accuracy_op, feed_dict = _feed_dict)
            test_accuracy_list.append(accuracy)
            
        test_accuracy = np.mean(test_accuracy_list)
        
        summary = sess.run(test_summary_op, feed_dict = {
            test_summary_dic['Accuracy/Test_Accuracy'] : test_accuracy,
        })
        train_writer.add_summary(summary, iter)
        
        if best_test_accuracy <= test_accuracy:
            best_test_accuracy = test_accuracy
            saver.save(sess, ckpt_format.format(iter))            
        
        train_time = time.time()
        test_time = int(time.time() - test_time)

        log_print('[i] iter = {}, test_accuracy = {:.2f}, best_test_accuracy = {:.2f}, test_time = {}sec'.format(iter, test_accuracy, best_test_accuracy, test_time), log_txt_path)

saver.save(sess, ckpt_format.format('end'))

