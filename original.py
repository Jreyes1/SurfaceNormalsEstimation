#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path as osp
import os

from tensorflow.contrib import layers
from skimage.io import imread, imsave

DATASET_PATH = '/home/jorge/Documents/cos429/cos429challenge'
BATCH_SIZE = 8
SUMMARIES_PATH = '/home/jorge/Documents/cos429/cos429challenge/summaries'
MAX_ITERATION = 1000

DUMP_FOLDER = '/home/jorge/Documents/cos429/cos429challenge/prediction'

if not osp.exists(DUMP_FOLDER):
    os.makedirs(DUMP_FOLDER)


def build_model():


    color = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
    mask = tf.placeholder(dtype=tf.bool, shape=[None, 128, 128])
    target = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])

    # conv1a = layers.conv2d(color, 128, [11, 11], 1, 'same', activation_fn=tf.nn.relu)
    # conv1b = layers.conv2d(conv1a, 128, [11, 11], 1, 'same', activation_fn=tf.nn.relu)
    # print("Conv1 Shape")
    # print(conv1b.get_shape())
    # max1 = tf.layers.max_pooling2d(inputs = conv1b, pool_size=[2,2], strides=2)
    # print("Max1 Shape")
    # print(max1.get_shape())

    # conv2a = layers.conv2d(max1, 128, [7, 7], 1, 'same', activation_fn=tf.nn.relu)
    # conv2b = layers.conv2d(conv2a, 128, [7, 7], 1, 'same', activation_fn=tf.nn.relu)
    # print ("Conv2 shape")
    # print(conv2b.get_shape())
    # max2 = tf.layers.max_pooling2d(inputs = conv2b, pool_size=[2,2], strides=2)
    # print("Max2 Shape")
    # print(max2.get_shape())

    # conv3a = layers.conv2d(max2, 128, [5, 5], 1, 'same', activation_fn=tf.nn.relu)
    # conv3b = layers.conv2d(conv3a, 128, [5, 5], 1, 'same', activation_fn=tf.nn.relu)
    # print ("Conv3 shape")
    # print(conv3b.get_shape())
    # max3 = tf.layers.max_pooling2d(inputs = conv3b, pool_size=[2,2], strides=2)
    # print("Max3 Shape")
    # print(max3.get_shape())
  
    # conv4a = layers.conv2d(max3, 128, [3, 3], 1, 'same', activation_fn=tf.nn.relu)
    # conv4b = layers.conv2d(conv4a, 128, [3, 3], 1, 'same', activation_fn=tf.nn.relu)
    # print("Conv4 shape")
    # print(conv4b.get_shape())
    # max4 = tf.layers.max_pooling2d(inputs = conv4b, pool_size=[2,2], strides=2)
    # print("Max 4 shape")
    # print(max4.get_shape())


    # conv5a = layers.conv2d(max4, 128, [1, 1], 1, 'same', activation_fn=tf.nn.relu)
    # conv5b = layers.conv2d(conv4a, 128, [1, 1], 1, 'same', activation_fn=tf.nn.relu)
    # print("Conv5 shape")
    # print(conv5b.get_shape())
    # max5 = tf.layers.max_pooling2d(inputs = conv4b, pool_size=[2,2], strides=2)
    # print("Max 5 shape")
    # print(max5.get_shape())


    # # bottle neck here
    # conv6 = layers.conv2d(max5, 128, [1, 1], 1, 'same', activation_fn=tf.nn.relu)    
    # print("Conv6 shape (Bottle Neck)")
    # print(conv6.get_shape())
    

    # deconv7 = tf.layers.conv2d_transpose(conv6, 128, [1,1], 2, 'same',  activation =tf.nn.relu)
    # print ("Deconv7 shape")
    # print(deconv7.get_shape())
    # conv7a = layers.conv2d(tf.add(deconv7, conv4b), 128, [3, 3], 1, 'same', activation_fn=tf.nn.relu)   
    # #conv7a = layers.conv2d(deconv7, 128, [3, 3], 1, 'same', activation_fn=tf.nn.relu)   
    # conv7b = layers.conv2d(conv7a, 128, [3, 3], 1, 'same', activation_fn=tf.nn.relu)   
    # print("Conv 7 shape")
    # print(conv7b.get_shape())
    

    # deconv8 = tf.layers.conv2d_transpose(conv7b, 128, [3,3], 2, 'same',  activation=tf.nn.relu)
    # print ("Deconv8 shape")
    # print(deconv8.get_shape())
    # conv8a = layers.conv2d(tf.add(deconv8, conv3b), 128, [5, 5], 1, 'same', activation_fn=tf.nn.relu)   
    # #conv8a = layers.conv2d(deconv8, 128, [5, 5], 1, 'same', activation_fn=tf.nn.relu)   
    # conv8b = layers.conv2d(conv8a, 128, [5, 5], 1, 'same', activation_fn=tf.nn.relu)   
    # print("Conv 8 shape")
    # print(conv8b.get_shape())

    # deconv9 = tf.layers.conv2d_transpose(conv8b, 128, [5,5], 2, 'same',  activation=tf.nn.relu)
    # print ("Deconv9 shape")
    # print(deconv9.get_shape())
    # conv9a = layers.conv2d(tf.add(deconv9, conv2b), 128, [7, 7], 1, 'same', activation_fn=tf.nn.relu)   
    # #conv9a = layers.conv2d(deconv9, 128, [7, 7], 1, 'same', activation_fn=tf.nn.relu)   
    # conv9b = layers.conv2d(conv9a, 128, [7, 7], 1, 'same', activation_fn=tf.nn.relu)   
    # print("Conv 9 shape")
    # print(conv9b.get_shape())

    # deconv10 = tf.layers.conv2d_transpose(conv9b, 128, [7,7], 2, 'same',  activation=tf.nn.relu)
    # print ("Deconv10 shape")
    # print(deconv10.get_shape())
    # conv10a = layers.conv2d(tf.add(deconv10,conv1b), 128, [11, 11], 1, 'same', activation_fn=tf.nn.relu)   
    # conv10a = layers.conv2d(deconv10, 128, [11, 11], 1, 'same', activation_fn=tf.nn.relu)   
    
    # predict = layers.conv2d(conv10a, 3, [1, 1], 1, 'same', activation_fn=None)   
    # print("Predict Size")
    # print(predict.get_shape())

    conv1 = layers.conv2d(color, 128, [5, 5], 1, 'same', activation_fn=tf.nn.relu)
    max1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2,2], strides=2)
    conv2 = layers.conv2d(max1, 128, [3, 3], 1, 'same', activation_fn=tf.nn.relu)
    
    deconv3 = tf.layers.conv2d_transpose(conv2, 128, [3,3], 2, 'same',  activation=tf.nn.relu)
    conv4 = layers.conv2d(tf.add(deconv3,conv1), 128, [5, 5], 1, 'same', activation_fn=tf.nn.relu)
    predict = layers.conv2d(conv4, 3, [1, 1], 1, 'same', activation_fn=tf.nn.softmax)

    predict_n = tf.nn.l2_normalize(predict, axis=3)
    target_n = tf.nn.l2_normalize(target, axis=3)

    cosine_angle = tf.reduce_sum(predict_n * target_n, axis=3)
    print("Test")
    print(cosine_angle)
    loss = -tf.reduce_mean(tf.boolean_mask(cosine_angle, mask))

    # print("Loss")
    # print(loss)

    return color, mask, target, predict_n, loss


def load_train_data(iteration, batch_size):
    total = 20000
    start = (iteration * batch_size) % total

    color_npy = np.zeros([batch_size, 128, 128, 3], dtype=np.float32)
    mask_npy = np.zeros([batch_size, 128, 128], dtype=np.uint8)
    target_npy = np.zeros([batch_size, 128, 128, 3], dtype=np.float32)

    for i in range(batch_size):
        color_path = osp.join(DATASET_PATH, 'train', 'color', '{}.png'.format(i + start))
        mask_path = osp.join(DATASET_PATH, 'train', 'mask', '{}.png'.format(i + start))
        target_path = osp.join(DATASET_PATH, 'train', 'normal', '{}.png'.format(i + start))
        color_npy[i, ...] = imread(color_path)
        mask_npy[i, ...] = imread(mask_path, as_gray=True)
        target_npy[i, ...] = imread(target_path)

    target_npy = target_npy / 255.0 * 2 - 1

    return color_npy, mask_npy, target_npy


def train():
    color, mask, target, predict_n, loss = build_model()
    loss_summ = tf.summary.scalar('training_loss', loss)
    writer = tf.summary.FileWriter(SUMMARIES_PATH)

    train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    with tf.device("/cpu:0"):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(MAX_ITERATION):
                color_npy, mask_npy, target_npy = load_train_data(i, BATCH_SIZE)
                feed_dict = {
                    color: color_npy,
                    mask: mask_npy,
                    target: target_npy
                }
                loss_val, summ, _ = sess.run([loss, loss_summ, train_op], feed_dict=feed_dict)
                writer.add_summary(summ, i)

                print(loss_val)

            for i in range(20000):
                color_npy, mask_npy, target_npy = load_train_data(i, 1)
                feed_dict = {
                    color: color_npy,
                    mask: mask_npy,
                    target: target_npy
                }
                predict_val, = sess.run([predict_n], feed_dict={color: color_npy, target: target_npy})

                predict_img = ((predict_val.squeeze(0) + 1) / 2 * 255).astype(np.uint8)
                imsave(osp.join(DUMP_FOLDER, '{}.png'.format(i)), predict_img)


if __name__ == '__main__':
    train()
