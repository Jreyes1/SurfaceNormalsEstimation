#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path as osp
import os

# for tensorboard
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir,now)


from tensorflow.contrib import layers
from skimage.io import imread, imsave

DATASET_PATH = '/home/jorge/Documents/cos429/cos429challenge'
BATCH_SIZE = 16
SUMMARIES_PATH = '/home/jorge/Documents/cos429/cos429challenge/summaries'
MAX_ITERATION = 4000

DUMP_FOLDER = '/home/jorge/Documents/cos429/cos429challenge/prediction'

if not osp.exists(DUMP_FOLDER):
    os.makedirs(DUMP_FOLDER)

# to indicate that batch normalization is along channels (channels last)
chanDim = -1
############ CONV MODULE #########################################
def conv_unit(data, numConv, kernelSize, filterDepth):

    
    x = data


    # convolution layers
    for i in range(0, numConv):
        x = tf.layers.conv2d(inputs=x, filters=filterDepth, kernel_size=kernelSize, strides=1,
            padding="same",activation=tf.nn.relu)
        x = tf.layers.batch_normalization(inputs=x, axis = chanDim,momentum = 0.5)

    # pooling layer
    x = tf.layers.max_pooling2d(inputs=x,pool_size=[2,2],strides=2)
    
    return x

############ DECONV MODULE #########################################

def deconv_unit(data, numConv, kernelSize, filterDepth, last):
    x = data

    # upsampling layer
    x = tf.layers.conv2d_transpose(x, filterDepth, kernelSize, 2, 'same',
      activation=tf.nn.relu) 

    # convolution layers
    for i in range(0, numConv):
        x = tf.layers.conv2d(inputs=x, filters=filterDepth, kernel_size=kernelSize, strides=1,
            padding="same",activation=tf.nn.relu)
        x = tf.layers.batch_normalization(inputs=x, axis = chanDim,momentum = 0.5) 
        
        # perform additional convoltion to very last output of architecture to have filter
        # depth of 3 and perform softmax
        if (last and i == (numConv-1)):
            # x = tf.nn.softmax(x) 
            x = tf.layers.conv2d(inputs=x, filters=3, kernel_size=kernelSize, strides=1,
            padding="same",activation=tf.nn.softmax)
                  
    return x





def build_model():
    color = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])
    mask = tf.placeholder(dtype=tf.bool, shape=[None, 128, 128])
    target = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])

    

    numConv = [2,2,3,3,3,1]
    kernelSize = [[3,3],[3,3],[3,3],[3,3],[3,3],[1,1]]
    filterDepth = [64, 128, 256, 512,512,1024]

    downConv = [None] * (len(numConv) + 1)
    downConv[0] = color
    print("Size of DownConv: 0")
    print(downConv[0].get_shape())

    ## down convolutions
    for i in range(1,len(numConv)+1):
        downConv[i] = conv_unit(downConv[i-1],numConv[i-1],kernelSize[i-1],filterDepth[i-1])
        print("Size of DownConv %d" % (i))
        print(downConv[i].get_shape())

    print ("Bottle Neck")
    
    last = False
    upConv = [None] * (len(numConv)+1)
    upConv[0] = downConv[i]
    print("Size of UpConv 0")
    print(upConv[0].get_shape())
    
    # up convolutions
    # pair = len(numConv)
    # for i in range(1,len(numConv)+1):
    #     if (i == len(numConv)):
    #         last = True
    #     upConv[i] = deconv_unit(tf.add(upConv[i-1],downConv[pair]),numConv[pair-1],
    #         kernelSize[pair-1],filterDepth[pair-1], last)
    #     print("Size After UpConv %d" % i)
    #     print(upConv[i].get_shape())

    pair = len(numConv)
    for i in range(1,len(numConv)+1):
        if (i == len(numConv)):
            last = True
        upConv[i] = deconv_unit(upConv[i-1],numConv[pair-1],kernelSize[pair-1],filterDepth[pair-1], last)
        print("Size After UpConv %d" % i)
        print(upConv[i].get_shape())
        pair = pair -1

    predict = upConv[i]


    predict_n = tf.nn.l2_normalize(predict, axis=3)
    target_n = tf.nn.l2_normalize(target, axis=3)

    cosine_angle = tf.reduce_sum(predict_n * target_n, axis=3)

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

    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    with tf.device('/gpu:0'):
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
