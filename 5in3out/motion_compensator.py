import numpy as np
import tensorflow as tf
import cv2
import tqdm
from network_mc import Network
import load
import logging
from imutils import face_utils
import imutils
import dlib
import os
import pdb

IMAGE_SIZE = 128
IMAGE_DEPTH = 3  # no of frames
BATCH_SIZE = 16
BATCH_FRAME_SIZE = IMAGE_DEPTH*BATCH_SIZE   # batch size times frame size


def test():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3, IMAGE_DEPTH])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x)
    sess = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    if tf.train.get_checkpoint_state('/home/avisek/vineel/glcic-master/motion_compensation/backup'):
    	saver = tf.train.Saver()
        saver.restore(sess, '/home/avisek/vineel/glcic-master/motion_compensation/backup/latest')

