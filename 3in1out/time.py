import poissonblending 
import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
from imutils import face_utils
import imutils
import dlib
sys.path.append('..')
from network5 import Network
from pytictoc import TicToc

IMAGE_SIZE = 128
LOCAL_SIZE = 64
BATCH_SIZE = 4
PRETRAIN_EPOCH = 100

test_npy = './test.npy'
mask_npy = './mask.npy'

detector = dlib.get_frontal_face_detector()

t = TicToc()

def test():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask3 = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    points = tf.placeholder(tf.int32, [BATCH_SIZE, 4])
    
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, mask3, points, local_x, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()
    saver.restore(sess, '../backup/latest')

    x_test = np.load(test_npy)
    mask = np.load(mask_npy)
    #mask = np.array(mask)
    print('\n')
    print(mask.shape)
    #np.random.shuffle(x_test)
    x_test = np.array([a / 127.5 - 1 for a in x_test])
    #mask = np.array([a / 255 for a in mask])

    step_num = int(len(x_test) / BATCH_SIZE)

    cnt = 0
    #t.tic()
    for i in tqdm.tqdm(range(step_num)):
        x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        mask_batch = mask[i*BATCH_SIZE:(i + 1) * BATCH_SIZE]
        t.tic()
        completion = sess.run(model.completion, feed_dict={x: x_batch, mask3: mask_batch, is_training: False})
        t.toc('elapsed time:')
        imitation = sess.run(model.imitation, feed_dict={x: x_batch, mask3: mask_batch, is_training: False})
       
        for i in range(BATCH_SIZE):
            cnt += 1
            raw = x_batch[i]
            raw2 = np.array((raw + 1) * 127.5, dtype=np.uint8)
            #masked = raw2 * (1 - mask_batch[i]) + np.ones_like(raw2) * mask_batch[i] * 255
            masked = raw2 * (1 - mask_batch[i]) 
            
            mask1 = (mask_batch[i])*255
            mask2 = mask_batch[i]
            #xc = int(points_batch[i][0]+(points_batch[i][2]-points_batch[i][0])/2);
            #yc = int(points_batch[i][1]+(points_batch[i][3]-points_batch[i][1])/2);
            img = completion[i]
            img3 = imitation[i]
            img2 = np.array((img + 1) * 127.5, dtype=np.uint8)
            raw2 = cv2.cvtColor(raw2, cv2.COLOR_BGR2RGB)
     
            masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            pm = poisson_blend(raw, img3, 1-mask2)
            pm2 = np.array((pm + 1) * 127.5, dtype=np.uint8)
            pm2 = cv2.cvtColor(pm2, cv2.COLOR_BGR2RGB)
            #normal_clone = cv2.seamlessClone(img, raw, mask1, (xc,yc), cv2.NORMAL_CLONE);
            #mixed_clone = cv2.seamlessClone(img2, raw2, mask1, (xc,yc), cv2.MIXED_CLONE);
            dst = './output8/'.format("{0:06d}".format(cnt))
            #output_image([['Input', masked], ['Output', img2], ['Ground Truth', raw2]], dst, cnt)
            output_image([['P_Blending', pm2]], dst, cnt)
            output_image([['Input', masked], ['Output', img2], ['Ground Truth', raw2], ['Mask Batch', mask1]], dst, cnt)
            #output_image([['Input', masked], ['Output', img], ['Ground Truth', raw]], dst, cnt)
    #t.toc('Elapsed time: ')

  

def poisson_blend(imgs1, imgs2, mask):
        # call this while performing correctness experiment
     out = np.zeros(imgs1.shape)

        #for i in range(0, len(imgs1)):
     img1 = (imgs1 + 1.)/2.
     img2 = (imgs2 + 1.)/2.
     out =  np.clip((poissonblending.blend(img1, img2, 1 - mask) - 0.5) * 2, -1.0, 1.0)
     return out

def output_image(images, dst, cnt):
    for i, image in enumerate(images):
        text, img = image
        cv2.imwrite(str(dst)+'img_'+str(cnt)+'_'+text+'.jpg', (img))


if __name__ == '__main__':
    test()
    
