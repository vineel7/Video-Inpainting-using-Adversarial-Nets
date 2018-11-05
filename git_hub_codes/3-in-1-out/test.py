import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from network import Network
import load
import logging
from imutils import face_utils
import imutils
import dlib

IMAGE_SIZE = 128
IMAGE_DEPTH = 3  # no of frames
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
BATCH_FRAME_SIZE = IMAGE_DEPTH*BATCH_SIZE   # batch size times frame size
PRETRAIN_EPOCH = 20
Td_EPOCH = 10
Tot_EPOCH = 60

#test_npy = './lfw.npy'
logging.basicConfig(level=logging.DEBUG, filename='/home/avisek/vineel/glcic-master/3_in_1_video/test_output/lossvalues.log',filemode='a+',format='%(asctime)s %(message)s')

out_loc1='/home/avisek/vineel/glcic-master/3_in_1_video/test_output/original'
out_loc2='/home/avisek/vineel/glcic-master/3_in_1_video/test_output/inpainted'
out_loc3='/home/avisek/vineel/glcic-master/3_in_1_video/test_output/perturbed'

detector = dlib.get_frontal_face_detector()

def test():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])
    points = tf.placeholder(tf.int32, [BATCH_SIZE, 4])

    #model = Network(x, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    model = Network(x, mask, points, local_x, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if tf.train.get_checkpoint_state('./backup'):
    	saver = tf.train.Saver()
    	saver.restore(sess, '../backup/latest')

    _, x_test = load.load()
    #np.random.shuffle(x_test)
    x_test = np.array([a / 127.5 - 1 for a in x_test])

    step_num = int(len(x_test) / BATCH_SIZE)

    cnt = 0
    for i in tqdm.tqdm(range(step_num)-1):
	x_batch = []
	for j in range(BATCH_SIZE):
		x_batch.append(x_test[BATCH_SIZE*i+j:BATCH_SIZE*i+j+3])
	np.array(x_batch)
        #x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        #_, mask_batch = get_points()
	points_batch, mask_batch = get_points2(x_batch)
	x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
	mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to 5D
	completion2, c_loss = sess.run([model.completion, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: False})
        #completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
	gc_loss += c_loss

	x_batch_test = x_batch[:, 1, :, :, :]
	x_pert = x_batch * (1-mask_batch)
	x_pert_test = x_pert[:, 1, :, :, :]

        for i in range(BATCH_SIZE):
            cnt += 1
            raw = x_batch_test[i]
            raw = np.array((raw + 1) * 127.5, dtype=np.uint8)
            perturbed = x_pert_test[i]
	    perturbed = np.array((perturbed + 1) * 127.5, dtype=np.uint8)
            #masked = raw * (1 - mask_batch[i]) + np.ones_like(raw) * mask_batch[i] * 255
            img = completion2[i]
            img = np.array((img + 1) * 127.5, dtype=np.uint8)
            #dst = ''/home/avisek/vineel/glcic-master/3_in_1_video/test_output/{}.jpg'.format("{0:06d}".format(cnt))
            #output_image([['Input', masked], ['Output', img], ['Ground Truth', raw]], dst)
	    cv2.imwrite(str(out_loc1)+'original_'+str(cnt) + '.jpg', (raw))
	    cv2.imwrite(str(out_loc2)+'inpainted_'+str(cnt)+'_loss_'+str(c_loss) + '.jpg', (img))
	    cv2.imwrite(str(out_loc3)+'perturbed_'+str(cnt) + '.jpg', (perturbed))
    logging.info('Completion loss:  %s  ', str(gc_loss/step_num))
def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h
        
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)
    

def get_points2(x_batch):
    points = []
    mask = []
    global x,y,w,h
    #x,y,w,h=np.array([35,35,60,60])
    for i in range(BATCH_FRAME_SIZE):
        image = np.array((x_batch[i] + 1) * 127.5, dtype=np.uint8)
        rects = detector(image,1)
        for (k, rect) in enumerate(rects):
            x,y,w,h = np.array([rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()])def get_points2(x_batch):
        Face_Hole_max_w = min(HOLE_MAX,int(w))
        Face_Hole_max_h = min(HOLE_MAX,int(h))
        L = np.random.randint(HOLE_MIN, Face_Hole_max_w)
        M = np.random.randint(HOLE_MIN, Face_Hole_max_h)
        p1 = x + np.random.randint(0, w - L)
        q1 = y + np.random.randint(0, h - M)
        p2 = p1 + L
        q2 = q1 + M
        p3 = p1 + int(L/2)
        q3 = q1 + int(M/2)
        x1 = p3 - LOCAL_SIZE/2
        if(x1<1):
            t = 1 - x1;
            x1 = 1;
            p2 = p2 + t;
            p1 = p1 + t;
            p3 = p3 + t;
        y1 = q3 - LOCAL_SIZE/2
        if(y1<1):
            t = 1 - y1;
            y1 = 1;
            q2 = q2 + t;
            q1 = q1 + t;
            q3 = q3 + t;
        x2 = p3 + LOCAL_SIZE/2
        if(x2>127):
            t = x2 -127;
            x2 = 127;
            p2 = p2 - t;
            p1 = p1 - t;
            x1 = x1 - t;
        y2 = q3 + LOCAL_SIZE/2
        if(y2>127):
            t = y2 -127;
            y2 = 127;
            q2 = q2 - t;
            q1 = q1 - t;
            y1 = y1 - t;
        points.append([x1, y1, x2, y2])
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[int(q1):int(q2) + 1, int(p1):int(p2) + 1] = 1
        mask.append(m)


    return np.array(points), np.array(mask)

def output_image2(images, dst):
    fig = plt.figure()
    for i, image in enumerate(images):
        text, img = image
        fig.add_subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.xlabel(text)
    plt.savefig(dst)
    plt.close()

#def output_image(images, dst):
#	cv2.imwrite(str(loc)+'img_'+'epoch_'+str(cnt)+'_loss_'+str(out_loss) + '.jpg', (tile_img))
if __name__ == '__main__':
    test()
    
