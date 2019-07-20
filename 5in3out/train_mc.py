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
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
BATCH_FRAME_SIZE = IMAGE_DEPTH*BATCH_SIZE   # batch size times frame size
Tot_EPOCH = 31

logging.basicConfig(level=logging.DEBUG, filename='/home/avisek/vineel/glcic-master/motion_compensation/lossvalues.log',filemode='a+',format='%(asctime)s %(message)s')



def train():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3, IMAGE_DEPTH])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x)
    sess = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)

    mc_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.mc_loss, global_step=global_step, var_list=model.mc_variables)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if tf.train.get_checkpoint_state('/home/avisek/vineel/glcic-master/motion_compensation/backup'):
        saver = tf.train.Saver()
        saver.restore(sess, '/home/avisek/vineel/glcic-master/motion_compensation/backup/latest')

    #x_test = load.load()
    data_path1 = '/home/avisek/vineel/glcic-master/motion_compensation/npy_hmdb/test'
    x_test = np.load(os.path.join(data_path1, 'x_train_128_8.npy'))

    size_test = len(x_test)
    x_test_mc = []
    for ind in range(1, size_test-1):
    	l = []
        l.append(x_test[ind-1])
        l.append(x_test[ind])
        l.append(x_test[ind+1])
        l = np.array(l)
        l = np.moveaxis(l, 0, -1)
        x_test_mc.append(l)
    x_test_mc = np.array(x_test_mc)
    x_test_mc  = np.array([a / 127.5 - 1 for a in x_test_mc])

    #x_test = np.array([a / 127.5 - 1 for a in x_test])
    #step_num1 = int(len(x_train) / BATCH_SIZE)
    #print(x_train[5,5,5,1,1])
    #print(x_train.shape)
    #x_train = x_train[0:50]
    #x_train = np.array([a / 127.5 - 1 for a in x_train])
    #print(np.mean(np.abs(x_train[0,:,:,:,1]-x_train[0,:,:,:,0])))
    step_num = 6512
    #step_num1 = int(len(x_train) / BATCH_SIZE)
    data_path = '/home/avisek/vineel/glcic-master/motion_compensation/npy_hmdb/train/npy'
    out_loc = '/home/avisek/vineel/glcic-master/motion_compensation/sample/'

    while sess.run(epoch) < Tot_EPOCH:

        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('------------epoch: {}'.format(sess.run(epoch)))


        #=======================================    motion compensation network  =========================================================
	#if True:
        for j in tqdm.tqdm(range(1,step_num)):

	    print('loading-------------------------------------------------------------------------------------- '+str(j))
            x_train = np.load(os.path.join(data_path, 'x_train_128_'+str(j)+'.npy'))
	    #print(x_train.shape)
	    size_train = len(x_train)
	    x_train_mc = []
	    for ind in range(1, size_train-1):
	    	l = []
		l.append(x_train[ind-1])
		l.append(x_train[ind])
		l.append(x_train[ind+1])
		l = np.array(l)
		l = np.moveaxis(l, 0, -1)
		x_train_mc.append(l)
	    x_train_mc = np.array(x_train_mc)
            x_train_mc  = np.array([a / 127.5 - 1 for a in x_train_mc])
	    #print(x_train_mc.shape)
            step_num1 = int(len(x_train_mc) / BATCH_SIZE)

            for i in range(step_num1):

                x_batch = x_train_mc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                #x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3, IMAGE_DEPTH)  # reshaping to 5D
		#print(x_batch.shape)
                _, mc_loss,trainable_var = sess.run([mc_train_op, model.mc_loss,model.mc_variables], feed_dict={x: x_batch,is_training: True})
		#pdb.set_trace()
		chk_var = trainable_var[0]
		#print(np.mean(chk_var))
		if (i==0 and j==3):
			x_test_batch = x_test_mc[0:BATCH_SIZE]
			test_loss,compensated_images = sess.run([model.mc_loss,model.warped_frames], feed_dict={x: x_test_batch,is_training: False})
			compensated_images = np.array(compensated_images)
			#print(compensated_images.shape)
			compensated_images =  np.squeeze(compensated_images)
			print('mc loss for test batch in epoch '+str(sess.run(epoch))+'--> '+str(test_loss))
			#print(compensated_images.shape)
			#print('x_batch shape')
			#print(x_batch.shape)
			#print(np.mean(compensated_images))
			#print(np.max(compensated_images))
			#print('mean before mc-->'+str(np.mean(x_batch[:,:,:,:,0]))+'mean after mc-->'+str(np.mean(compensated_images[0:BATCH_SIZE,:,:,:])))
			img_l = compensated_images[0]
			img_r = compensated_images[BATCH_SIZE]
			img_o_l = x_test_batch[0,:,:,:,0]
                        img_o_r = x_test_batch[0,:,:,:,2]
                        img_o_c = x_test_batch[0,:,:,:,1]
			#print(np.mean(np.abs(img_l-img_o_c)))
                        #print(np.mean(np.abs(img_r-img_o_c)))
			img_l = np.array((img_l + 1) * 127.5, dtype=np.uint8)
			img_r = np.array((img_r + 1) * 127.5, dtype=np.uint8)
			img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
			img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
			cv2.imwrite(str(out_loc)+'compensated_t-1_'+str(sess.run(epoch))+'.jpg' ,(img_l))
			cv2.imwrite(str(out_loc)+'compensated_t+1_'+str(sess.run(epoch))+'.jpg' ,(img_r))
			#img_o_l = x_batch[0,:,:,:,0]
			#img_o_r = x_batch[0,:,:,:,2]
			#img_o_c = x_batch[0,:,:,:,1]
			img_o_l = np.array((img_o_l + 1) * 127.5, dtype=np.uint8)
			img_o_r = np.array((img_o_r + 1) * 127.5, dtype=np.uint8)
			img_o_c = np.array((img_o_c + 1) * 127.5, dtype=np.uint8)
			#print(img_o_l.shape)
			#print('ori shape')
			img_o_l = cv2.cvtColor(img_o_l, cv2.COLOR_BGR2RGB)
			img_o_r = cv2.cvtColor(img_o_r, cv2.COLOR_BGR2RGB)
			img_o_c = cv2.cvtColor(img_o_c, cv2.COLOR_BGR2RGB)
			cv2.imwrite(str(out_loc)+'original_t-1_'+str(sess.run(epoch))+'_'+str(test_loss)+'.jpg' ,(img_o_l))
			cv2.imwrite(str(out_loc)+'original_t+1_'+str(sess.run(epoch))+'_'+str(test_loss)+'.jpg' ,(img_o_r))
			cv2.imwrite(str(out_loc)+'original_t_'+str(sess.run(epoch)) +'_'+str(test_loss)+'.jpg',(img_o_c))
			#print(np.mean(np.abs(img_l-img_o_c)))
			#print(np.mean(np.abs(img_r-img_o_c)))
                    	logging.info('epoch: %s   training mc loss:  %s  ', str(sess.run(epoch)),str(mc_loss))

            #print('epoch--'+str(sess.run(epoch))+'   MC loss: {}'.format(mc_loss))
            #logging.info('epoch: %s   MC loss:  %s  ', str(sess.run(epoch)),str(mc_loss))

#            step_num2 = int(len(x_test) / BATCH_FRAME_SIZE)
#            gc_loss=0
#            for i in range(step_num2):
#                x_batch = x_test[i * BATCH_FRAME_SIZE:(i + 1) * BATCH_FRAME_SIZE]
#                x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
#                completion2, c_loss = sess.run([model.completion, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: False})
#                gc_loss += c_loss;

#            gc_loss /= step_num2
#            cnt = sess.run(epoch)
#            x_batch = x_test[:BATCH_FRAME_SIZE]
#            x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
#            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
        saver = tf.train.Saver()
        saver.save(sess, './backup/latest', write_meta_graph=False)
	saver.save(sess, './backup/'+str(sess.run(epoch)), write_meta_graph=False)

if __name__ == '__main__':
    train()
