import numpy as np
import tensorflow as tf
import cv2
import tqdm
from network_5in3_mc import Network
import load
import logging
from imutils import face_utils
import imutils
import dlib


IMAGE_SIZE = 128
IMAGE_DEPTH = 5  # no of frames
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
BATCH_FRAME_SIZE = IMAGE_DEPTH*BATCH_SIZE   # batch size times frame size
PRETRAIN_EPOCH = 0
Td_EPOCH = 10
Tot_EPOCH = 40

logging.basicConfig(level=logging.DEBUG, filename='/home/avisek/vineel/glcic-master/5_in_3_video/lossvalues_prevcor.log',filemode='a+',format='%(asctime)s %(message)s')

dir1='/home/avisek/vineel/glcic-master/5_in_3_video/output_5in3_prevcor/'
dir2='/home/avisek/vineel/glcic-master/5_in_3_video/original_5in3_prevcor/'
dir3='/home/avisek/vineel/glcic-master/5_in_3_video/perturbed_5in3_prevcor/'

detector = dlib.get_frontal_face_detector()

def train():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, 3, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])
    points = tf.placeholder(tf.int32, [BATCH_SIZE, 3, 4])

    model = Network(x, mask, points, local_x, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)

    g_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
    combined_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(model.combined_loss, global_step=global_step, var_list=model.g_variables)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    variables_mc_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mc')
    checkpoint_path_mc = '/home/avisek/vineel/glcic-master/motion_compensation/backup/latest'
    saver_pre_mc = tf.train.Saver(variables_mc_restore)
    saver_pre_mc.restore(sess, checkpoint_path_mc)

    variables_inpaint = tf.trainable_variables()
    variables_inpaint_torestore = [var for var in variables_inpaint if var not in variables_mc_restore]
    checkpoint_path_inpaint = '/home/avisek/vineel/glcic-master/5_in_3_video/backup_5_in_3/latest'
    saver_pre_inpaint = tf.train.Saver(variables_inpaint_torestore)
    saver_pre_inpaint.restore(sess,checkpoint_path_inpaint)

    """if tf.train.get_checkpoint_state('/home/avisek/vineel/glcic-master/5_in_3_video/backup_5_in_3'):
        saver = tf.train.Saver()
        saver.restore(sess, '/home/avisek/vineel/glcic-master/5_in_3_video/backup_5_in_3/latest')"""

    print('restoring model done.. loading numpy')

    x_train, x_test = load.load()
    print('loading numpy done.. normalizing')
    x_train = np.array([a / 127.5 - 1 for a in x_train])
    x_test = np.array([a / 127.5 - 1 for a in x_test])

    print('normalization done..')

    step_num = int(len(x_train) / BATCH_FRAME_SIZE)

    while sess.run(epoch) < Tot_EPOCH:
        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))


        #=======================================    Completion network - phase 1 =========================================================
        if sess.run(epoch) <= PRETRAIN_EPOCH:
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_FRAME_SIZE:(i + 1) * BATCH_FRAME_SIZE]
                points_batch, mask_batch = get_points2(x_batch)
                x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
                mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to 5D

                _, g_loss = sess.run([g_train_op, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: True})

                if i%50==0:
                    print('epoch--'+str(sess.run(epoch))+'   Completion loss: {}'.format(g_loss))
                    logging.info('epoch: %s   batch: %s   Completion loss:  %s  ', str(sess.run(epoch)),str(i),str(g_loss))

            step_num2 = int(len(x_test) / BATCH_FRAME_SIZE)
            gc_loss=0
            for i in range(step_num2):
                x_batch = x_test[i * BATCH_FRAME_SIZE:(i + 1) * BATCH_FRAME_SIZE]
                points_batch, mask_batch = get_points2(x_batch)
                x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
                mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to 5D
                completion2, c_loss = sess.run([model.completion_3d, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: False})
                gc_loss += c_loss;

            gc_loss /= step_num2
            cnt = sess.run(epoch)
            x_batch = x_test[:BATCH_FRAME_SIZE]
            _, mask_batch = get_points2(x_batch)
            x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
            mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to  5D
            x_pert = x_batch * (1-mask_batch)
            completion = sess.run(model.completion_3d, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            save_samples(dir1, cnt,gc_loss, completion)
            save_samples(dir2, cnt,gc_loss, x_batch[:, 1:4, :, :, :])
            save_samples(dir3, cnt,gc_loss, x_pert[:, 1:4, :, :, :])
            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)
            saver.save(sess, './backup/'+str(sess.run(epoch)), write_meta_graph=False)

        #=======================================    Discrimitation network - phase 2 ========================================================
        elif sess.run(epoch) <= (PRETRAIN_EPOCH+Td_EPOCH):
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_FRAME_SIZE:(i + 1) * BATCH_FRAME_SIZE]
                points_batch, mask_batch = get_points2(x_batch)
                x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
                mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to 5D
                points_batch = points_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, 4)  # reshaping to 5D
                points_batch = points_batch[:, 1:4, :]
                local_x_batch = []

                for j in range(BATCH_SIZE):
                	for k in range(3):
                		x1, y1, x2, y2 = points_batch[j][k]
                		local_x_batch.append(x_batch[j][1+k][int(y1):int(y2), int(x1):int(x2), :])
                local_x_batch = np.array(local_x_batch)
                local_x_batch = local_x_batch.reshape(BATCH_SIZE, 3, LOCAL_SIZE, LOCAL_SIZE, 3)


                _, d_loss, dfake_loss = sess.run(
                    [d_train_op, model.d_loss, model.dfake_loss],
                    feed_dict={x: x_batch, mask: mask_batch, points: points_batch, local_x: local_x_batch, is_training: True})

                if i%50==0:
                    print('epoch--'+str(sess.run(epoch))+'   Discriminator loss: {}'.format(d_loss))
                    logging.info('epoch: %s batch:  %s Discriminator loss:  %s GAN loss:  %s ', str(sess.run(epoch)),str(i), str(d_loss), str(dfake_loss))
            step_num2 = int(len(x_test) / BATCH_FRAME_SIZE)
            gc_loss=0
            for i in range(step_num2):
                x_batch = x_test[i * BATCH_FRAME_SIZE:(i + 1) * BATCH_FRAME_SIZE]
                points_batch, mask_batch = get_points2(x_batch)
                x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
                mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to 5D
                completion2, c_loss = sess.run([model.completion_3d, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: False})
                gc_loss += c_loss

            gc_loss /= step_num2
            cnt = sess.run(epoch)
            x_batch = x_test[:BATCH_FRAME_SIZE]
            _, mask_batch = get_points2(x_batch)
            x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
            mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to  5D
            x_pert = x_batch * (1-mask_batch)
            completion = sess.run(model.completion_3d, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            save_samples(dir1, cnt,gc_loss, completion)
            save_samples(dir2, cnt,gc_loss, x_batch[:, 1:4, :, :, :])
            save_samples(dir3, cnt,gc_loss, x_pert[:, 1:4, :, :, :])
            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)
            saver.save(sess, './backup/'+str(sess.run(epoch)), write_meta_graph=False)
        # ==================================== generator & discriminator networks - Phase 3 =========================================
        else:
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_FRAME_SIZE:(i + 1) * BATCH_FRAME_SIZE]
                points_batch, mask_batch = get_points2(x_batch)
                x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
                mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to 5D
                points_batch = points_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, 4)  # reshaping to 5D
                points_batch = points_batch[:, 1:4, :]
                local_x_batch = []
                for j in range(BATCH_SIZE):
                	for k in range(3):
                        	x1, y1, x2, y2 = points_batch[j][k]
                        	local_x_batch.append(x_batch[j][1+k][int(y1):int(y2), int(x1):int(x2), :])
                local_x_batch = np.array(local_x_batch)
                local_x_batch = local_x_batch.reshape(BATCH_SIZE, 3, LOCAL_SIZE, LOCAL_SIZE, 3)

                _, d_loss = sess.run(
                    [d_train_op, model.d_loss],
                    feed_dict={x: x_batch, mask: mask_batch, points: points_batch, local_x: local_x_batch, is_training: True})
                
                _, combined_loss, g_loss, dfake_loss = sess.run([combined_op, model.combined_loss, model.g_loss, model.dfake_loss], feed_dict={x: x_batch, mask: mask_batch, points: points_batch, local_x: local_x_batch, is_training: True})

                
                if i%50==0:
                    #print('Combined loss: {}'.format(combined_loss))
                    print('epoch--'+str(sess.run(epoch))+'   Combined loss: {}'.format(combined_loss))
                    #print('Discriminator loss: {}'.format(d_loss))
                    print('epoch--'+str(sess.run(epoch))+'   Discriminator loss: {}'.format(d_loss))
                    print('epoch--'+str(sess.run(epoch))+'   MSE loss: {}'.format(g_loss))
                    print('epoch--'+str(sess.run(epoch))+'   GAN loss: {}'.format(dfake_loss))
                    logging.info('epoch: %s batch:  %s combined loss:  %s Discriminator loss:  %s ', str(sess.run(epoch)), str(i),str(combined_loss), str(d_loss))
                    logging.info('epoch: %s batch:  %s MSE loss:  %s GAN loss: %s', str(sess.run(epoch)), str(i),str(g_loss),str(dfake_loss))


            step_num2 = int(len(x_test) / BATCH_FRAME_SIZE)
            gc_loss=0
            for i in range(step_num2):
                x_batch = x_test[i * BATCH_FRAME_SIZE:(i + 1) * BATCH_FRAME_SIZE]
                points_batch, mask_batch = get_points2(x_batch)
                x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
                mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to 5D
                completion2, c_loss = sess.run([model.completion_3d, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: False})
                gc_loss += c_loss;

            gc_loss /= step_num2
            cnt = sess.run(epoch)
            x_batch = x_test[:BATCH_FRAME_SIZE]
            _, mask_batch = get_points2(x_batch)
            x_batch = x_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 3)  # reshaping to 5D
            mask_batch = mask_batch.reshape(BATCH_SIZE, IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE, 1)  # reshaping to  5D
            x_pert = x_batch * (1-mask_batch)
            completion = sess.run(model.completion_3d, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            #img_tile(dir1, cnt,gc_loss, completion[:, 1, :, :, :])
            #img_tile2(dir2, cnt,gc_loss, x_batch[:, 2, :, :, :])
            #img_tile2(dir3, cnt,gc_loss, x_pert[:, 2, :, :, :])
            save_samples(dir1, cnt,gc_loss, completion)
            save_samples(dir2, cnt,gc_loss, x_batch[:, 1:4, :, :, :])
            save_samples(dir3, cnt,gc_loss, x_pert[:, 1:4, :, :, :])
            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)
            saver.save(sess, './backup/'+str(sess.run(epoch)), write_meta_graph=False)

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
        m[q1:q2+1 , p1:p2+1 ] = 1
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
            x,y,w,h = np.array([rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top()])
        
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
	#if i%3!=0:
        m[int(q1):int(q2) + 1, int(p1):int(p2) + 1] = 1
        mask.append(m)


    return np.array(points), np.array(mask)


def img_tile(loc, cnt, out_loss, imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
 	if imgs.ndim != 3 and imgs.ndim != 4:
 		raise ValueError('imgs has wrong number of dimensions.')
 	n_imgs = imgs.shape[0]

 	tile_shape = None
 	# Grid shape
 	img_shape = np.array(imgs.shape[1:3])
 	if tile_shape is None:
 		img_aspect_ratio = img_shape[1] / float(img_shape[0])
 		aspect_ratio *= img_aspect_ratio
 		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
 		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
 		grid_shape = np.array((tile_height, tile_width))
 	else:
 		assert len(tile_shape) == 2
 		grid_shape = np.array(tile_shape)

 	# Tile image shape
 	tile_img_shape = np.array(imgs.shape[1:])
 	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

 	# Assemble tile image
 	tile_img = np.empty(tile_img_shape)
 	tile_img[:] = border_color
 	for i in range(grid_shape[0]):
 		for j in range(grid_shape[1]):
 			img_idx = j + i*grid_shape[1]
 			if img_idx >= n_imgs:
 				# No more images - stop filling out the grid.
 				break
 			img = imgs[img_idx]
 			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

 			yoff = (img_shape[0] + border) * i
 			xoff = (img_shape[1] + border) * j
 			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

 	cv2.imwrite(str(loc)+'img_'+'epoch_'+str(cnt)+'_loss_'+str(out_loss) + '.jpg', (tile_img + 1)*127.5)


def img_tile2(loc, cnt, out_loss, imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
 	if imgs.ndim != 3 and imgs.ndim != 4:
 		raise ValueError('imgs has wrong number of dimensions.')
 	n_imgs = imgs.shape[0]

 	tile_shape = None
 	# Grid shape
 	img_shape = np.array(imgs.shape[1:3])
 	if tile_shape is None:
 		img_aspect_ratio = img_shape[1] / float(img_shape[0])
 		aspect_ratio *= img_aspect_ratio
 		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
 		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
 		grid_shape = np.array((tile_height, tile_width))
 	else:
 		assert len(tile_shape) == 2
 		grid_shape = np.array(tile_shape)

 	# Tile image shape
 	tile_img_shape = np.array(imgs.shape[1:])
 	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

 	# Assemble tile image
 	tile_img = np.empty(tile_img_shape)
 	tile_img[:] = border_color
 	for i in range(grid_shape[0]):
 		for j in range(grid_shape[1]):
 			img_idx = j + i*grid_shape[1]
 			if img_idx >= n_imgs:
 				# No more images - stop filling out the grid.
 				break
 			img = imgs[img_idx]
 			img = np.array((img + 1) * 127.5, dtype=np.uint8)
 			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

 			yoff = (img_shape[0] + border) * i
 			xoff = (img_shape[1] + border) * j
 			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

 	cv2.imwrite(str(loc)+'img_'+'epoch_'+str(cnt)+'_loss_'+str(out_loss) + '.jpg', (tile_img))

def save_samples(dir,cnt,loss,x_batch):

	frames_no = 3
	for i in range(BATCH_SIZE):
		for j in range(frames_no):
			frame = x_batch[i][j]
			frame = np.array((frame + 1) * 127.5, dtype=np.uint8)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			cv2.imwrite(str(dir)+'inpainted_epoch_'+str(cnt)+'frame_%02d.jpg' % (3*i+j) ,(frame))

if __name__ == '__main__':
    train()

