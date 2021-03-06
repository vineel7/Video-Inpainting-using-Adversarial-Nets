import tensorflow as tf
from image_warp import *


class Network:

    def __init__(self, data_batch):

    	self.flow, self.warped_frames = self.model_mc(data_batch)
	self.mc_loss = self.get_loss(data_batch,self.flow,self.warped_frames)
	self.mc_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mc')
	#print(self.mc_variables)
    def model_mc(self, data_batch):

        if True:
            with tf.variable_scope('mc'):
                #neighboring_frames = tf.expand_dims(tf.concat([data_batch[:,:, :, :, 0], data_batch[:,:, :, :, 2]],axis=0), axis=3)
		neighboring_frames = tf.concat([data_batch[:,:, :, :, 0], data_batch[:,:, :, :, 2]],axis=0)
		#print('neighboring frame shape')
		#print(neighboring_frames.get_shape().as_list())
                lr_input = tf.concat([tf.concat([data_batch[:,:, :, :, 1], data_batch[:,:, :, :, 0]], axis=3),
                                      tf.concat([data_batch[:,:, :, :, 1], data_batch[:,:, :, :, 2]], axis=3)], axis=0)
		#print('input shape')
		#print(lr_input.get_shape().as_list())
                with tf.variable_scope('coarse_flow'):
                    net = tf.layers.conv2d(lr_input, 24, 5, strides=2, activation=tf.nn.relu, padding='same',
                                           name='conv1', kernel_initializer=tf.keras.initializers.he_normal())
		    print('layer1 shape')
		    print(net.get_shape().as_list())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv2', kernel_initializer=tf.keras.initializers.he_normal())
		    print(net.get_shape().as_list())
                    net = tf.layers.conv2d(net, 24, 3, strides=2, activation=tf.nn.relu, padding='same',
                                           name='conv3', kernel_initializer=tf.keras.initializers.he_normal())
		    print(net.get_shape().as_list())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv4', kernel_initializer=tf.keras.initializers.he_normal())
		    print(net.get_shape().as_list())
                    net = tf.layers.conv2d(net, 32, 3, strides=1, activation=tf.nn.tanh, padding='same',
                                           name='conv5', kernel_initializer=tf.keras.initializers.he_normal())
		    print(net.get_shape().as_list())
                    coarse_flow = tf.depth_to_space(net, 4)                      # check
		    print(net.get_shape().as_list())

		    #print('coarse flow output shape')
                    #print(coarse_flow.get_shape().as_list())
                    warped_frames = image_warp(neighboring_frames, coarse_flow)
		    #print('warped frames after course flow output shape')
                    #print(warped_frames.get_shape().as_list())

                ff_input = tf.concat([lr_input, coarse_flow, warped_frames], axis=3)
                with tf.variable_scope('fine_flow'):
                    net = tf.layers.conv2d(ff_input, 24, 5, strides=2, activation=tf.nn.relu, padding='same',
                                           name='conv1', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv2', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv3', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 24, 3, strides=1, activation=tf.nn.relu, padding='same',
                                           name='conv4', kernel_initializer=tf.keras.initializers.he_normal())
                    net = tf.layers.conv2d(net, 8, 3, strides=1, activation=tf.nn.tanh, padding='same',
                                           name='conv5', kernel_initializer=tf.keras.initializers.he_normal())
                    fine_flow = tf.depth_to_space(net, 2)           #check
		    #print('fine flow output shape')
                    #print(fine_flow.get_shape().as_list())
                    flow = coarse_flow + fine_flow

		    #print('final flow output shape')
                    #print(flow.get_shape().as_list())
                    warped_frames = image_warp(neighboring_frames, flow)
		    #print('final mc warped frames output shape')
                    #print(warped_frames.get_shape().as_list())

                    return flow,warped_frames
        else:
            flow = []
            warped_frames = []
            sr_input = data_batch[0]


    def get_loss(self, data_batch, flow,warped_frames):

    	cur_frames = tf.concat([data_batch[:,:, :, :, 1], data_batch[:,:, :, :, 1]], axis=0)
	#print('get loss---- shape of cur frame')
	#print(cur_frames.get_shape().as_list())
        #warp_loss = tf.losses.mean_squared_error(cur_frames, warped_frames)
	warp_loss = 2*tf.nn.l2_loss(cur_frames - warped_frames)
        warp_loss = tf.reduce_mean(warp_loss)

        grad_x_kernel = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32, shape=(1, 3, 1, 1))
        grad_y_kernel = tf.constant([-1.0, 0, 1.0], dtype=tf.float32, shape=(3, 1, 1, 1))
        flow = tf.expand_dims(tf.concat([flow[:, :, :, 0], flow[:, :, :, 1]], axis=0), axis=3)
        flow_grad_x = tf.nn.conv2d(flow, grad_x_kernel, [1, 1, 1, 1], padding='VALID')[:, 1:-1, :, :]
        flow_grad_y = tf.nn.conv2d(flow, grad_y_kernel, [1, 1, 1, 1], padding='VALID')[:, :, 1:-1, :]
        huber_loss = tf.sqrt(0.01 + tf.reduce_sum(flow_grad_x * flow_grad_x + flow_grad_y * flow_grad_y))

        tf.summary.scalar('Warp_loss', warp_loss)
        tf.summary.scalar('Huber_loss', huber_loss)

        return  warp_loss + 0.01 * huber_loss

