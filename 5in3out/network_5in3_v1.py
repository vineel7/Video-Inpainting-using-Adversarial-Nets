from layer import *
import numpy as np

class Network:
    def __init__(self, x, mask, points, local_x, is_training, batch_size):
        self.alpha = 1
        self.batch_size = batch_size

#	self.imitation_3d = []
#	self.completion_3d = []
#        for i in range(3):
#		x_3d = []
#		mask_3d = []
#		for j in range(3):
#			x_3d = x_3d.append(x[:, i+j, :, :, :])
#			mask_3d = mask_3d.append(mask[:, i+j, :, :, :])
#		x_3d = tf.convert_to_tensor(x_3d)
#		print('x_3d shape')
#		print(tf.shape(x_3d))
#		mask_3d = tf.convert_to_tensor(mask_3d)
#		print(tf.shape(mask_3d))
#		if i==0:
#			self.imitation = self.generator(x_3d * (1 - mask_3d), is_training,reuse=False)
#		else
#			self.imitation = self.generator(x_3d * (1 - mask_3d), is_training,reuse=True)
#       	self.completion = self.imitation * mask_3d[:, 1, :, :, :] + x_3d[:, 1, :, :, :] * (1 - mask_3d[:, 1, :, :, :])
#		self.imitation_3d.append(self.imitation)
#		self.completion_3d.append(self.completion)

#	self.completion_3d = tf.convert_to_tensor(self.completion_3d)
#	print('generator output shape')
#	print(tf.shape(self.completion_3d))

        self.completion_3d,self.imitation_3d = self.completion_3d(x, mask,is_training)
        #print('generator output shape -- ')
        #print(self.completion_3d.get_shape().as_list())

#        self.local_completionG = []

#        for j in range(batch_size):
#	    for k in range(3):
#	            cord = points[j][k]
#	            self.cropped = tf.image.crop_to_bounding_box(self.completion_3d[j,k,:,:,:], cord[1],cord[0],cord[3]-cord[1],cord[2]-cord[0])
#	            self.local_completionG.append(self.cropped)

#        self.local_completionG = tf.convert_to_tensor(self.local_completionG)
#        self.local_completionG = tf.reshape(self.local_completionG, [batch_size, 3, 64, 64, 3])

        self.local_completionG = self.local_completion(self.completion_3d,points)
        #print('shape of local completion output')
        #print(self.local_completionG.get_shape().as_list())

        self.real = self.discriminator(x[:, 1:4, :, :, :], local_x, reuse=False)
        self.fake = self.discriminator(self.completion_3d, self.local_completionG, reuse=True)
        self.g_loss = self.calc_g_loss(x[:, 1:4, :, :, :], self.completion_3d)/(3*self.batch_size)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.dfake_loss = self.calc_dfake_loss(self.fake)
        self.combined_loss = self.g_loss + self.alpha * self.dfake_loss
        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')




    def completion_3d(self, x, mask,is_training):
        self.imitation_3d = []
        self.completion_3d = []
        for i in range(3):
                #self.x_3d = []
                #self.mask_3d = []
                #for j in range(3):
                #        self.x_3d.append(x[:, i+j, :, :, :])
                #        self.mask_3d.append(mask[:, i+j, :, :, :])
                self.x_3d = x[:,i:i+3,:,:,:]
                #print("type is :", type(self.x_3d))
                mask_center = mask[:,i+1,:,:,:]
                #print(mask_center.get_shape().as_list())
                self.mask_3d = tf.stack([mask_center, mask_center, mask_center],1)
                #print(self.mask_3d.get_shape().as_list())
                #self.mask_3d = tf.concat([self.mask_3d, self.mask_3d],1)
                #print(self.mask_3d.get_shape().as_list())
                #self.x_3d = tf.convert_to_tensor(self.x_3d)
                #print('x_3d shape')
                #print(tf.shape(self.x_3d))
        	#print(self.x_3d.get_shape().as_list())
                #self.mask_3d = tf.convert_to_tensor(self.mask_3d)
                #print(tf.shape(self.mask_3d))
        	#print('mask_3d shape')
        	#print(self.mask_3d.get_shape().as_list())
                if i==0:
                        self.imitation = self.generator(self.x_3d * (1 - self.mask_3d), is_training,reuse=False)
                else:
                        self.imitation = self.generator(self.x_3d * (1 - self.mask_3d), is_training,reuse=True)
                self.completion = self.imitation * self.mask_3d[:, 1, :, :, :] + self.x_3d[:, 1, :, :, :] * (1 - self.mask_3d[:, 1, :, :, :])
                self.imitation_3d.append(self.imitation)
                self.completion_3d.append(self.completion)
		#print('self_imitation shape -- ')
		#print(self.imitation_3d.get_shape().as_list())
		#print('self_completion shape -- ')
                #print(self.completion_3d.get_shape().as_list())
        self.completion_3d = tf.convert_to_tensor(self.completion_3d)
        self.completion_3d = tf.transpose(self.completion_3d,[1,0,2,3,4])
        self.imitation_3d = tf.convert_to_tensor(self.imitation_3d)
        self.imitation_3d = tf.transpose(self.imitation_3d,[1,0,2,3,4])
        #print('generator output shape')
        #print(tf.shape(self.completion_3d))

        return self.completion_3d,self.imitation_3d

    def local_completion(self,completion_3d,points):
        self.local_completionG = []
        for j in range(self.batch_size):
            for k in range(3):
                    cord = points[j][k]
                    self.cropped = tf.image.crop_to_bounding_box(self.completion_3d[j,k,:,:,:], cord[1],cord[0],cord[3]-cord[1],cord[2]-cord[0])
                    self.local_completionG.append(self.cropped)

        self.local_completionG = tf.convert_to_tensor(self.local_completionG)
        self.local_completionG = tf.reshape(self.local_completionG, [self.batch_size, 3, 64, 64, 3])

        return self.local_completionG

    def generator(self, x, is_training,reuse):
        with tf.variable_scope('generator',reuse=reuse):
            with tf.variable_scope('conv1'):
                print('gen layer1 input shape -- ')
                print(x.get_shape().as_list())
                x = conv3_layer(x, [3, 5, 5, 3, 64], 1)
                print(x.get_shape().as_list())
                x = batch_normalize_3d(x, is_training)
                x = tf.nn.relu(x)
                x = tf.squeeze(x,axis=[1])
                print('gen layer1 out shape')
                print(x.get_shape().as_list())
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 256], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated1'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated2'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 4)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated3'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 8)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated4'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 16)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv1'):
                x = deconv_layer(x, [4, 4, 128, 256], [self.batch_size, 64, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv9'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [4, 4, 64, 128], [self.batch_size, 128, 128, 64], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv10'):
                x = conv_layer(x, [3, 3, 64, 32], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv11'):
                x = conv_layer(x, [3, 3, 32, 3], 1)
                x = tf.nn.tanh(x)

        return x


    def discriminator(self, global_x, local_x, reuse):
        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('conv1'):
		    #print('glo dis input shape')
                    #print(x.get_shape().as_list())
                    x = conv3_layer(x, [3, 5, 5, 3, 64], 2)
                    x = batch_normalize_3d(x, is_training)
                    x = tf.nn.relu(x)
                    x = tf.squeeze(x,axis=[1])
		    #print('shape of global dis 1 layer out')
                    #print(x.get_shape().as_list())
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv5'):
                    x = conv_layer(x, [5, 5, 512, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        def local_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('local'):
                with tf.variable_scope('conv1'):
		    #print('loc dis input shape')
		    #print(x.get_shape().as_list())
                    x = conv3_layer(x, [3, 5, 5, 3, 64], 2)
                    x = batch_normalize_3d(x, is_training)
                    x = tf.nn.relu(x)
                    x = tf.squeeze(x,axis=[1])
		    #print('shape of local dis 1 layer out')
		    #print(x.get_shape().as_list())
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        with tf.variable_scope('discriminator', reuse=reuse):
            global_output = global_discriminator(global_x)
            local_output = local_discriminator(local_x)
            with tf.variable_scope('concatenation'):
                output = tf.concat((global_output, local_output), 1)
                output = full_connection_layer(output, 1)
        return output


    def calc_g_loss(self, x, completion):
        print('input to loss '+str(x.get_shape().as_list()))
        loss = 2*tf.nn.l2_loss(x - completion)
        print('loss :'+str(loss.get_shape().as_list()))
        return tf.reduce_mean(loss)

    def calc_dfake_loss(self, fake):
        dfake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))
        return dfake_loss


    def calc_d_loss(self, real, fake):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake)

