from layer import *

class Network:
    def __init__(self, x, mask, local_x, global_completion, local_completion, is_training, batch_size):
        self.alpha = 10
        self.batch_size = batch_size
        self.chksiz = x*(1-mask)
        self.imitation = self.generator(self.chksiz, is_training)
        #print('xshape')
        #print(x.get_shape().as_list())
        #print('maskshape')
        #print(mask.get_shape().as_list())
        print('generatorshape')
        print(self.imitation.get_shape().as_list())
        #print(self.chksiz.get_shape().as_list())
        self.completion = self.imitation * mask[:, 1, :, :, :] + x[:, 1, :, :, :] * (1 - mask[:, 1, :, :, :])
        self.real = self.discriminator(x[:, 1, :, :, :], local_x, reuse=False)
        self.fake = self.discriminator(global_completion, local_completion, reuse=True)
        self.g_loss = self.calc_g_loss(x[:, 1, :, :, :], self.completion)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.dfake_loss = self.calc_dfake_loss(self.fake)
        self.combined_loss = self.g_loss + self.alpha * self.dfake_loss
        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')


    def generator(self, x, is_training):
        with tf.variable_scope('generator'):
            with tf.variable_scope('conv1'):
                x = conv3_layer(x, [3, 5, 5, 3, 64], 1)
                print('shape after conv1')
                print(x.get_shape().as_list())
                x = batch_normalize_3d(x, is_training)
                x = tf.nn.relu(x)
                x = tf.squeeze(x,axis=[1])
                print('shape after layer1')
                print(x.get_shape().as_list())
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 128], 2)
                #print('shape after conv2')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer2')
                #print(x.get_shape().as_list())
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                #print('shape after conv3')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer3')
                #print(x.get_shape().as_list())
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 256], 2)
                #print('shape after conv4')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer4')
                #print(x.get_shape().as_list())
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                #print('shape after conv5')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer5')
                #print(x.get_shape().as_list())
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                #print('shape after conv6')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer6')
                #print(x.get_shape().as_list())
            with tf.variable_scope('dilated1'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 2)
                #print('shape after dilated1')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer7')
                #print(x.get_shape().as_list())
            with tf.variable_scope('dilated2'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 4)
                #print('shape after dilated2')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer8')
                #print(x.get_shape().as_list())
            with tf.variable_scope('dilated3'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 8)
                #print('shape after dilated3')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer9')
                #print(x.get_shape().as_list())
            with tf.variable_scope('dilated4'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 16)
                #print('shape after dilated4')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer10')
                #print(x.get_shape().as_list())
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                #print('shape after conv7')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer11')
                #print(x.get_shape().as_list())
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                #print('shape after conv8')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer12')
                #print(x.get_shape().as_list())
            with tf.variable_scope('deconv1'):
                x = deconv_layer(x, [4, 4, 128, 256], [self.batch_size, 64, 64, 128], 2)
                #print('shape after deconv1')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer13')
                #print(x.get_shape().as_list())
            with tf.variable_scope('conv9'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                #print('shape after conv9')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer14')
                #print(x.get_shape().as_list())
            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [4, 4, 64, 128], [self.batch_size, 128, 128, 64], 2)
                #print('shape after deconv2')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer15')
                #print(x.get_shape().as_list())
            with tf.variable_scope('conv10'):
                x = conv_layer(x, [3, 3, 64, 32], 1)
                #print('shape after conv10')
                #print(x.get_shape().as_list())
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                #print('shape after layer16')
                #print(x.get_shape().as_list())
            with tf.variable_scope('conv11'):
                x = conv_layer(x, [3, 3, 32, 3], 1)
                #print('shape after conv11')
                #print(x.get_shape().as_list())
                x = tf.nn.tanh(x)
                #print('shape after layer17')
                #print(x.get_shape().as_list())

        return x


    def discriminator(self, global_x, local_x, reuse):
        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
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
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
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
        loss = tf.nn.l2_loss(x - completion)
        return tf.reduce_mean(loss)

    def calc_dfake_loss(self, fake):
        dfake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))
        return dfake_loss


    def calc_d_loss(self, real, fake):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake)

