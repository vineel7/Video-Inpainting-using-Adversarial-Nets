#!/usr/bin/python
# -*- coding: utf-8 -*-
from layer import *
import numpy as np
from image_warp import *

image_width = 3
image_size = 128
local_size = 64
channels = 3


class Network:

    def __init__(
        self,
        x,
        mask,
        points,
        local_x,
        is_training,
        batch_size,
        ):
        self.alpha = 1
        self.batch_size = batch_size
        self.mc_variables = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope='mc')

# ================ completion part ============================

        (self.completion_3d, self.imitation_3d) = \
            self.completionnetwork_3d(x, mask, is_training)

    # print('generator output shape -- ')
        # print(self.imitation_3d.get_shape().as_list())

        self.local_completionG = \
            self.local_completion(self.completion_3d, points)

    # print('shape of local completion output')
    # print(self.local_completionG.get_shape().as_list())

# =================  MC part ===========================================================================

        self.completion_3d_mc = tf.transpose(self.completion_3d, [0, 2,
                3, 4, 1])

    # print('shape after mc for inpainted--'+str(self.completion_3d_mc.get_shape().as_list()))

        (self.flow1, self.completion_3d_mc) = \
            self.model_mc(self.completion_3d_mc, reuse=False)

    # print('shape after mc for inpainted--'+str(self.completion_3d_mc.get_shape().as_list()))

        self.completion_3d_mc = \
            tf.concat([tf.concat([self.completion_3d_mc[0:batch_size, :
                      , :, :], x[:, 2, :, :, :]], axis=0),
                      self.completion_3d_mc[batch_size:2 * batch_size, :
                      , :, :]], axis=0)

    # print('shape after adding middle'+str(self.completion_3d_mc.get_shape().as_list()))

        self.completion_3d_mc_reshape = []
        for i in range(batch_size):
            t1 = self.completion_3d_mc[i, :, :, :]
            t2 = self.completion_3d_mc[i + batch_size, :, :, :]
            t3 = self.completion_3d_mc[i + 2 * batch_size, :, :, :]
            t4 = tf.stack([t1, t2, t3], axis=0)

        # print(t4.get_shape().as_list())

            self.completion_3d_mc_reshape.append(t4)
        self.completion_3d_mc_reshape = \
            tf.convert_to_tensor(self.completion_3d_mc_reshape)

    # print('shape after reshaping inpainted--'+str(self.completion_3d_mc_reshape.get_shape().as_list()))
    # print('=========================================')

        self.local_completionG_mc = \
            tf.transpose(self.local_completionG, [0, 2, 3, 4, 1])

    # print('input shape to mc for local inpainted '+str(self.local_completionG_mc.get_shape().as_list()))

        (self.flow2, self.local_completionG_mc) = \
            self.model_mc(self.local_completionG_mc, reuse=True)

    # print('shape after mc for local inpainted '+str(self.local_completionG_mc.get_shape().as_list()))

        self.local_completionG_mc = \
            tf.concat([tf.concat([self.local_completionG_mc[0:
                      batch_size, :, :, :], local_x[:, 1, :, :, :]],
                      axis=0), self.local_completionG_mc[batch_size:2
                      * batch_size, :, :, :]], axis=0)

    # print('shape after adding middle '+str(self.local_completionG_mc.get_shape().as_list()))

        self.local_completionG_mc_reshape = []
        for i in range(batch_size):
            t1 = self.local_completionG_mc[i, :, :, :]
            t2 = self.local_completionG_mc[i + batch_size, :, :, :]
            t3 = self.local_completionG_mc[i + 2 * batch_size, :, :, :]
            t4 = tf.stack([t1, t2, t3], axis=0)

                # print(t4.get_shape().as_list())

            self.local_completionG_mc_reshape.append(t4)
        self.local_completionG_mc_reshape = \
            tf.convert_to_tensor(self.local_completionG_mc_reshape)

        # print('shape after reshaping '+str(self.local_completionG_mc_reshape.get_shape().as_list()))
    # print('========================================')

        self.real_glo_dis_mc = tf.transpose(x[:, 1:4, :, :, :], [0, 2,
                3, 4, 1])

    # print('input to mc for real x--'+str(self.real_glo_dis_mc.get_shape().as_list()))

        (self.flow3, self.real_glo_dis_mc) = \
            self.model_mc(self.real_glo_dis_mc, reuse=True)

    # print('output after mc for real x --'+str(self.real_glo_dis_mc.get_shape().as_list()))

        self.real_glo_dis_mc = \
            tf.concat([tf.concat([self.real_glo_dis_mc[0:batch_size, :,
                      :, :], x[:, 2, :, :, :]], axis=0),
                      self.real_glo_dis_mc[batch_size:2 * batch_size, :
                      , :, :]], axis=0)

    # print('shape after adding middle--'+str(self.real_glo_dis_mc.get_shape().as_list()))

        self.real_glo_dis_mc_reshape = []
        for i in range(batch_size):
            t1 = self.real_glo_dis_mc[i, :, :, :]
            t2 = self.real_glo_dis_mc[i + batch_size, :, :, :]
            t3 = self.real_glo_dis_mc[i + 2 * batch_size, :, :, :]
            t4 = tf.stack([t1, t2, t3], axis=0)

                # print(t4.get_shape().as_list())

            self.real_glo_dis_mc_reshape.append(t4)
        self.real_glo_dis_mc_reshape = \
            tf.convert_to_tensor(self.real_glo_dis_mc_reshape)

        # print('shape after reshaping--'+str(self.real_glo_dis_mc_reshape.get_shape().as_list()))
    # print('=======================')

        self.real_loc_dis_mc = tf.transpose(local_x, [0, 2, 3, 4, 1])

    # print('input shape to mc for local real--'+str(self.real_loc_dis_mc.get_shape().as_list()))

        (self.flow4, self.real_loc_dis_mc) = \
            self.model_mc(self.real_loc_dis_mc, reuse=True)

    # print('output after mc for local real--'+str(self.real_loc_dis_mc.get_shape().as_list()))

        self.real_loc_dis_mc = \
            tf.concat([tf.concat([self.real_loc_dis_mc[0:batch_size, :,
                      :, :], local_x[:, 1, :, :, :]], axis=0),
                      self.real_loc_dis_mc[batch_size:2 * batch_size, :
                      , :, :]], axis=0)

    # print('shape after adding middle--'+str(self.real_loc_dis_mc.get_shape().as_list()))

        self.real_loc_dis_mc_reshape = []
        for i in range(batch_size):
            t1 = self.real_loc_dis_mc[i, :, :, :]
            t2 = self.real_loc_dis_mc[i + batch_size, :, :, :]
            t3 = self.real_loc_dis_mc[i + 2 * batch_size, :, :, :]
            t4 = tf.stack([t1, t2, t3], axis=0)

                # print(t4.get_shape().as_list())

            self.real_loc_dis_mc_reshape.append(t4)
        self.real_loc_dis_mc_reshape = \
            tf.convert_to_tensor(self.real_loc_dis_mc_reshape)

        # print('shape after reshaping--'+str(self.real_loc_dis_mc_reshape.get_shape().as_list()))

# ============ discriminator part =================================

        self.real = self.discriminator(self.real_glo_dis_mc_reshape,
                self.real_loc_dis_mc_reshape, reuse=False)
        self.fake = self.discriminator(self.completion_3d_mc_reshape,
                self.local_completionG_mc_reshape, reuse=True)
        self.g_loss = self.calc_g_loss(x[:, 1:4, :, :, :],
                self.completion_3d) / (3 * self.batch_size)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.dfake_loss = self.calc_dfake_loss(self.fake)
        self.combined_loss = self.g_loss + self.alpha * self.dfake_loss
        self.g_variables = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope='generator')
        self.d_variables = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope='discriminator')

    def completionnetwork_3d(
        self,
        x,
        mask,
        is_training,
        ):
        self.imitation_3d = []
        self.completion_3d = []
        for i in range(3):
            self.x_3d = x[:, i:i + 3, :, :, :]

                # print("type is :", type(self.x_3d))

            self.mask_3d = mask[:, i:i + 3, :, :, :]

                # print('x_3d shape')
        # print(self.x_3d.get_shape().as_list())
        # print('mask_3d shape')
        # print(self.mask_3d.get_shape().as_list())

            if i == 0:
                self.imitation = self.generator(self.x_3d * (1
                        - self.mask_3d), is_training, reuse=False)
            else:
                self.imitation = self.generator(self.x_3d * (1
                        - self.mask_3d), is_training, reuse=True)
            self.completion = self.imitation * self.mask_3d[:, 1, :, :,
                    :] + self.x_3d[:, 1, :, :, :] * (1 - self.mask_3d[:
                    , 1, :, :, :])
            self.imitation_3d.append(self.imitation)
            self.completion_3d.append(self.completion)
        self.completion_3d = tf.convert_to_tensor(self.completion_3d)
        self.completion_3d = tf.transpose(self.completion_3d, [1, 0, 2,
                3, 4])
        self.imitation_3d = tf.convert_to_tensor(self.imitation_3d)
        self.imitation_3d = tf.transpose(self.imitation_3d, [1, 0, 2,
                3, 4])
        return (self.completion_3d, self.imitation_3d)

    def local_completion(self, completion_3d, points):
        self.local_completionG = []
        for j in range(self.batch_size):
            for k in range(3):
                cord = points[j][k]
                self.cropped = \
                    tf.image.crop_to_bounding_box(self.completion_3d[j,
                        k, :, :, :], cord[1], cord[0], cord[3]
                        - cord[1], cord[2] - cord[0])
                self.local_completionG.append(self.cropped)

        self.local_completionG = \
            tf.convert_to_tensor(self.local_completionG)
        self.local_completionG = tf.reshape(self.local_completionG,
                [self.batch_size, 3, 64, 64, 3])

        return self.local_completionG

    def generator(
        self,
        x,
        is_training,
        reuse,
        ):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('conv1'):

        # print('gen layer1 input shape -- ')
                # print(x.get_shape().as_list())

                x = conv3_layer(x, [3, 5, 5, 3, 64], 1)
                x = batch_normalize_3d(x, is_training)
                x = tf.nn.relu(x)
                x = tf.squeeze(x, axis=[1])

        # print('gen layer1 out shape')
                # print(x.get_shape().as_list())

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
                x = deconv_layer(x, [4, 4, 128, 256], [self.batch_size,
                                 64, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv9'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [4, 4, 64, 128], [self.batch_size,
                                 128, 128, 64], 2)
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

    def discriminator(
        self,
        global_x,
        local_x,
        reuse,
        ):

        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('conv1'):

            # print('glo dis input shape')
                    # print(x.get_shape().as_list())

                    x = conv3_layer(x, [3, 5, 5, 3, 64], 2)
                    x = batch_normalize_3d(x, is_training)
                    x = tf.nn.relu(x)
                    x = tf.squeeze(x, axis=[1])

            # print('shape of global dis 1 layer out')
                    # print(x.get_shape().as_list())

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

            # print('loc dis input shape')
            # print(x.get_shape().as_list())

                    x = conv3_layer(x, [3, 5, 5, 3, 64], 2)
                    x = batch_normalize_3d(x, is_training)
                    x = tf.nn.relu(x)
                    x = tf.squeeze(x, axis=[1])

            # print('shape of local dis 1 layer out')
            # print(x.get_shape().as_list())

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
        loss = 2 * tf.nn.l2_loss(x - completion)
        return tf.reduce_mean(loss)

    def calc_dfake_loss(self, fake):
        dfake_loss = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake,
                           labels=tf.ones_like(fake)))
        return dfake_loss

    def calc_d_loss(self, real, fake):
        d_loss_real = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real,
                           labels=tf.ones_like(real)))
        d_loss_fake = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake,
                           labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake)

    def model_mc(self, data_batch, reuse):

        if True:
            with tf.variable_scope('mc', reuse=reuse):

                # neighboring_frames = tf.expand_dims(tf.concat([data_batch[:,:, :, :, 0], data_batch[:,:, :, :, 2]],axis=0), axis=3)

                neighboring_frames = tf.concat([data_batch[:, :, :, :,
                        0], data_batch[:, :, :, :, 2]], axis=0)

                # print('neighboring frame shape')
                # print(neighboring_frames.get_shape().as_list())

                lr_input = tf.concat([tf.concat([data_batch[:, :, :, :,
                        1], data_batch[:, :, :, :, 0]], axis=3),
                        tf.concat([data_batch[:, :, :, :, 1],
                        data_batch[:, :, :, :, 2]], axis=3)], axis=0)

                # print('input shape')
                # print(lr_input.get_shape().as_list())

                with tf.variable_scope('coarse_flow'):
                    net = tf.layers.conv2d(
                        lr_input,
                        24,
                        5,
                        strides=2,
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv1',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    net = tf.layers.conv2d(
                        net,
                        24,
                        3,
                        strides=1,
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv2',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    net = tf.layers.conv2d(
                        net,
                        24,
                        3,
                        strides=2,
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv3',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    net = tf.layers.conv2d(
                        net,
                        24,
                        3,
                        strides=1,
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv4',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    net = tf.layers.conv2d(
                        net,
                        32,
                        3,
                        strides=1,
                        activation=tf.nn.tanh,
                        padding='same',
                        name='conv5',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    coarse_flow = tf.depth_to_space(net, 4)  # check

                    # print('coarse flow output shape')
                    # print(coarse_flow.get_shape().as_list())

                    warped_frames = image_warp(neighboring_frames,
                            coarse_flow)

                    # print('warped frames after course flow output shape')
                    # print(warped_frames.get_shape().as_list())

                ff_input = tf.concat([lr_input, coarse_flow,
                        warped_frames], axis=3)
                with tf.variable_scope('fine_flow'):
                    net = tf.layers.conv2d(
                        ff_input,
                        24,
                        5,
                        strides=2,
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv1',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    net = tf.layers.conv2d(
                        net,
                        24,
                        3,
                        strides=1,
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv2',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    net = tf.layers.conv2d(
                        net,
                        24,
                        3,
                        strides=1,
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv3',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    net = tf.layers.conv2d(
                        net,
                        24,
                        3,
                        strides=1,
                        activation=tf.nn.relu,
                        padding='same',
                        name='conv4',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    net = tf.layers.conv2d(
                        net,
                        8,
                        3,
                        strides=1,
                        activation=tf.nn.tanh,
                        padding='same',
                        name='conv5',
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        )
                    fine_flow = tf.depth_to_space(net, 2)  # check

                    # print('fine flow output shape')
                    # print(fine_flow.get_shape().as_list())

                    flow = coarse_flow + fine_flow

                    # print('final flow output shape')
                    # print(flow.get_shape().as_list())

                    warped_frames = image_warp(neighboring_frames, flow)

                    # print('final mc warped frames output shape')
                    # print(warped_frames.get_shape().as_list())

                    return (flow, warped_frames)
        else:
            flow = []
            warped_frames = []
            sr_input = data_batch[0]

    def get_loss(
        self,
        data_batch,
        flow,
        warped_frames,
        ):

        cur_frames = tf.concat([data_batch[:, :, :, :, 1], data_batch[:
                               , :, :, :, 1]], axis=0)

        # print('get loss---- shape of cur frame')
        # print(cur_frames.get_shape().as_list())
        # warp_loss = tf.losses.mean_squared_error(cur_frames, warped_frames)

        warp_loss = 2 * tf.nn.l2_loss(cur_frames - warped_frames)
        warp_loss = tf.reduce_mean(warp_loss)

        grad_x_kernel = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32,
                                    shape=(1, 3, 1, 1))
        grad_y_kernel = tf.constant([-1.0, 0, 1.0], dtype=tf.float32,
                                    shape=(3, 1, 1, 1))
        flow = tf.expand_dims(tf.concat([flow[:, :, :, 0], flow[:, :, :
                              , 1]], axis=0), axis=3)
        flow_grad_x = tf.nn.conv2d(flow, grad_x_kernel, [1, 1, 1, 1],
                                   padding='VALID')[:, 1:-1, :, :]
        flow_grad_y = tf.nn.conv2d(flow, grad_y_kernel, [1, 1, 1, 1],
                                   padding='VALID')[:, :, 1:-1, :]
        huber_loss = tf.sqrt(0.01 + tf.reduce_sum(flow_grad_x
                             * flow_grad_x + flow_grad_y * flow_grad_y))

        tf.summary.scalar('Warp_loss', warp_loss)
        tf.summary.scalar('Huber_loss', huber_loss)

        return warp_loss + 0.01 * huber_loss
