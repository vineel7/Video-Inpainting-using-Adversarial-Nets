from layer import *
import tensorflow as tf


def generator(x, is_training):
        with tf.variable_scope('generator'):
            with tf.variable_scope('conv1'):
                x = conv3_layer(x, [3, 5, 5, 3, 64], 1)
                x = batch_normalize_3d(x, is_training)
                x = tf.nn.relu(x)
                x = tf.squeeze(x,axis=[1])
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)

        return x



g=tf.Graph()
run_meta=tf.RunMetadata()
with g.as_default():
     x=tf.Variable(tf.random_normal([1,3,128,128,3]))
     y=generator(x,tf.constant(False));
     opts=tf.profiler.ProfileOptionBuilder.float_operation()
     pats=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
     flops=tf.profiler.profile(g,options=opts)
     param=tf.profiler.profile(g, options=pats)
     if flops is not None:
         print(flops.total_float_ops)
     if param is not None:
         print(param.total_parameters)

