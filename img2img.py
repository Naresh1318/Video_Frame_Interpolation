import numpy as np
import tensorflow as tf
import mission_control as mc
import ops
import utils


# Placeholders
input_image = tf.placeholder(dtype=tf.float32, shape=[None, 288, 352, 3], name='Input_image')
target_image = tf.placeholder(dtype=tf.float32, shape=[None, 288, 352, 3], name='Target_image')
global_step = tf.placeholder(dtype=tf.int64, shape=[], name="Global_Step")


def generator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # Encoder
    conv_1 = ops.lrelu(ops.cnn_2d(x, weight_shape=[4, 4, 3, 64], strides=[1, 2, 2, 1], name='g_e_conv_1'))
    conv_2 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_1, weight_shape=[4, 4, 64, 128],
                                                 strides=[1, 2, 2, 1], name='g_e_conv_2'),
                                      center=True, scale=True, is_training=True, scope='g_e_batch_Norm_2'))
    conv_3 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_2, weight_shape=[4, 4, 128, 256],
                                                 strides=[1, 2, 2, 1], name='g_e_conv_3'),
                                      center=True, scale=True, is_training=True, scope='g_e_batch_Norm_3'))
    conv_4 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_3, weight_shape=[4, 4, 256, 512],
                                                 strides=[1, 2, 2, 1], name='g_e_conv_4'),
                                      center=True, scale=True, is_training=True, scope='g_e_batch_Norm_4'))
    conv_5 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_4, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='g_e_conv_5'),
                                      center=True, scale=True, is_training=True, scope='g_e_batch_Norm_5'))
    conv_6 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_5, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='g_e_conv_6'),
                                      center=True, scale=True, is_training=True, scope='g_e_batch_Norm_6'))
    conv_7 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_6, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='g_e_conv_7'),
                                      center=True, scale=True, is_training=True, scope='g_e_batch_Norm_7'))
    conv_8 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_7, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='g_e_conv_8'),
                                      center=True, scale=True, is_training=True, scope='g_e_batch_Norm_8'))

    # Decoder
    dconv_1 = ops.lrelu(tf.nn.dropout(ops.batch_norm(ops.cnn_2d_trans(conv_8, weight_shape=[4, 4, 512, 512], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, conv_8.get_shape()[1].value*2, conv_8.get_shape()[2].value*2, 512], name='g_d_dconv_1'), center=True, scale=True, is_training=True, scope='g_d_batch_Norm_1'), keep_prob=0.5))
    dconv_1 = tf.concat([dconv_1, conv_7], axis=3)
    dconv_2 = ops.lrelu(tf.nn.dropout(ops.batch_norm(ops.cnn_2d_trans(dconv_1, weight_shape=[4, 4, 512, 1024], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_1.get_shape()[1].value*2, (dconv_1.get_shape()[2].value+1)*2, 512], name='g_d_dconv_2'), center=True, scale=True, is_training=True, scope='g_d_batch_Norm_2'), keep_prob=0.5))
    dconv_2 = tf.concat([dconv_2, conv_6], axis=3)
    dconv_3 = ops.lrelu(tf.nn.dropout(ops.batch_norm(ops.cnn_2d_trans(dconv_2, weight_shape=[4, 4, 512, 1024], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, (dconv_2.get_shape()[1].value+1)*2, (dconv_2.get_shape()[2].value+1)*2, 512], name='g_d_dconv_3'), center=True, scale=True, is_training=True, scope='g_d_batch_Norm_3'), keep_prob=0.5))
    dconv_3 = tf.concat([dconv_3, conv_5], axis=3)
    dconv_4 = ops.lrelu(ops.batch_norm(ops.cnn_2d_trans(dconv_3, weight_shape=[4, 4, 512, 1024], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_3.get_shape()[1].value*2, dconv_3.get_shape()[2].value*2, 512], name='g_d_dconv_4'), center=True, scale=True, is_training=True, scope='g_d_batch_Norm_4'))
    dconv_4 = tf.concat([dconv_4, conv_4], axis=3)
    dconv_5 = ops.lrelu(ops.batch_norm(ops.cnn_2d_trans(dconv_4, weight_shape=[4, 4, 256, 1024], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_4.get_shape()[1].value*2, dconv_4.get_shape()[2].value*2, 256], name='g_d_dconv_5'), center=True, scale=True, is_training=True, scope='g_d_batch_Norm_5'))
    dconv_5 = tf.concat([dconv_5, conv_3], axis=3)
    dconv_6 = ops.lrelu(ops.batch_norm(ops.cnn_2d_trans(dconv_5, weight_shape=[4, 4, 128, 512], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_5.get_shape()[1].value*2, dconv_5.get_shape()[2].value*2, 128], name='g_d_dconv_6'), center=True, scale=True, is_training=True, scope='g_d_batch_Norm_6'))
    dconv_6 = tf.concat([dconv_6, conv_2], axis=3)
    dconv_7 = ops.lrelu(ops.batch_norm(ops.cnn_2d_trans(dconv_6, weight_shape=[4, 4, 64, 256], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_6.get_shape()[1].value*2, dconv_6.get_shape()[2].value*2, 64], name='g_d_dconv_7'), center=True, scale=True, is_training=True, scope='g_d_batch_Norm_7'))
    dconv_7 = tf.concat([dconv_7, conv_1], axis=3)
    dconv_8 = tf.nn.tanh(ops.cnn_2d_trans(dconv_7, weight_shape=[4, 4, 3, 128], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_7.get_shape()[1].value*2, dconv_7.get_shape()[2].value*2, 3], name='g_d_dconv_8'))
    return dconv_8


def discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    conv_1 = ops.lrelu(ops.batch_norm(ops.cnn_2d(x, weight_shape=[4, 4, 6, 64],
                                                 strides=[1, 2, 2, 1], name='dis_conv_1'),
                                      center=True, scale=True, is_training=True, scope='dis_batch_Norm_1'))
    conv_2 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_1, weight_shape=[4, 4, 64, 128],
                                                 strides=[1, 2, 2, 1], name='dis_conv_2'),
                                      center=True, scale=True, is_training=True, scope='dis_batch_Norm_2'))
    conv_3 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_2, weight_shape=[4, 4, 64, 256],
                                                 strides=[1, 2, 2, 1], name='dis_conv_3'),
                                      center=True, scale=True, is_training=True, scope='dis_batch_Norm_3'))
    conv_4 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_3, weight_shape=[4, 4, 256, 512],
                                                 strides=[1, 2, 2, 1], name='dis_conv_4'),
                                      center=True, scale=True, is_training=True, scope='dis_batch_Norm_4'))
    conv_5 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_4, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='dis_conv_5'),
                                      center=True, scale=True, is_training=True, scope='dis_batch_Norm_5'))
    conv_6 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_5, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='dis_conv_6'),
                                      center=True, scale=True, is_training=True, scope='dis_batch_Norm_6'))
    output = ops.dense(conv_6, 4 * 5, 1, name='dis_output')
    return output


def train():
    with tf.variable_scope(tf.get_variable_scope()):
        generated_image = generator(input_image)

    discriminator_real_input = tf.concat([input_image, target_image], axis=3)
    discriminator_fake_input = tf.concat([input_image, generated_image], axis=3)

    with tf.variable_scope(tf.get_variable_scope()):
        real_discriminator_op = discriminator(discriminator_real_input)
        fake_discriminator_op = discriminator(discriminator_fake_input, reuse=True)

    # GAN losses
    generator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                         (labels=tf.ones_like(fake_discriminator_op), logits=fake_discriminator_op))
    discriminator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                             (labels=tf.zeros_like(fake_discriminator_op),
                                              logits=fake_discriminator_op))
    discriminator_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                             (labels=tf.ones_like(real_discriminator_op), logits=real_discriminator_op))

    eps = 1e-5
    l1_loss = tf.reduce_mean(tf.abs(generated_image - target_image + eps))

    discriminator_loss = discriminator_fake_loss + discriminator_real_loss

    generator_loss = mc.discriminator_weight * generator_fake_loss + mc.l1_weight * l1_loss

    # Collect trainable parameter
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    g_learning_rate = tf.train.exponential_decay(mc.generator_lr, global_step,
                                                 1, 0.96, staircase=True)
    d_learning_rate = tf.train.exponential_decay(mc.discriminator_lr, global_step,
                                                 1, 0.96, staircase=True)

    generator_optimizer = tf.train.AdamOptimizer(g_learning_rate, beta1=mc.beta1).minimize(generator_loss,
                                                                                           var_list=g_vars)
    discriminator_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=mc.beta1).minimize(discriminator_loss,
                                                                                               var_list=d_vars)





