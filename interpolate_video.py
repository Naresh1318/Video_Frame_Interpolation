import numpy as np
import tensorflow as tf
import mission_control as mc
import ops
import utils

input_frames = tf.placeholder(dtype=tf.float32, shape=[None, 288, 352, 6], name="Input_Frames")
target_frame = tf.placeholder(dtype=tf.float32, shape=[None, 288, 352, 3], name="Target_Frame")


def generator(x, reuse=False):
    # TODO: Add dropout??
    if reuse:
        tf.get_variable_scope().reuse_variables()
    # Encoder
    conv_b_1 = ops.conv_block(x, filter_size=3, stride_length=2, n_maps=32, name='g_conv_b_1')
    conv_b_2 = ops.conv_block(conv_b_1, filter_size=3, stride_length=2, n_maps=64, name='g_conv_b_2')
    conv_b_3 = ops.conv_block(conv_b_2, filter_size=3, stride_length=2, n_maps=64, name='g_conv_b_3')
    conv_b_4 = ops.conv_block(conv_b_3, filter_size=3, stride_length=2, n_maps=128, name='g_conv_b_4')

    # Decoder
    conv_tb_1 = ops.conv_t_block(conv_b_4, filter_size=4, stride_length=2, n_maps=128,
                                 output_shape= [mc.batch_size, conv_b_4.get_shape()[1].value*2, conv_b_4.get_shape()[2].value*2, 128], name='g_conv_tb_1')
    conv_tb_1 = tf.concat([conv_tb_1, conv_b_3], axis=3)
    conv_tb_2 = ops.conv_t_block(conv_tb_1, filter_size=4, stride_length=2, n_maps=64,
                                 output_shape=[mc.batch_size, conv_tb_1.get_shape()[1].value*2, conv_tb_1.get_shape()[2].value*2, 64], name='g_conv_tb_2')
    conv_tb_2 = tf.concat([conv_tb_2, conv_b_2], axis=3)
    conv_tb_3 = ops.conv_t_block(conv_tb_2, filter_size=4, stride_length=2, n_maps=64,
                                 output_shape=[mc.batch_size, conv_tb_2.get_shape()[1].value*2, conv_tb_2.get_shape()[2].value*2, 64], name='g_conv_tb_3')
    conv_tb_3 = tf.concat([conv_tb_3, conv_b_1], axis=3)
    conv_tb_4 = ops.conv_t_block(conv_tb_3, filter_size=4, stride_length=2, n_maps=32,
                                 output_shape=[mc.batch_size, conv_tb_3.get_shape()[1].value*2, conv_tb_3.get_shape()[2].value*2, 32], name='g_conv_tb_4')

    output = ops.cnn_2d_trans(conv_tb_4, weight_shape=[4, 4, 3, conv_tb_4.get_shape()[-1].value], strides=[1, 1, 1, 1],
                              output_shape=[mc.batch_size, conv_tb_4.get_shape()[1].value, conv_tb_4.get_shape()[2].value, 3], name='g_output')
    return output


def discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv_b_1 = ops.conv_block(x, filter_size=4, stride_length=2, n_maps=8, name='d_conv_b_1')
    conv_b_2 = ops.conv_block(conv_b_1, filter_size=4, stride_length=2, n_maps=16, name='d_conv_b_2')
    conv_b_3 = ops.conv_block(conv_b_2, filter_size=4, stride_length=2, n_maps=32, name='d_conv_b_3')
    conv_b_4 = ops.conv_block(conv_b_3, filter_size=4, stride_length=2, n_maps=64, name='d_conv_b_4')
    conv_b_5 = ops.conv_block(conv_b_4, filter_size=4, stride_length=2, n_maps=1, name='d_conv_b_5')
    conv_b_5_r = tf.reshape(conv_b_5, [-1, 11*9*1], name='d_reshape')
    output = ops.dense(conv_b_5_r, 11*9, 1, name='d_output')
    return output


def train():
    with tf.variable_scope(tf.get_variable_scope()):
        predicted_frame = generator(input_frames)

    discriminator_real_input = tf.concat([input_frames, target_frame], axis=2)
    discriminator_fake_input = tf.concat([input_frames, predicted_frame], axis=2)
    with tf.variable_scope(tf.get_variable_scope()):
        real_discriminator_op = discriminator(discriminator_real_input)
        fake_discriminator_op = discriminator(discriminator_fake_input, reuse=True)

    # GAN loss
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_discriminator_op), logits=fake_discriminator_op))
    discriminator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_discriminator_op), logits=fake_discriminator_op))
    discriminator_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_discriminator_op), logits=real_discriminator_op))
    discriminator_loss = discriminator_fake_loss + discriminator_real_loss

    # Collect trainable parameter
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    generator_optimizer = tf.train.AdamOptimizer().minimize(generator_loss, var_list=g_vars)
    discriminator_optimizer = tf.train.AdamOptimizer().minimize(discriminator_loss, var_list=d_vars)

    l2_loss = tf.reduce_mean(tf.square(predicted_frame - target_frame))
    optimizer = tf.train.AdamOptimizer().minimize(l2_loss)
    tf.summary.scalar('l2_loss', tensor=l2_loss)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        file_writer = tf.summary.FileWriter(logdir='./Tensorboard', graph=sess.graph)
        train_data, train_target, test_data, test_target = utils.generate_dataset_from_video('./Dataset/videos/bus_cif.y4m')
        for e in range(mc.n_epochs):
            n_batches = int(len(train_data)/mc.batch_size)
            for b in range(n_batches):
                batch_indx = np.random.permutation(len(train_data))[:mc.batch_size]
                train_data_batch = [train_data[t] for t in batch_indx]
                train_target_batch = [train_target[t] for t in batch_indx]
                _, l, s = sess.run([optimizer, l2_loss, summary_op], feed_dict={input_frames: train_data_batch, target_frame: train_target_batch})
                print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, mc.n_epochs, b, n_batches, l))
                file_writer.add_summary(s)


if __name__ == '__main__':
    train()
