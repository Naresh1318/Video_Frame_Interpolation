import numpy as np
import tensorflow as tf
import mission_control as mc
import ops
import utils
import sys

input_frames = tf.placeholder(dtype=tf.float32, shape=[None, 288, 352, 6], name="Input_Frames")
target_frame = tf.placeholder(dtype=tf.float32, shape=[None, 288, 352, 3], name="Target_Frame")
global_step = tf.placeholder(dtype=tf.int64, shape=[], name="Global_Step")

train_data, train_target, test_data, test_target, mean_img = utils.generate_dataset_from_video(mc.video_path)


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
                                 output_shape=[mc.batch_size, conv_b_4.get_shape()[1].value * 2,
                                               conv_b_4.get_shape()[2].value * 2, 128], name='g_conv_tb_1')
    conv_tb_1 = tf.concat([conv_tb_1, conv_b_3], axis=3)
    conv_tb_2 = ops.conv_t_block(conv_tb_1, filter_size=4, stride_length=2, n_maps=64,
                                 output_shape=[mc.batch_size, conv_tb_1.get_shape()[1].value * 2,
                                               conv_tb_1.get_shape()[2].value * 2, 64], name='g_conv_tb_2')
    conv_tb_2 = tf.concat([conv_tb_2, conv_b_2], axis=3)
    conv_tb_3 = ops.conv_t_block(conv_tb_2, filter_size=4, stride_length=2, n_maps=64,
                                 output_shape=[mc.batch_size, conv_tb_2.get_shape()[1].value * 2,
                                               conv_tb_2.get_shape()[2].value * 2, 64], name='g_conv_tb_3')
    conv_tb_3 = tf.concat([conv_tb_3, conv_b_1], axis=3)
    conv_tb_4 = ops.conv_t_block(conv_tb_3, filter_size=4, stride_length=2, n_maps=32,
                                 output_shape=[mc.batch_size, conv_tb_3.get_shape()[1].value * 2,
                                               conv_tb_3.get_shape()[2].value * 2, 32], name='g_conv_tb_4')

    output = ops.cnn_2d_trans(conv_tb_4, weight_shape=[4, 4, 3, conv_tb_4.get_shape()[-1].value], strides=[1, 1, 1, 1],
                              output_shape=[mc.batch_size, conv_tb_4.get_shape()[1].value,
                                            conv_tb_4.get_shape()[2].value, 3], name='g_output')
    return output


def discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv_b_1 = ops.conv_block(x, filter_size=4, stride_length=2, n_maps=8, name='d_conv_b_1')
    conv_b_2 = ops.conv_block(conv_b_1, filter_size=4, stride_length=2, n_maps=16, name='d_conv_b_2')
    conv_b_3 = ops.conv_block(conv_b_2, filter_size=4, stride_length=2, n_maps=32, name='d_conv_b_3')
    conv_b_4 = ops.conv_block(conv_b_3, filter_size=4, stride_length=2, n_maps=64, name='d_conv_b_4')
    conv_b_5 = ops.conv_block(conv_b_4, filter_size=4, stride_length=2, n_maps=1, name='d_conv_b_5')
    conv_b_5_r = tf.reshape(conv_b_5, [-1, 11 * 9 * 1], name='d_reshape')
    output = ops.dense(conv_b_5_r, 11 * 9, 1, name='d_output')
    return output


def train():
    with tf.variable_scope(tf.get_variable_scope()):
        predicted_frame = generator(input_frames)

    discriminator_real_input = tf.concat([input_frames, target_frame], axis=3)
    discriminator_fake_input = tf.concat([input_frames, predicted_frame], axis=3)
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
    l1_loss = tf.reduce_mean(tf.abs(predicted_frame - target_frame + eps))

    predicted_frame_mean_added = predicted_frame + mean_img
    predicted_frame_mean_added_clipped = tf.clip_by_value(predicted_frame_mean_added, 0, 1)
    target_frame_mean_added_clipped = tf.clip_by_value(target_frame + mean_img, 0, 1)

    clipping_loss = tf.reduce_mean(tf.square(predicted_frame_mean_added_clipped - predicted_frame_mean_added))
    ms_ssim_loss = tf.reduce_mean(
        -tf.log(utils.tf_ms_ssim(predicted_frame_mean_added_clipped, target_frame_mean_added_clipped)))

    discriminator_loss = discriminator_fake_loss + discriminator_real_loss
    generator_loss = mc.discriminator_weight * generator_fake_loss + mc.l1_weight * l1_loss + \
                     mc.clip_weight * clipping_loss + mc.ms_ssim_weight * ms_ssim_loss

    # Collect trainable parameter
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    g_learning_rate = tf.train.exponential_decay(mc.generator_lr, global_step,
                                                 1, 0.96, staircase=True)
    d_learning_rate = tf.train.exponential_decay(mc.discriminator_lr, global_step,
                                                 1, 0.96, staircase=True)

    generator_optimizer = tf.train.AdamOptimizer(g_learning_rate, beta1=mc.beta1).minimize(generator_loss,
                                                                                           var_list=g_vars)
    discriminator_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=mc.beta1).minimize(discriminator_loss,
                                                                                               var_list=d_vars)

    # Summaries
    tf.summary.scalar('l1_loss', l1_loss)
    tf.summary.scalar('clipping_loss', clipping_loss)
    tf.summary.scalar('ms_ssim_loss', ms_ssim_loss)
    tf.summary.scalar('discriminator_loss', discriminator_loss)
    tf.summary.scalar('generator_fake_loss', generator_fake_loss)
    tf.summary.scalar('generator_loss', generator_loss)
    tf.summary.scalar('generator_lr', g_learning_rate)
    tf.summary.scalar('discriminator_lr', d_learning_rate)
    tf.summary.image('generated_fake_frame', predicted_frame_mean_added_clipped)
    tf.summary.image('Before_frame', input_frames[:, :, :, :3] + mean_img)
    tf.summary.image('After_frame', input_frames[:, :, :, 3:] + mean_img)
    tf.summary.image('Target_frame', target_frame_mean_added_clipped)
    summary_op = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        file_writer = tf.summary.FileWriter(logdir='./Tensorboard', graph=sess.graph)
        step = 1
        for e in range(mc.n_epochs):
            n_batches = int(len(train_data) / mc.batch_size)
            for b in range(n_batches):
                batch_indx = np.random.permutation(len(train_data))[:mc.batch_size]
                train_data_batch = [train_data[t] for t in batch_indx]
                train_target_batch = [train_target[t] for t in batch_indx]

                for i in range(1):
                    sess.run(discriminator_optimizer,
                             feed_dict={input_frames: train_data_batch, target_frame: train_target_batch,
                                        global_step: step})

                for i in range(1):
                    sess.run(generator_optimizer,
                             feed_dict={input_frames: train_data_batch, target_frame: train_target_batch,
                                        global_step: step})

                s, l, dl, gl = sess.run([summary_op, l1_loss, discriminator_loss, generator_fake_loss],
                                        feed_dict={input_frames: train_data_batch, target_frame: train_target_batch,
                                                   global_step: step})

                print("\rEpoch: {}/{} \t Batch: {}/{}  l1_loss: {} disc_loss: {} gen_loss: {}".format(e, mc.n_epochs, b,
                                                                                                      n_batches, l, dl,
                                                                                                      gl))
                sys.stdout.flush()
                file_writer.add_summary(s)


if __name__ == '__main__':
    train()
