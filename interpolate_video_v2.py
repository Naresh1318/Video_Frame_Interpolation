import numpy as np
import tensorflow as tf
import mission_control as mc
import matplotlib.pyplot as plt
import ops
import utils
import sys

input_frames = tf.placeholder(dtype=tf.float32, shape=[None, 288, 352, 6], name="Input_Frames")
target_frame = tf.placeholder(dtype=tf.float32, shape=[None, 288, 352, 3], name="Target_Frame")
global_step = tf.placeholder(dtype=tf.int64, shape=[], name="Global_Step")

train_data, train_target, test_data, test_target, mean_img = utils.split_video_frames_v2(mc.images_path)


def rn_generator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # Encoder
    conv_1 = ops.lrelu(ops.cnn_2d(x, weight_shape=[4, 4, 3, 64], strides=[1, 2, 2, 1], name='g_rn_e_conv_1'))
    conv_2 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_1, weight_shape=[4, 4, 64, 128],
                                                 strides=[1, 2, 2, 1], name='g_rn_e_conv_2'),
                                      center=True, scale=True, is_training=True, scope='g_rn_e_batch_Norm_2'))
    conv_3 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_2, weight_shape=[4, 4, 128, 256],
                                                 strides=[1, 2, 2, 1], name='g_rn_e_conv_3'),
                                      center=True, scale=True, is_training=True, scope='g_rn_e_batch_Norm_3'))
    conv_4 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_3, weight_shape=[4, 4, 256, 512],
                                                 strides=[1, 2, 2, 1], name='g_rn_e_conv_4'),
                                      center=True, scale=True, is_training=True, scope='g_rn_e_batch_Norm_4'))
    conv_5 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_4, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='g_rn_e_conv_5'),
                                      center=True, scale=True, is_training=True, scope='g_rn_e_batch_Norm_5'))
    conv_6 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_5, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='g_rn_e_conv_6'),
                                      center=True, scale=True, is_training=True, scope='g_rn_e_batch_Norm_6'))
    conv_7 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_6, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='g_rn_e_conv_7'),
                                      center=True, scale=True, is_training=True, scope='g_rn_e_batch_Norm_7'))
    conv_8 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_7, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='g_rn_e_conv_8'),
                                      center=True, scale=True, is_training=True, scope='g_rn_e_batch_Norm_8'))

    # Decoder
    dconv_1 = ops.lrelu(tf.nn.dropout(ops.batch_norm(ops.cnn_2d_trans(conv_8, weight_shape=[2, 2, 512, 512], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, conv_8.get_shape()[1].value+1, conv_8.get_shape()[2].value+1, 512], name='g_rn_d_dconv_1'), center=True, scale=True, is_training=True, scope='g_rn_d_batch_Norm_1'), keep_prob=0.5))
    dconv_1 = tf.concat([dconv_1, conv_7], axis=3)
    dconv_2 = ops.lrelu(tf.nn.dropout(ops.batch_norm(ops.cnn_2d_trans(dconv_1, weight_shape=[4, 4, 512, 1024], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_1.get_shape()[1].value*2-1, dconv_1.get_shape()[2].value*2, 512], name='g_rn_d_dconv_2'), center=True, scale=True, is_training=True, scope='g_rn_d_batch_Norm_2'), keep_prob=0.5))
    dconv_2 = tf.concat([dconv_2, conv_6], axis=3)
    dconv_3 = ops.lrelu(tf.nn.dropout(ops.batch_norm(ops.cnn_2d_trans(dconv_2, weight_shape=[4, 4, 512, 1024], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_2.get_shape()[1].value*2-1, dconv_2.get_shape()[2].value*2-1, 512], name='g_rn_d_dconv_3'), center=True, scale=True, is_training=True, scope='g_rn_d_batch_Norm_3'), keep_prob=0.5))
    dconv_3 = tf.concat([dconv_3, conv_5], axis=3)
    dconv_4 = ops.lrelu(ops.batch_norm(ops.cnn_2d_trans(dconv_3, weight_shape=[4, 4, 512, 1024], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_3.get_shape()[1].value*2, dconv_3.get_shape()[2].value*2, 512], name='g_rn_d_dconv_4'), center=True, scale=True, is_training=True, scope='g_rn_d_batch_Norm_4'))
    dconv_4 = tf.concat([dconv_4, conv_4], axis=3)
    dconv_5 = ops.lrelu(ops.batch_norm(ops.cnn_2d_trans(dconv_4, weight_shape=[4, 4, 256, 1024], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_4.get_shape()[1].value*2, dconv_4.get_shape()[2].value*2, 256], name='g_rn_d_dconv_5'), center=True, scale=True, is_training=True, scope='g_rn_d_batch_Norm_5'))
    dconv_5 = tf.concat([dconv_5, conv_3], axis=3)
    dconv_6 = ops.lrelu(ops.batch_norm(ops.cnn_2d_trans(dconv_5, weight_shape=[4, 4, 128, 512], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_5.get_shape()[1].value*2, dconv_5.get_shape()[2].value*2, 128], name='g_rn_d_dconv_6'), center=True, scale=True, is_training=True, scope='g_rn_d_batch_Norm_6'))
    dconv_6 = tf.concat([dconv_6, conv_2], axis=3)
    dconv_7 = ops.lrelu(ops.batch_norm(ops.cnn_2d_trans(dconv_6, weight_shape=[4, 4, 64, 256], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_6.get_shape()[1].value*2, dconv_6.get_shape()[2].value*2, 64], name='g_rn_d_dconv_7'), center=True, scale=True, is_training=True, scope='g_rn_d_batch_Norm_7'))
    dconv_7 = tf.concat([dconv_7, conv_1], axis=3)
    dconv_8 = tf.nn.tanh(ops.cnn_2d_trans(dconv_7, weight_shape=[4, 4, 3, 128], strides=[1, 2, 2, 1], output_shape=[mc.batch_size, dconv_7.get_shape()[1].value*2, dconv_7.get_shape()[2].value*2, 3], name='g_rn_d_dconv_8'))
    return dconv_8


def rn_discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    conv_1 = ops.lrelu(ops.batch_norm(ops.cnn_2d(x, weight_shape=[4, 4, 6, 64],
                                                 strides=[1, 2, 2, 1], name='d_rn_conv_1'),
                                      center=True, scale=True, is_training=True, scope='d_rn_batch_Norm_1'))
    conv_2 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_1, weight_shape=[4, 4, 64, 128],
                                                 strides=[1, 2, 2, 1], name='d_rn_conv_2'),
                                      center=True, scale=True, is_training=True, scope='d_rn_batch_Norm_2'))
    conv_3 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_2, weight_shape=[4, 4, 128, 256],
                                                 strides=[1, 2, 2, 1], name='d_rn_conv_3'),
                                      center=True, scale=True, is_training=True, scope='d_rn_batch_Norm_3'))
    conv_4 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_3, weight_shape=[4, 4, 256, 512],
                                                 strides=[1, 2, 2, 1], name='d_rn_conv_4'),
                                      center=True, scale=True, is_training=True, scope='d_rn_batch_Norm_4'))
    conv_5 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_4, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='d_rn_conv_5'),
                                      center=True, scale=True, is_training=True, scope='d_rn_batch_Norm_5'))
    conv_6 = ops.lrelu(ops.batch_norm(ops.cnn_2d(conv_5, weight_shape=[4, 4, 512, 512],
                                                 strides=[1, 2, 2, 1], name='d_rn_conv_6'),
                                      center=True, scale=True, is_training=True, scope='d_rn_batch_Norm_6'))
    conv_6 = tf.reshape(conv_6, [-1, 5*6*512])
    output = ops.dense(conv_6, 5*6*512, 1, name='d_rn_output')
    return output


def sin_generator(x, reuse=False):
    # TODO: Add dropout??
    if reuse:
        tf.get_variable_scope().reuse_variables()
    # Encoder
    conv_b_1 = ops.conv_block(x, filter_size=3, stride_length=2, n_maps=32, name='g_sin_conv_b_1')
    conv_b_2 = ops.conv_block(conv_b_1, filter_size=3, stride_length=2, n_maps=64, name='g_sin_conv_b_2')
    conv_b_3 = ops.conv_block(conv_b_2, filter_size=3, stride_length=2, n_maps=64, name='g_sin_conv_b_3')
    conv_b_4 = ops.conv_block(conv_b_3, filter_size=3, stride_length=2, n_maps=128, name='g_sin_conv_b_4')

    # Decoder
    conv_tb_1 = ops.conv_t_block(conv_b_4, filter_size=4, stride_length=2, n_maps=128,
                                 output_shape=[mc.batch_size, conv_b_4.get_shape()[1].value * 2,
                                               conv_b_4.get_shape()[2].value * 2, 128], name='g_sin_conv_tb_1')
    conv_tb_1 = tf.concat([conv_tb_1, conv_b_3], axis=3)
    conv_tb_2 = ops.conv_t_block(conv_tb_1, filter_size=4, stride_length=2, n_maps=64,
                                 output_shape=[mc.batch_size, conv_tb_1.get_shape()[1].value * 2,
                                               conv_tb_1.get_shape()[2].value * 2, 64], name='g_sin_conv_tb_2')
    conv_tb_2 = tf.concat([conv_tb_2, conv_b_2], axis=3)
    conv_tb_3 = ops.conv_t_block(conv_tb_2, filter_size=4, stride_length=2, n_maps=64,
                                 output_shape=[mc.batch_size, conv_tb_2.get_shape()[1].value * 2,
                                               conv_tb_2.get_shape()[2].value * 2, 64], name='g_sin_conv_tb_3')
    conv_tb_3 = tf.concat([conv_tb_3, conv_b_1], axis=3)
    conv_tb_4 = ops.conv_t_block(conv_tb_3, filter_size=4, stride_length=2, n_maps=32,
                                 output_shape=[mc.batch_size, conv_tb_3.get_shape()[1].value * 2,
                                               conv_tb_3.get_shape()[2].value * 2, 32], name='g_sin_conv_tb_4')

    output = ops.cnn_2d_trans(conv_tb_4, weight_shape=[4, 4, 3, conv_tb_4.get_shape()[-1].value], strides=[1, 1, 1, 1],
                              output_shape=[mc.batch_size, conv_tb_4.get_shape()[1].value,
                                            conv_tb_4.get_shape()[2].value, 3], name='g_sin_output')
    return output


def sin_discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv_b_1 = ops.conv_block(x, filter_size=4, stride_length=2, n_maps=8, name='d_sin_conv_b_1')
    conv_b_2 = ops.conv_block(conv_b_1, filter_size=4, stride_length=2, n_maps=16, name='d_sin_conv_b_2')
    conv_b_3 = ops.conv_block(conv_b_2, filter_size=4, stride_length=2, n_maps=32, name='d_sin_conv_b_3')
    conv_b_4 = ops.conv_block(conv_b_3, filter_size=4, stride_length=2, n_maps=64, name='d_sin_conv_b_4')
    conv_b_5 = ops.conv_block(conv_b_4, filter_size=4, stride_length=2, n_maps=1, name='d_sin_conv_b_5')
    conv_b_5_r = tf.reshape(conv_b_5, [-1, 11 * 9 * 1], name='d_sin_reshape')
    output = ops.dense(conv_b_5_r, 11 * 9, 1, name='d_sin_output')
    return output


def train():
    with tf.variable_scope(tf.get_variable_scope()):
        sin_output_frame = sin_generator(input_frames)

    sin_discriminator_real_input = tf.concat([input_frames, target_frame], axis=3)
    sin_discriminator_fake_input = tf.concat([input_frames, sin_output_frame], axis=3)

    with tf.variable_scope(tf.get_variable_scope()):
        sin_real_discriminator_op = sin_discriminator(sin_discriminator_real_input)
        sin_fake_discriminator_op = sin_discriminator(sin_discriminator_fake_input, reuse=True)

    with tf.variable_scope(tf.get_variable_scope()):
        rn_output_frame = rn_generator(sin_output_frame)

    rn_discriminator_real_input = tf.concat([sin_output_frame, target_frame], axis=3)
    rn_discriminator_fake_input = tf.concat([sin_output_frame, rn_output_frame], axis=3)

    with tf.variable_scope(tf.get_variable_scope()):
        rn_real_discriminator_op = rn_discriminator(rn_discriminator_real_input)
        rn_fake_discriminator_op = rn_discriminator(rn_discriminator_fake_input, reuse=True)

    # GAN losses
    sin_generator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                             (labels=tf.ones_like(sin_fake_discriminator_op),
                                              logits=sin_fake_discriminator_op))
    sin_discriminator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                 (labels=tf.zeros_like(sin_fake_discriminator_op),
                                                  logits=sin_fake_discriminator_op))
    sin_discriminator_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                 (labels=tf.ones_like(sin_real_discriminator_op),
                                                  logits=sin_real_discriminator_op))

    rn_generator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                            (labels=tf.ones_like(rn_fake_discriminator_op),
                                             logits=rn_fake_discriminator_op))
    rn_discriminator_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                (labels=tf.zeros_like(rn_fake_discriminator_op),
                                                 logits=rn_fake_discriminator_op))
    rn_discriminator_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                                (labels=tf.ones_like(rn_real_discriminator_op),
                                                 logits=rn_real_discriminator_op))

    eps = 1e-5

    sin_l1_loss = tf.reduce_mean(tf.abs(sin_output_frame - target_frame + eps))

    rn_l1_loss = tf.reduce_mean(tf.abs(rn_output_frame - target_frame + eps))

    predicted_frame_mean_added = sin_output_frame + mean_img
    predicted_frame_mean_added_clipped = tf.clip_by_value(predicted_frame_mean_added, 0, 1)
    target_frame_mean_added_clipped = tf.clip_by_value(target_frame + mean_img, 0, 1)

    clipping_loss = tf.reduce_mean(tf.square(predicted_frame_mean_added_clipped - predicted_frame_mean_added))
    ms_ssim_loss = tf.reduce_mean(
        -tf.log(utils.tf_ms_ssim(predicted_frame_mean_added_clipped, target_frame_mean_added_clipped)))

    rn_output_frame_mean_added_clipped = tf.clip_by_value(rn_output_frame + mean_img, 0, 1)

    sin_discriminator_loss = sin_discriminator_fake_loss + sin_discriminator_real_loss
    sin_generator_loss = mc.discriminator_weight * sin_generator_fake_loss + mc.l1_weight * sin_l1_loss + \
                     mc.clip_weight * clipping_loss + mc.ms_ssim_weight * ms_ssim_loss

    rn_discriminator_loss = rn_discriminator_fake_loss + rn_discriminator_real_loss
    rn_generator_loss = rn_generator_fake_loss + mc.rn_weight * rn_l1_loss

    # Collect trainable parameter
    t_vars = tf.trainable_variables()
    d_sin_vars = [var for var in t_vars if 'd_sin_' in var.name]
    g_sin_vars = [var for var in t_vars if 'g_sin_' in var.name]
    d_rn_vars = [var for var in t_vars if 'd_rn_' in var.name]
    g_rn_vars = [var for var in t_vars if 'g_rn_' in var.name]

    g_learning_rate = tf.train.exponential_decay(mc.generator_lr, global_step,
                                                 1, 0.999, staircase=True)
    d_learning_rate = tf.train.exponential_decay(mc.discriminator_lr, global_step,
                                                 1, 0.999, staircase=True)

    sin_generator_optimizer = tf.train.AdamOptimizer(g_learning_rate, beta1=mc.beta1).minimize(sin_generator_loss,
                                                                                               var_list=g_sin_vars)
    sin_discriminator_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=mc.beta1).minimize(sin_discriminator_loss,
                                                                                                   var_list=d_sin_vars)
    rn_generator_optimizer = tf.train.AdamOptimizer(g_learning_rate, beta1=mc.beta1).minimize(rn_generator_loss,
                                                                                              var_list=g_rn_vars)
    rn_discriminator_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=mc.beta1).minimize(rn_discriminator_loss,
                                                                                                  var_list=d_rn_vars)

    # Summaries
    tf.summary.scalar('sin_l1_loss', sin_l1_loss)
    tf.summary.scalar('rn_l1_loss', rn_l1_loss)
    tf.summary.scalar('clipping_loss', clipping_loss)
    tf.summary.scalar('ms_ssim_loss', ms_ssim_loss)
    tf.summary.scalar('sin_discriminator_loss', sin_discriminator_loss)
    tf.summary.scalar('rn_discriminator_loss', rn_discriminator_loss)
    tf.summary.scalar('sin_generator_fake_loss', sin_generator_fake_loss)
    tf.summary.scalar('rn_generator_fake_loss', rn_generator_fake_loss)
    tf.summary.scalar('sin_generator_loss', sin_generator_loss)
    tf.summary.scalar('rn_generator_loss', rn_generator_loss)
    tf.summary.scalar('generator_lr', g_learning_rate)
    tf.summary.scalar('discriminator_lr', d_learning_rate)
    tf.summary.image('sin_generated_fake_frame', predicted_frame_mean_added_clipped)
    tf.summary.image('rn_generated_fake_image', rn_output_frame_mean_added_clipped)
    tf.summary.image('Before_frame', input_frames[:, :, :, :3] + mean_img)
    tf.summary.image('After_frame', input_frames[:, :, :, 3:] + mean_img)
    tf.summary.image('Target_frame', target_frame_mean_added_clipped)
    summary_op = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        file_writer = tf.summary.FileWriter(logdir=mc.results_path + '/Tensorboard_v3', graph=sess.graph)

        if mc.train_model:
            step = 1
            for e in range(mc.n_epochs):
                n_batches = int(len(train_data) / mc.batch_size)
                for b in range(n_batches):
                    batch_indx = np.random.permutation(len(train_data))[:mc.batch_size]
                    train_data_batch = [train_data[t] for t in batch_indx]
                    train_target_batch = [train_target[t] for t in batch_indx]

                    for i in range(1):
                        sess.run(sin_discriminator_optimizer,
                                 feed_dict={input_frames: train_data_batch, target_frame: train_target_batch,
                                            global_step: step})

                    for i in range(1):
                        sess.run(sin_generator_optimizer,
                                 feed_dict={input_frames: train_data_batch, target_frame: train_target_batch,
                                            global_step: step})

                    for i in range(1):
                        sess.run(rn_discriminator_optimizer,
                                 feed_dict={input_frames: train_data_batch, target_frame: train_target_batch,
                                            global_step: step})

                    for i in range(1):
                        sess.run(rn_generator_optimizer,
                                 feed_dict={input_frames: train_data_batch, target_frame: train_target_batch,
                                            global_step: step})

                    s, sin_l, sin_dl, sin_gl, rn_l, rn_dl, rn_gl, gs = \
                        sess.run([summary_op, sin_l1_loss, sin_discriminator_loss, sin_generator_fake_loss, rn_l1_loss,
                                  rn_discriminator_loss, rn_generator_fake_loss, global_step],
                                 feed_dict={input_frames: train_data_batch, target_frame: train_target_batch,
                                            global_step: step})

                    print("\rEpoch: {}/{} \t Batch: {}/{}  sin_l1_loss: {} sin_disc_loss: {} sin_gen_loss: {} \t "
                          "rn_l1_loss: {} rn_disc_loss: {} rn_gen_loss: {}".format(e, mc.n_epochs, b, n_batches, sin_l,
                                                                                   sin_dl, sin_gl, rn_l, rn_dl, rn_gl))
                    sys.stdout.flush()
                    file_writer.add_summary(s, step)
                    step += 1

                # TODO: Testing part not done yet

            # Save the trained model
            saver.save(sess, save_path=mc.results_path + "/Saved_models/earth")
        else:
            saver.restore(sess, save_path=tf.train.latest_checkpoint(mc.results_path + "/Saved_models/"))

            # TODO: Up-sample the entire video and produce a gif or a new video
            video_frames = utils.split_video_frames_v3(mc.images_path)

            intermediate_frames = []

            for i, frames in enumerate(video_frames):
                frames = frames.reshape(1, 288, 352, 6)
                inter_frame = sess.run(rn_output_frame, feed_dict={input_frames: frames})
                intermediate_frames.append(inter_frame)
                print("Generating frame: {}/{}".format(i, len(video_frames)))

            # Combine the input and the generated frames
            all_frames = []
            for i, frame in enumerate(intermediate_frames):
                all_frames.append(np.clip(video_frames[i][:, :, :3] + mean_img, 0, 1))
                frame = np.clip(frame + mean_img, 0, 1)
                all_frames.append(frame[0])
                # all_frames.append(np.clip(video_frames[i][:, :, 3:] + mean_img, 0, 1))
                print("Upsampling frame: {}/{}".format(i, len(intermediate_frames)))

            # Save all the generated images
            for i, f in enumerate(all_frames):
                plt.imsave("./Results/Output_frames/{:05d}.png".format(i), arr=f)


if __name__ == '__main__':
    train()
