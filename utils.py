import skvideo.io
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os


def generate_dataset_from_video(video_path):
    """
    Convert the video frame into the desired format for training and testing
    :param video_path: String, path of the video
    :return: train_data   -> [N, H, W, 6]
             train_target -> [H, W, 3]
             test_data    -> [N, H, W, 6]
             test_target  -> [H, W, 3]
    """
    train_data = []
    train_target = []
    test_data = []
    test_target = []

    frames = skvideo.io.vread(video_path)
    frames = np.array(frames / 255, dtype=np.float32)
    mean_img = np.mean(frames[::2], 0)
    frames = frames - mean_img

    for frame_index in range(len(frames)):
        if frame_index % 2 == 1:
            test_target.append(frames[frame_index])
        else:
            try:
                train_data.append(np.append(frames[frame_index], frames[frame_index + 4], axis=2))
                train_target.append(frames[frame_index + 2])
                test_data.append(np.append(frames[frame_index], frames[frame_index + 2], axis=2))
            except IndexError:
                print("Dataset generation done!")
                break
    train_data = np.array(train_data).reshape([-1, 288, 352, 6])
    train_target = np.array(train_target).reshape([-1, 288, 352, 3])
    test_data = np.array(test_data).reshape([-1, 288, 352, 6])
    test_target = np.array(test_target).reshape([-1, 288, 352, 3])

    return train_data, train_target, test_data, test_target, mean_img


def split_video_frames(video_path):
    video_frames = []

    frames = skvideo.io.vread(video_path)
    frames = np.array(frames / 255, dtype=np.float32).reshape([-1, 288, 352, 3])
    mean_img = np.mean(frames[::2], 0)
    frames = frames - mean_img

    for frame_index in range(len(frames)):
        try:
            video_frames.append(np.append(frames[frame_index], frames[frame_index + 1], axis=2))
        except IndexError:
            print("Dataset prepared!")
            break

    video_frames = np.array(video_frames).reshape([-1, 288, 352, 6])
    return video_frames


# TODO: Remove this function later
def split_video_frames_v2(images_path):

    frames = []
    train_data = []
    train_target = []
    test_data = []
    test_target = []

    img_paths = sorted(os.listdir(images_path))[1:]

    for i, img_path in enumerate(img_paths):
        img = Image.open(images_path + '/' + img_path)
        img = ImageOps.crop(img, 130)
        img = np.array(img.resize([352, 288]))[:, :, :3]
        img = img/255
        frames.append(img)
    frames = np.array(frames).reshape([-1, 288, 352, 3])

    mean_img = np.mean(frames[::2], 0)
    frames = frames - mean_img
    for frame_index in range(len(frames)):
        if frame_index % 2 == 1:
            test_target.append(frames[frame_index])
        else:
            try:
                train_data.append(np.append(frames[frame_index], frames[frame_index + 4], axis=2))
                train_target.append(frames[frame_index + 2])
                test_data.append(np.append(frames[frame_index], frames[frame_index + 2], axis=2))
            except IndexError:
                print("Dataset generation done!")
                break
    train_data = np.array(train_data).reshape([-1, 288, 352, 6])
    train_target = np.array(train_target).reshape([-1, 288, 352, 3])
    test_data = np.array(test_data).reshape([-1, 288, 352, 6])
    test_target = np.array(test_target).reshape([-1, 288, 352, 3])

    return train_data, train_target, test_data, test_target, mean_img


# TODO: Remove this later
def split_video_frames_v3(images_path):
    video_frames = []
    frames = []

    img_paths = sorted(os.listdir(images_path))[1:]

    for i, img_path in enumerate(img_paths):
        img = Image.open(images_path + '/' + img_path)
        img = ImageOps.crop(img, 130)
        img = np.array(img.resize([352, 288]))[:, :, :3]
        img = img/255
        frames.append(img)
    frames = np.array(frames).reshape([-1, 288, 352, 3])

    for i, frame in enumerate(frames):
        plt.imsave('./Dataset/i_{:05d}.png'.format(i), frame)

    mean_img = np.mean(frames[::2], 0)
    frames = frames - mean_img

    for frame_index in range(len(frames)):
        try:
            video_frames.append(np.append(frames[frame_index], frames[frame_index + 1], axis=2))
        except IndexError:
            print("Dataset prepared!")
            break

    video_frames = np.array(video_frames).reshape([-1, 288, 352, 6])
    return video_frames


# TODO: Understand this
def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    with tf.variable_scope("ms_ssim_loss"):
        img1 = tf.image.rgb_to_grayscale(img1)
        img2 = tf.image.rgb_to_grayscale(img2)
        weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
        mssim = []
        mcs = []
        for l in range(level):
            ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
            mssim.append(tf.reduce_mean(ssim_map))
            mcs.append(tf.reduce_mean(cs_map))
            filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            img1 = filtered_im1
            img2 = filtered_im2

        # list to tensor of dim D+1
        mssim = tf.stack(mssim, axis=0)
        mcs = tf.stack(mcs, axis=0)

        value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) *
                 (mssim[level - 1] ** weight[level - 1]))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                              (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)
