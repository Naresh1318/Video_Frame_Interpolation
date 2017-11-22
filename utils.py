import skvideo.io
import numpy as np


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
    for frame_index in range(len(frames)):
        if frame_index % 2 == 1:
            test_target.append(frames[frame_index])
        else:
            try:
                train_data.append(np.append(frames[frame_index], frames[frame_index + 4], axis=2))
                train_target.append(frames[frame_index + 4])
                test_data.append(np.append(frames[frame_index], frames[frame_index + 2], axis=2))
            except IndexError:
                print("Dataset generation done!")
                break
    return train_data, train_target, test_data, test_target

