import torchvision
import random
import numpy as np
import torch
import cv2
import mmcv


def normalize(video_frames, mean, std):
    """
    Normalize a group of image with mean and std.

    Args:
        video_frames (numpy.ndarray): (num_clips, T, H, W, C)
        mean (list): Mean
        std (list): Std

    Returns:
        numpy.ndarray: (num_clips*T, H, W, C)
    """
    # (num_clips, T, H, W, C) -> (num_clips*T, H, W, C)
    (num_clips, T, H, W, C) = video_frames.shape
    video_frames = video_frames.reshape(-1, H, W, C)

    # The calculation can refer to https://github.com/open-mmlab/mmcv/blob/c6c230df1b780976ee99f59e4644941967db39f9/mmcv/image/photometric.py#L24
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    # # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    for frame in video_frames:
        cv2.subtract(frame, mean, frame)    # inplace
        cv2.multiply(frame, stdinv, frame)  # inplace

    return video_frames


def random_crop(video_frames, random_size):
    """
    For train/val mode, each frame is randomly cropped so that its short side ranges in [256, 320] pixels and keep original aspect ratio.

    Args:
        video_frames (torch.tensor): input video frames with shape (N, C, H, W)
        random_size (list): [min_scale, max_scale], min_scale is the minimal size to scale the frames, max_scale is the maximal size to scale the frames.
    """
    size = int(round(np.random.uniform(min_size, max_size)))

    height = video_frames.shape[2]
    width = video_frames[3]

    if (width <= height and width == size) or (height <= width and height == size):
        return video_frames

    new_width = size
    new_height = size

    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))

    return torch.nn.functional.interpolate(
        video_frames, size=(new_height, new_width), mode='bilinear', align_corners=False
    )




def uniform_crop(video_frames, sampling_strategy):
    """
    For test mode, there are two sampling strategies:
        1. Three-crop: three random crops of size 256*256, usually for large dataset like Kinetics.
        2. Ten-crop: 4 cornor crops and 1 center crop of size 224*224, then flips these crops, ususally for middle dataset like Something-Something or small dataset like UCF101.

    Args:
        video_frames (torch.tensor): input video frames with shape (N, C, H, W)
        sampling_strategy (string): three-crop or two-crop
    """

    assert sampling_strategy in [
        "three-crop",
        "ten-crop"
    ], "Only support three-crop or ten-crop for test mode!"
