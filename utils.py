import torchvision
import random
import numpy as np
import torch
import cv2
import mmcv


# N, C, H, W
def three_crop(video_frames, crop_size):
    """
    Crop images into three crops with equal intervals along the shorter side.

    Args:
        video_frames (torch.Tensor): input video frames with shape (N, C, H, W).
        crop_size (int): output size of cropped image.

    Returns:
        (list): Each element is a cropped video frames with shape (N, C, H, W), length of the list is 3.
    """
    crop_num = 3

    height = video_frames.shape[2]
    width = video_frames.shape[3]
    new_height = crop_size
    new_width = crop_size

    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))

    # change the shorter side to crop_size and keep aspect ratio
    video_frames = torch.nn.functional.interpolate(
        video_frames,
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False
    )

    cropped_video_frames = list()

    # crop along width
    if new_height < new_width:
        duration = (new_width - new_height) // (crop_num - 1)

        left_crop = torchvision.transforms.functional.crop(
            frame,  0, 0, crop_size, crop_size
        )
        center_crop = torchvision.transforms.functional.crop(
            frame, 0, duration, crop_size, crop_size
        )
        right_crop = torchvision.transforms.functional.crop(
            frame, 0, duration*2, crop_size, crop_size
        )

        cropped_video_frames.append(left_crop)
        cropped_video_frames.append(center_crop)
        cropped_video_frames.append(right_crop)

    # crop along height
    else:
        duration = (new_height - new_width) // (crop_num - 1)

        top_crop = torchvision.transforms.functional.crop(
            frame, 0, 0, crop_size, crop_size
        )
        center_crop = torchvision.transforms.functional.crop(
            frame, duration, 0, crop_size, crop_size
        )
        bottom_crop = torchvision.transforms.functional.crop(
            frame, duration*2, 0, crop_size, crop_size
        )

        cropped_video_frames.append(top_crop)
        cropped_video_frames.append(center_crop)
        cropped_video_frames.append(bottom_crop)

    return cropped_video_frames


def ten_crop(video_frames, crop_size):
    """
    Crop images into ten crops, including top-left, top-right, bottom-left, bottom-right, center-crop and their horizontal flips.

    Args:
        video_frames (torch.Tensor): input video frames with shape (N, C, H, W).
        crop_size (int): output size of cropped image.

    Returns:
        (list): Each element is a cropped video frames with shape (N, C, H, W), length of the list is 10.
    """
    return list(
        torchvision.transforms.functional.ten_crop(video_frames, crop_size)
    )


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
        video_frames (torch.Tensor): input video frames with shape (N, C, H, W)
        random_size (list): [min_scale, max_scale], min_scale is the minimal size to scale the frames, max_scale is the maximal size to scale the frames.

    Return:
        torch.Tensor: Scaled frames with shape (N, C, H, W)
    """
    size = int(round(np.random.uniform(min_size, max_size)))

    height = video_frames.shape[2]
    width = video_frames[3]

    if (width <= height and width == size) or (
        height <= width and height == size):
        return video_frames

    new_width = size
    new_height = size

    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))

    return torch.nn.functional.interpolate(
        video_frames,
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False
    )


def uniform_crop(video_frames, sampling_strategy, crop_size):
    """
    For test mode, there are two sampling strategies:
        1. Three-crop: three random crops of size 256*256, usually for large dataset like Kinetics.
        2. Ten-crop: 4 cornor crops and 1 center crop of size 224*224, then flips these crops, ususally for middle dataset like Something-Something or small dataset like UCF101.

    Args:
        video_frames (torch.Tensor): input video frames with shape (N, C, H, W)
        sampling_strategy (string): three-crop or two-crop

    Return:
        (list): list of frames, each element in list is a sequence of video frames with a crop, with three-crop, length of the list is 3, for ten-crop, length of the list is 10.

    """
    assert sampling_strategy in [
        "three-crop",
        "ten-crop"
    ], "Only support three-crop or ten-crop for test mode!"

    if sampling_strategy == "three-crop":
        return three_crop(video_frames, crop_size)
    elif sampling_strategy == "ten-crop":
        return ten_crop(video_frames, crop_size)


def horizontal_flip(video_frames, flip_probability):
    """
    Horizontal flip video frames with given probability.

    Args:
        video_frames (torch.Tensor): input video frames with shape (N, C, H, W).
        flip_probability (float): the probability of horizontal flipping.

    Returns:
        (torch.Tensor): the horizontal flipped video frames.
    """
    if torch.rand(1) < torch.Tensor(flip_probability):
        return torchvision.transforms.functional.hflip(video_frames)

    return video_frames