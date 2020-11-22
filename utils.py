import torchvision
import random
import numpy as np
import torch
import cv2
import math


def three_crop(image, crop_size):
    """
    Given a image, return a list of three cropped images.

    Args:
        image (PIL.Image): Input image.
        crop_size (int): The needed size for cropping.

    Returns:
        list: Cropped images.
    """
    crop_num = 3

    height = image.size[0]
    width = image.size[1]
    new_height = crop_size
    new_width = crop_size

    if width < height:
        new_height = int(math.floor((float(height) / width) * crop_size))
    else:
        new_width = int(math.floor((float(width) / height) * crop_size))
    # import pdb; pdb.set_trace()
    # change the shorter side to crop_size and keep aspect ratio
    image_resizer = torchvision.transforms.Resize((new_height, new_width))
    image = image_resizer(image)

    cropped_images = list()

    # crop along width
    if new_height < new_width:
        duration = (new_width - new_height) // (crop_num - 1)

        left_crop = torchvision.transforms.functional.crop(
            image,  0, 0, crop_size, crop_size
        )
        center_crop = torchvision.transforms.functional.crop(
            image, 0, duration, crop_size, crop_size
        )
        right_crop = torchvision.transforms.functional.crop(
            image, 0, duration*2, crop_size, crop_size
        )

        cropped_images.append(left_crop)
        cropped_images.append(center_crop)
        cropped_images.append(right_crop)

    # crop along height
    else:
        duration = (new_height - new_width) // (crop_num - 1)

        top_crop = torchvision.transforms.functional.crop(
            image, 0, 0, crop_size, crop_size
        )
        center_crop = torchvision.transforms.functional.crop(
            image, duration, 0, crop_size, crop_size
        )
        bottom_crop = torchvision.transforms.functional.crop(
            image, duration*2, 0, crop_size, crop_size
        )

        cropped_images.append(top_crop)
        cropped_images.append(center_crop)
        cropped_images.append(bottom_crop)

    return cropped_images


def ten_crop(image, crop_size):
    return list(
        torchvision.transforms.functional.ten_crop(image, crop_size, 0.5)
    )


def center_crop(image, crop_size):
    return [torchvision.transforms.functional.center_crop(image, crop_size)]


def random_crop(images, crop_size):
    cropped_images = [torchvision.transforms.RandomCrop(crop_size)(image) for image in images]
    return cropped_images


def test_crop(images, crop_size, num_crops):
    """
    Return cropped images with crop_size.

    Args:
        images (list): Input images.
        crop_size (int): Cropped image's size.
        num_crops (int): Must be 10, 3, or 1, representing ten-crop, three-crop and center-crop.

    Returns:
        list: The cropped images.
    """
    cropped_images = []
    if num_crops == 1:
        for image in images:
            cropped_images += center_crop(image, crop_size)
    elif num_crops == 3:
        for image in images:
            cropped_images += three_crop(image, crop_size)
    elif num_crops == 10:
        for image in images:
            cropped_images += ten_crop(image, crop_size)
    else:
        raise ValueError("num_crops must be 1, 3 or 10, means center crop, three crop and ten crop.")

    return cropped_images


def horizontal_flip(image, flip_probability):
    """
    Horizontal flip video frames with given probability.

    Args:
        video_frames (torch.Tensor): input video frames with shape (N, C, H, W).
        flip_probability (float): the probability of horizontal flipping.

    Returns:
        (torch.Tensor): the horizontal flipped video frames.
    """
    if torch.rand(1) < torch.Tensor([flip_probability]):
        return torchvision.transforms.functional.hflip(image)

    return image