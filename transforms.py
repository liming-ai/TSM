import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch


class GroupRandomCrop(object):
    def __init__(self, crop_size):
        self.worker = torchvision.transforms.RandomCrop(crop_size)

    def __call__(self, images):
        return [self.worker(image) for image in images]


class GroupCenterCrop(object):
    def __init__(self, crop_size):
        self.worker = torchvision.transforms.CenterCrop(crop_size)

    def __call__(self, images):
        return [self.worker(image) for image in images]


class GroupRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.worker = torchvision.transforms.RandomHorizontalFlip(p)

    def __call__(self, images):
        return [self.worker(image) for image in images]


class GroupResize(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.Resize(size)

    def __call__(self, images):
        return [self.worker(image) for image in images]


class GroupTenCrop(object):
    def __init__(self, crop_size):
        self.worker = torchvision.transforms.TenCrop(crop_size)
        self.crop_size = crop_size

    def __call__(self, images):
        width, height = images[0].size
        assert min(width, height) == self.crop_size, "Before ten-crop, image's size must match crop_size, please add 'GroupResize(crop_size)' before 'GroupTenCrop(crop_size)' in self.transforms"

        cropped_images = list()
        for image in images:
            cropped_images += list(self.worker(image))

        return cropped_images


class GroupThreeCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, images):
        width, height = images[0].size
        assert min(width, height) == self.crop_size, "Before three-crop, image's size must match crop_size, please add 'GroupResize(crop_size)' before 'GroupThreeCrop(crop_size)' in self.transforms"

        cropped_width = self.crop_size
        cropped_height = self.crop_size

        # if width < height, then width == crop_size, width_step = 0
        # if height < width, then height == crop_size, height_step = 0
        width_step = (width - cropped_width) // 2  # (341-256)//2
        height_step = (height - cropped_height) // 2 # 0

        offsets = list()
        offsets.append((0 * width_step, 0 * height_step))  # left or top
        offsets.append((1 * width_step, 1 * height_step))  # mid or center
        offsets.append((2 * width_step, 2 * height_step))  # right or bottom

        cropped_images = list()

        for w, h in offsets:
            for image in images:
                cropped_image = image.crop((w, h, w + cropped_width, h + cropped_height))
                cropped_images.append(cropped_image)

        return cropped_images


class GroupToTensor(object):
    def __init__(self):
        self.worker = torchvision.transforms.ToTensor()

    def __call__(self, images):
        images = [self.worker(image) for image in images]
        return torch.stack(images)  # from list to tensor with shape (num_frames, C, H, W)


class GroupBatchNormalize(object):
    def __init__(self, mean, std):
        self.worker = torchvision.transforms.Normalize(mean, std)

    def __call__(self, tensors):
        return self.worker(tensors)


class GroupRandomMultiScaleCrop(object):
    def __init__(self, input_size, scales=None):
        self.input_size = [input_size, input_size] if isinstance(input_size, int) else input_size
        self.scales = [1, .875, .75, .66] if scales is None else scales

    def __call__(self, images):
        image_size = images[0].size

        w, h, w_step, h_step = self._sample_crop_size(image_size)
        cropped_images = [image.crop((w, h, w + w_step, h + h_step)) for image in images]
        output_images = [image.resize((self.input_size[0], self.input_size[1]), Image.BILINEAR) for image in images]

        return output_images

    def _sample_crop_size(self, img_size):
        image_w, image_h = img_size[0], img_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * scale) for scale in self.scales]
        crop_w = [self.input_size[0] if abs(crop_size - self.input_size[0]) < 3 else crop_size for crop_size in crop_sizes]
        crop_h = [self.input_size[1] if abs(crop_size - self.input_size[1]) < 3 else crop_size for crop_size in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= 1:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset


    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        ret.append((0, 2 * h_step))  # center left
        ret.append((4 * w_step, 2 * h_step))  # center right
        ret.append((2 * w_step, 4 * h_step))  # lower center
        ret.append((2 * w_step, 0 * h_step))  # upper center

        ret.append((1 * w_step, 1 * h_step))  # upper left quarter
        ret.append((3 * w_step, 1 * h_step))  # upper right quarter
        ret.append((1 * w_step, 3 * h_step))  # lower left quarter
        ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret