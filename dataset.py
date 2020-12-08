import os
import random
import torch
from torch.utils.data import Dataset

import logging as logging

import argparse
import numpy as np
from PIL import Image
import torchvision
from transforms import *


# logger = logging.get_logger(__name__)
class VideoRecord(object):
    def __init__(self, line):
        self._data = line

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class UCF101Dataset(Dataset):
    """
    Video dataset for UCF101.

    The dataset loads clips of raw videos and apply specified transforms to
    return a dict containing the frame tensors and other information.

    The annotation file is a txt file with multiple lines, and each line
    indicates a sample Video with the filepath (without suffix) and label.

    Annotation File Example:
        IceDancing/v_IceDancing_g18_c05.avi 43
        BoxingPunchingBag/v_BoxingPunchingBag_g20_c03.avi 16
        CuttingInKitchen/v_CuttingInKitchen_g09_c02.avi 24

    Args:
        data_path (str): Directory that containing the frames or videos.
        anno_path (str): Path to the annotation file.
        mode (str): Must be 'train' 'val' or 'test'.
        sample_strategy (str): Must be 'sparse' or 'dense'.
            1. For 'sparse' sampling strategy, the whole video is divided into
               `num_segments` segments, and we only sample 1 frame from a
               segment, so `num_frames` and `sample_interval` must be 1.
            2. For 'dense' sampling strategy, `num_segments` must be 1, means
               the whole video is the only segment. We totally sample
               `num_frames` frames from the segment, each frame is sampled with
               interval `sample_interval`.

        num_frames (int): Num of frames in a segment.
        sample_interval (int): The distance required to sample a frame.
        num_segments (int): Num of segments.

        test_num_clips (int): Num of clips for testing, usually average their
            softmax value as the final result, must be 10, 3 or 1.
        test_num_crops (int): Num of crops for testing, must be 10, 5 or 1,
            means ten-crop, five-crop and center-crop.

        crop_size (int): Size for cropping.
        random_shift (bool): Weather random sample indices when mode=`train`
        image_template (str): Name template for dataset.
    """
    def __init__(self, data_path, anno_path, transforms=None, mode="train", sample_strategy="sparse",
                 num_frames=1, sample_interval=1, num_segments=8, test_num_clips=10, test_num_crops=3,
                 crop_size=224, random_shift=True, image_template="img_{:05d}.jpg"):

        super().__init__()

        assert mode in [
            'train', 'val', 'test'
        ], "Mode must be 'train', 'val' or 'test' to obtain dataset."

        if mode in ['train', 'val']:
            if sample_strategy == 'dense':
                assert num_segments == 1
            elif sample_strategy == 'sparse':
                assert num_frames == 1
                assert sample_interval == 1
            else:
                raise ValueError("Sample strategy must be 'sparse' or 'dense'")

        self.data_path = data_path
        self.anno_path = anno_path
        self.mode = mode
        self.sample_strategy = sample_strategy
        self.random_shift = random_shift
        self.image_template = image_template
        self.transforms = transforms
        self.crop_size = crop_size

        self.num_frames = num_frames
        self.sample_interval = sample_interval
        self.num_segments = num_segments

        self.test_num_clips = test_num_clips
        self.test_num_crops = test_num_crops

        self._load_annotations()


    def _load_annotations(self):
        videos_info = [x.strip().split(' ') for x in open(self.anno_path)]
        self.videos_list = [VideoRecord(item) for item in videos_info]

        print("Totally {} videos for {} dataset.".format(len(self.videos_list), self.mode))


    def _train_random_sample(self, record):
        if self.sample_strategy == 'dense':
            # sample_position is the max indices to sample a segment
            # the length of the segment is self.num_frames * self.sample_interval
            sample_position = max(1, record.num_frames - self.num_frames * self.sample_interval + 1)
            # random choose a index from [0, sample_position - 1) as the start index of the segment
            start_idx = 0 if sample_position == 1 else np.random.randint(0, sample_position - 1)
            # get the final index list, each element is the index of a frame used to train
            indices = [(idx * self.sample_interval + start_idx) % record.num_frames for idx in range(self.num_frames)]
            # the index in data_path is started from 1, e.g. rawframes/YoYo/v_YoYo_g01_c01/img_00001.jpg
            return np.array(indices) + 1
        elif self.sample_strategy == 'sparse':
            # divide the whole video into self.num_segment segments.
            num_frames_per_segment = record.num_frames // self.num_segments
            # random choose a frame in a segment
            if num_frames_per_segment > 0:
                # the first index of each segment.
                start_indice_per_segment = np.array(
                    [(idx * num_frames_per_segment) for idx in range(self.num_segments)]
                )
                # random choose a frame from each segment.
                indices = start_indice_per_segment + np.random.randint(
                    num_frames_per_segment, size=self.num_segments
                )
            else:
                indices = np.zeros((self.num_segments,))
            return indices + 1


    def _train_uniform_sample(self, record):
        if self.sample_strategy == 'dense':
            sample_position = max(1, record.num_frames - self.num_frames * self.sample_interval + 1)
            start_idx = 0 if sample_position == 1 else np.random.randint(0, sample_position - 1)
            indices = [(idx * self.sample_interval + start_idx) for idx in range(self.num_frames)]
            return np.array(indices) + 1
        elif self.sample_strategy == 'sparse':
            num_frames_per_segment = record.num_frames // self.num_segments
            if num_frames_per_segment > 0:
                start_indice_per_segment = np.array(
                    [(idx * num_frames_per_segment) for idx in range(self.num_segments)]
                )
                # choose the mid frame in a segment
                indices = start_indice_per_segment + np.multiply(
                    np.ones((self.num_segments,)), int(num_frames_per_segment / 2.0)
                )
                # change the type of indices' value from float to int
                indices = indices.astype(np.int64)
            else:
                indices = np.zeros((self.num_segments,))
            return indices + 1


    def _test_sample(self, record):
        num_frames_per_segment = record.num_frames // self.num_segments
        # when self.num_clips == 1, using sparse sampling, choose the mid frame each segment
        if self.test_num_clips == 1:
            assert self.sample_strategy == 'sparse', "when test_num_clips is 1 or 2, must use sparse sampling"
            start_indice_per_segment = np.array(
                [(idx * num_frames_per_segment) for idx in range(self.num_segments)]
            )
            indices = start_indice_per_segment + np.multiply(
                np.ones((self.num_segments,)), int(num_frames_per_segment / 2.0)
            )
            indices = indices.astype(np.int64)
            return indices + 1
        # when self.num_clips == 2, using sparse sampling
        # choose the mid and the last frame each segment
        elif self.test_num_clips == 2:
            assert self.sample_strategy == 'sparse', "when test_num_clips is 1 or 2, must use sparse sampling"
            num_frames_per_segment = record.num_frames // self.num_segments
            start_indice_per_segment = [(idx * num_frames_per_segment) for idx in range(self.num_segments)]
            # choose the mid frame
            first_clip_indices = [(start_index + num_frames_per_segment // 2) for start_index in start_indice_per_segment]
            # choose the last frame
            second_clip_indices = [(start_index + num_frames_per_segment - 1) for start_index in start_indice_per_segment]
            indices = np.concatenate((first_clip_indices, second_clip_indices))
            return indices + 1
        # when self.test_num_clips > 2, using dense sample strategy
        elif self.test_num_clips > 2:
            assert self.sample_strategy == 'dense', "when test_num_clips > 2, must use dense sampling"
            sample_position = max(1, record.num_frames - self.num_frames * self.sample_interval + 1)
            # uniformly get `test_num_clips` start indexes
            # each start index is corresponding to a clip, which used to test
            start_list = np.linspace(0, sample_position - 1, num=self.test_num_clips, dtype=int)
            indices = []
            for start_idx in start_list.tolist():
                indices += [(idx * self.sample_interval + start_idx) % record.num_frames for idx in range(self.num_frames)]
            return np.array(indices) + 1
        else:
            raise ValueError("test_num_clips must be a positive number")


    def _load_image(self, record, idx):
        try:
            return Image.open(os.path.join(self.data_path, record.path, self.image_template.format(idx))).convert('RGB')
        except Exception:
            print("=========================================")
            print(self.data_path)
            print(record.path)
            print(self.image_template.format(idx))
            print('error loading image: ', os.path.join(self.data_path, record.path, self.image_template.format(idx)))


    def __len__(self):
        return len(self.videos_list)


    def __getitem__(self, index):
        record = self.videos_list[index]

        if self.mode in ['train', 'val']:
            indices = self._train_random_sample(record) if self.random_shift else self._train_uniform_sample(record)
        elif self.mode in ['test']:
            indices = self._test_sample(record)

        images = []
        for index in indices:
            image = self._load_image(record, index)
            images.append(image)

        if self.transforms is not None:
            images = self.transforms(images)

        return images, record.label


if __name__ == "__main__":
    from opts import parser
    args = parser.parse_args()
    transforms = torchvision.transforms.Compose([
        GroupResize(args.crop_size),
        GroupThreeCrop(args.crop_size),
        GroupToTensor(),
        GroupBatchNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = UCF101Dataset(args.data_path, args.val_anno_path, transforms=transforms, mode='test',
                                  sample_strategy=args.sample_strategy, num_frames=args.num_frames,
                                  sample_interval=args.sample_interval, num_segments=args.num_segments,
                                  test_num_clips=args.test_num_clips, test_num_crops=args.test_num_crops,
                                  crop_size=args.crop_size, random_shift=args.random_shift)
    data, label = test_dataset.__getitem__(222)

    print(data.shape)