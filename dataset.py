import os
import random
import torch
from torch.utils.data import Dataset

import logging as logging

from decoder import decode
import utils
import argparse

# logger = logging.get_logger(__name__)


class UCF101Dataset(Dataset):
    """
    Video dataset for UCF101.

    The dataset loads clips of raw videos and apply specified transforms to
    return a dict containing the frame tensors and other information.

    The annotation file is a txt file with multiple lines, and each line
    indicates a sample Video with the filepath (without suffix) and label.

    Example:
        IceDancing/v_IceDancing_g18_c05.avi 43
        BoxingPunchingBag/v_BoxingPunchingBag_g20_c03.avi 16
        CuttingInKitchen/v_CuttingInKitchen_g09_c02.avi 24
    """
    def __init__(self, cfg, mode):
        """
        Construct the UCF101 video loader

        Args:
            cfg (CfgNode): Configs
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
        """
        super().__init__()

        self.cfg = cfg
        self.mode = mode

        assert mode in [
            'train',
            'val',
            'test'
        ], "Split '{}' not supported for UCF101".format(self.mode)

        if mode in ["train", "val"]:
            self.num_clips = cfg.train_num_clips
        elif mode in ["test"]:
            self.num_clips = cfg.test_num_clips * cfg.test_num_crops

        self.txt_file = os.path.join(self.cfg.root_path, "ucf101_{}_split_1_videos.txt".format(mode))


    def load_annotations(self):
        """
        Load annotations from txt file.
        """
        self.video_names = []
        self.video_labels = []

        with open(self.txt_file, "r") as f:
            for line in enumerate(f.read().splitlines()):
                line = line[-1].split()
                # e.g. YoYo/v_YoYo_g17_c04.avi
                self.video_names.append("videos/" + line[0])
                # e.g. 100
                self.video_labels.append(int(line[1]))


    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset
        """
        return len(self.video_labels)


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames and their frames.

        Args:
            index (int): the video index

        Returns:
            (torch.Tensor or list): for training and validation, return tensor with shape (N, C, H, W), for testing, return a list, the length of list is depended on three-crop or ten-crop testing strategy, each element of list is a tensor with shape (N, C, H, W), means a crop of sampled frames.
            (torch.Tensor): class of a video, shape is torch.Size([1])
        """
        self.load_annotations()
        # numpy.ndarray (num_clips, T, H, W, C)
        self.video_frames = decode(
            self.video_names[index],
            self.num_clips,
            self.cfg,
            self.mode
        )
        # Normalize and transform numpy.array to torch.Tensor
        self.video_frames = utils.normalize(
            self.video_frames, self.cfg.mean, self.cfg.std
        )
        # (N, H, W, C) -> (N, C, H, W), N = num_clips*cfg.num_frames
        self.video_frames = \
            torch.from_numpy(self.video_frames).permute(0, 3, 1, 2)

        if self.mode in ["train", "val"]:
            self.video_frames = utils.random_crop(
                self.video_frames, self.cfg.random_size
            )
            self.video_frames = utils.horizontal_flip(
                self.video_frames, self.cfg.horizontal_flip
            )

        return self.video_frames, torch.tensor(self.video_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSM Dataloader Test")
    parser.add_argument('--train_num_clips', type=int, default=8)
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--sampling_interval', type=int, default=1)
    parser.add_argument('--test_num_clips', type=int, default=10)
    parser.add_argument('--test_num_crops', type=int, default=3)
    parser.add_argument('--root_path', type=str, default="/home/liming/code/video/reproduce/TSM/data")
    parser.add_argument('--random_size', type=list, default=[256, 320])
    parser.add_argument('--horizontal_flip', type=float, default=0.5)
    parser.add_argument('--test_strategy', type=str, default="three-crop")
    parser.add_argument('--test_crop_size', type=int, default=224)
    parser.add_argument('--mean', type=list, default=[0, 0, 0])
    parser.add_argument('--std', type=list, default=[1, 1, 1])
    parser.add_argument('--target_fps', type=int, default=None)
    parser.add_argument('--temporal_jitter', type=bool, default=None)

    args = parser.parse_args()
    mode = "test"

    train_dataset = UCF101Dataset(args, mode)

    data, label = train_dataset.__getitem__(222)

    if mode in ["train", "val"]:
        print("Loading training data...")
        print(data.shape)
    elif mode in ["test"]:
        print("Loading test data...")
        print(len(data))
        print(data[0].shape)