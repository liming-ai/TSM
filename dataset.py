import os
import random
from torch.utils.data import Dataset

import logging as logging

from decoder import decode
import utils

logger = logging.get_logger(__name__)


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

        assert self.mode in [
            'train',
            'val',
            'test'
        ], "Split '{}' not supported for UCF101".format(self.mode)

        if mode in ["train", "val"]:
            self.num_clips = cfg.train_num_clips
        elif mode in ["test"]:
            self.num_clips = cfg.test_num_clips * cfg.test_num_crops

        self.txt_file = os.path.join(self.mode.root_path, "{}.txt".format(mode))


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
                self.video_names.append(line[0])
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
            self.cfg
        )
        # Normalize and transform numpy.array to torch.Tensor
        self.video_frames = utils.normalize(
            self.video_frames, cfg.mean, cfg.std
        )
        # (N, H, W, C) -> (N, C, H, W), N = num_clips*cfg.num_frames
        self.video_frames = \
            torch.from_numpy(self.video_frames).permute(0, 3, 1, 2)

        if self.mode in ["train", "val"]:
            self.video_frames = utils.random_crop(
                self.video_frames, cfg.random_size
            )
            self.video_frames = utils.horizontal_flip(
                self.video_frames, cfg.horizontal_flip
            )
        elif self.mode in ["test"]:
            self.video_frames = utils.uniform_crop(
                self.video_frames, cfg.test_strategy, cfg.test_crop_size
            )

        return self.video_frames, torch.tensor(self.video_labels)