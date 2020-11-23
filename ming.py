from opts import parser
import torch
import torch.nn as nn
import torchvision

from tsn import TSN
from dataset import UCF101Dataset
from utils import AverageMeter, accuracy
from transforms import GroupNormalize



model = TSN(num_classes=101, num_frames=8, backbone="resnet50_tsm",
            consensus_type="avg", dropout=0.5, shift_div=8,
            shift_mode="residual", pretrained=True).cuda()

state_dict = torch.load("output_dir/2020-11-23_15_00_09/checkpoints/best.pth.tar")

# print(state_dict['state_dict'])

model.load_state_dict(state_dict['state_dict'])
