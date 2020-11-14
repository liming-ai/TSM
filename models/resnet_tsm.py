import torch
import torch.nn as nn
from .resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152


class TemporalShift(nn.Module):
    """
    Temporal Shift Module.

    Args:
        net (nn.module): Module to make temporal shift.
        num_clips (int): Number of frame clips. Defaults to 3.
        shift_div (int): Number of divisions for shift. Defaults to 8.
    """

    def __init__(self, net, num_clips=3, shift_div=8):
        super().__init__()
        self.net = net
        self.num_clips = num_clips
        self.shift_div = shift_div

    def forward(self, x):
        x = self.shift(x, self.num_clips, self.shift_div)
        return self.net(x)

    def shift(self, x, num_clips, shift_div):
        """
        Perform temporal shift on the feature

        Args:
            x (torch.Tensor): The input feature map with shape (n, c, h, w) to be shifted.
            num_clips (int): Number of frame clips. Defaults to 3.
            shift_div (int): Number of divisions for shift. Defaults to 8.

        Returns:
            torch.Tensor: The shifted feature map with shape (n, c, h, w)
        """
        n, c, h, w = x.shape

        # [N // num_clips, num_clips, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_clips, c, h*w)
        # get shift fold
        fold = c // shift_div

        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold: 2*fold, :]
        right_split = x[:, :, 2*fold:, :]

        # shift left on num_clips channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)

        # shift right on num_clips channel in `right_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)

        out = torch.cat((left_split, mid_split, right_split), 2)

        return out.view(n, c, h, w)


class ResNetTSM(nn.Module):
    """
    ResNet backbone for TSM.

    Args:
        backbone (str): ResNet backbone, there are `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`. Defaults to 'resnet50'.
        num_clips (int): Number of frame clips. Defaults to 3.
        shift_div (int): Number of divisions for shift. Defaults to 8.
        shift_mode (str): In-place shift or residual shift. Defaults to 'residual'.
        pretrained (bool): Weather use pretrained ResNet from torchvision. Defaults to False.
    """
    def __init__(self,
                 backbone='resnet50',
                 num_clips=3,
                 shift_div=8,
                 shift_mode='residual',
                 pretrained=False):
        super().__init__()

        self.num_clips = num_clips
        self.shift_div = shift_div
        self.shift_mode = shift_mode
        self.backbone = backbone

        assert backbone in [
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ], "Only support resnet18, resnet34, resnet50, resnet101, resnet152"

        if backbone == "resnet18":
            self.net = resnet18(pretrained)
        elif backbone == "resnet34":
            self.net = resnet34(pretrained)
        elif backbone == "resnet50":
            self.net = resnet50(pretrained)
        elif backbone == "resnet101":
            self.net = resnet101(pretrained)
        elif backbone == "resnet152":
            self.net = resnet152(pretrained)

    def make_temporal_layer(self):
        """
        The `in-place` shift happens before each residual block.
        The `residual` shift happens before the first convolution in a residual branch.

        Returns:
            nn.Sequential: Blocks that shifted the input feature map.
        """

        assert self.shift_mode in ["in-place", "residual"], "shift_mode must be in-place or residual"

        if self.shift_mode == "in-place":

            def make_block_layer(stage):
                blocks = list(stage)
                for i, block in enumerate(blocks):
                    blocks[i] = TemporalShift(block, self.num_clips, self.shift_div)

                return nn.Sequential(*blocks)

            self.net.layer1 = make_block_layer(self.net.layer1)
            self.net.layer2 = make_block_layer(self.net.layer2)
            self.net.layer3 = make_block_layer(self.net.layer3)
            self.net.layer4 = make_block_layer(self.net.layer4)

        elif self.shift_mode == "residual":

            def make_block_layer(stage):
                if self.backbone in ["resnet18", "resnet34"]:
                    n_round = 1
                elif self.backbone in ["resnet50", "resnet101", "resnet152"]:
                    n_round = 2

                blocks = list(stage)
                for i, block in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(block.conv1, self.num_clips, self.shift_div)

                return nn.Sequential(*blocks)

            self.net.layer1 = make_block_layer(self.net.layer1)
            self.net.layer2 = make_block_layer(self.net.layer2)
            self.net.layer3 = make_block_layer(self.net.layer3)
            self.net.layer4 = make_block_layer(self.net.layer4)


    def forward(self, x):
        self.make_temporal_layer()
        return self.net(x)