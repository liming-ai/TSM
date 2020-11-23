import torch
import torch.nn as nn
from backbones.resnet_tsm import resnet18_tsm, resnet34_tsm, resnet50_tsm, resnet101_tsm, resnet152_tsm
from backbones.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class Identity(nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(nn.Module):
    """Segment Consensus Module

    In fact it's just a pooling module, there are three type of pooling in paper:
    max, average and weighted average. But here we only implement average.
    For more details please see: `Temporal Segment Networks: Towards Good Practices for Deep Action Recognition`
    arXiv: https://arxiv.org/pdf/1608.00859.pdf

    Args:
        consensus_type (str): The way to pooling features.
        dim (int, optional): The dimension to pooling. Defaults to 1.
    """
    def __init__(self, consensus_type, dim=1):
        super().__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        if self.consensus_type == 'avg':
            return torch.mean(input_tensor, dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            return input_tensor
        else:
            return None


class TSN(nn.Module):
    """TSN Model

    Args:
        num_classes (int): Num of classes for a specific dataset.
        num_frames (int): The number of frames used as model input in a video.
        backbone (str, optional): Backbone's name. Defaults to 'resnet50_tsm'.
        consensus_type (str, optional): Consensus operation's name. Defaults to 'avg'.
        dropout (float, optional): Probability of dropout. Defaults to 0.8.
        shift_div (int, optional): Number of divisions for shift. Defaults to 8.
        shift_mode (str, optional): In-place shift or residual shift. Defaults to 'residual'.
        pretrained (bool, optional): Whether useing ImageNet pretrained model. Defaults to True.
    """

    def __init__(self, num_classes, num_frames, backbone='resnet50_tsm',
                 consensus_type='avg', dropout=0.8, shift_div=8,
                 shift_mode='residual', pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.consensus = SegmentConsensus(consensus_type)
        self.dropout = dropout
        self.shift_div = shift_div
        self.shift_mode = shift_mode
        self.pretrained = pretrained
        self.softmax = nn.Softmax(dim=1)  # if discard `dim=1`, there is warning

        self._prepare_backbone(backbone)
        self._update_fc(num_classes)


    def _prepare_backbone(self, backbone):
        if backbone == 'resnet18':
            self.backbone = resnet18(self.pretrained)
        elif backbone == 'resnet34':
            self.backbone = resnet34(self.pretrained)
        elif backbone == 'resnet50':
            self.backbone = resnet50(self.pretrained)
        elif backbone == 'resnet101':
            self.backbone = resnet101(self.pretrained)
        elif backbone == 'resnet152':
            self.backbone = resnet152(self.pretrained)
        elif backbone == 'resnet18_tsm':
            self.backbone = resnet18_tsm(self.num_frames, self.shift_div, self.shift_mode, self.pretrained)
        elif backbone == 'resnet34_tsm':
            self.backbone = resnet34_tsm(self.num_frames, self.shift_div, self.shift_mode, self.pretrained)
        elif backbone == 'resnet50_tsm':
            self.backbone = resnet50_tsm(self.num_frames, self.shift_div, self.shift_mode, self.pretrained)
        elif backbone == 'resnet101_tsm':
            self.backbone = resnet101_tsm(self.num_frames, self.shift_div, self.shift_mode, self.pretrained)
        elif backbone == 'resnet152_tsm':
            self.backbone = resnet152_tsm(self.num_frames, self.shift_div, self.shift_mode, self.pretrained)
        else:
            return None


    def _update_fc(self, num_classes):
        """
        Update the fully connected layer, change the number of channels it outputs. If there is dropout, add it.

        Args:
            num_classes (int): Number of classes for a specific dataset.
        """
        feature_dim = getattr(self.backbone.net, 'fc').in_features

        self.new_fc = nn.Linear(feature_dim, num_classes)

        # Initialize the weights of new fully-connected layer.
        if hasattr(self.new_fc, 'weight'):
            nn.init.normal_(self.new_fc.weight, 0, std=0.001)
            nn.init.constant_(self.new_fc.bias, 0)
        # Add dropout.
        if self.dropout != 0:
            setattr(self.backbone.net, 'fc', nn.Sequential(
                nn.Dropout(p = self.dropout),
                self.new_fc
            ))
        else:
            setattr(self.backbone.net, 'fc', self.new_fc)


    def train(self, mode=True):
        """
        Freezing BatchNorm2D expect the first one.

        Args:
            mode (bool): Whether train mode or not.
        """
        super(TSN, self).train(mode)
        cnt = 0
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                cnt += 1
                if cnt > 1:
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


    def forward(self, x):
        out = self.backbone(x)  # (bt, c, h, w) -> (bt, num_classes)
        out = out.view((-1, self.num_frames) + out.size()[1:])  # (bt, num_classes) -> (batch, time, num_classes)
        out = self.consensus(out)  # (batch, time, num_classes) -> (batch, 1, num_classes)
        out = out.squeeze(1)  # (batch, 1, num_classes) -> (batch, num_classes)
        return out


if __name__ == '__main__':
    tsn_r50 = TSN(num_classes=10, num_frames=8)
    # batch size = 4, num_frames = 8
    x = torch.randn(4 * 8, 3, 224, 224)
    y = tsn_r50(x)
    print(y.shape)
