from opts import parser
import torch
import torch.nn as nn
import torchvision

from tsn import TSN
from dataset import UCF101Dataset
from utils import AverageMeter, accuracy
from transforms import *
from torch.utils.tensorboard import SummaryWriter

def main():
    args = parser.parse_args()

    model = TSN(num_classes=args.num_classes, num_frames=args.num_frames*args.num_segments, backbone=args.backbone,
                consensus_type=args.consensus_type, dropout=args.dropout, shift_div=args.shift_div,
                shift_mode=args.shift_mode, pretrained=args.pretrained)

    model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    pretrained_state_dict = torch.load("output_dir/2020-11-24_13_21_50/checkpoints/resnet50_tsm_25_sparse_1_1_8.pth.tar")
    model.load_state_dict(pretrained_state_dict['state_dict'])

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

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.cuda()
            label = label.cuda()

            b, n, c, h, w = data.shape  # e.g.  16, 8*2*3, 3, 224, 224, 2 clips and three-crop
            num_frames_per_video = args.num_frames * args.num_segments
            num_views = n // num_frames_per_video

            views = [data[:, num_frames_per_video*t:num_frames_per_video*(t+1), :, :, :] for t in range(num_views)]

            outputs = [torch.nn.functional.softmax(model(view.reshape(-1, c, h, w))) for view in views]
            output = sum(outputs) / len(outputs)

            top1_acc, top5_acc = accuracy(output, label, topk=(1, 5))
            top1.update(top1_acc.item())
            top5.update(top5_acc.item())

    print("Acc@1: {}, Acc@5: {}".format(top1.avg, top5.avg))


if __name__ == "__main__":
    main()
    # args = parser.parse_args()
    # writer = SummaryWriter()

    # transforms = torchvision.transforms.Compose([
    #     GroupResize(args.crop_size),
    #     GroupThreeCrop(args.crop_size),
    #     GroupToTensor(),
    #     # GroupBatchNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])

    # test_dataset = UCF101Dataset(args.data_path, args.val_anno_path, transforms=transforms, mode='test',
    #                               sample_strategy=args.sample_strategy, num_frames=args.num_frames,
    #                               sample_interval=args.sample_interval, num_segments=args.num_segments,
    #                               test_num_clips=args.test_num_clips, test_num_crops=args.test_num_crops,
    #                               crop_size=args.crop_size, random_shift=args.random_shift)


    # data, label = test_dataset.__getitem__(222)
    # print(data.shape)
    # writer.add_images('mine 1 clips 3 crops', data)
    # writer.close()