import os
import time
import datetime
import shutil
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from opts import parser
from tsn import TSN
from dataset import UCF101Dataset
from utils import AverageMeter, accuracy
from logger import Logger
from transforms import *

best_top1_acc = 0
iteration = 0

def main():
    global best_top1_acc, iteration, logger, args

    args = parser.parse_args()

    dir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, dir_name, "runs"))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, dir_name)):
        os.mkdir(os.path.join(args.output_dir, dir_name))
    if not os.path.exists(os.path.join(args.output_dir, dir_name, "logs")):
        os.mkdir(os.path.join(args.output_dir, dir_name, "logs"))
    if not os.path.exists(os.path.join(args.output_dir, dir_name, "checkpoints")):
        os.mkdir(os.path.join(args.output_dir, dir_name, "checkpoints"))

    log_name = args.backbone + ".log"
    logger = Logger(os.path.join(args.output_dir, dir_name, "logs", log_name))

    logger.info(args)

    model = TSN(num_classes=args.num_classes, num_frames=args.num_segments * args.num_frames,
                backbone=args.backbone, consensus_type=args.consensus_type, dropout=args.dropout,
                shift_div=args.shift_div, shift_mode=args.shift_mode, pretrained=args.pretrained)

    model = nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.load_from_github:
        model = load_state_dict(model, args.state_dict_path)

    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_steps)

    train_transforms = torchvision.transforms.Compose([
        GroupRandomMultiScaleCrop(args.crop_size),
        GroupResize(args.crop_size),
        GroupRandomHorizontalFlip(),
        GroupToTensor(),
        GroupBatchNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transforms = torchvision.transforms.Compose([
        GroupResize(args.crop_size),
        GroupCenterCrop(args.crop_size),
        GroupToTensor(),
        GroupBatchNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = UCF101Dataset(args.data_path, args.train_anno_path, transforms=train_transforms, mode='train',
                                  sample_strategy=args.sample_strategy, num_frames=args.num_frames,
                                  sample_interval=args.sample_interval, num_segments=args.num_segments,
                                  test_num_clips=args.test_num_clips, test_num_crops=args.test_num_crops,
                                  crop_size=args.crop_size, random_shift=args.random_shift)

    val_dataset = UCF101Dataset(args.data_path, args.val_anno_path, transforms=val_transforms, mode='val',
                                sample_strategy=args.sample_strategy, num_frames=args.num_frames,
                                sample_interval=args.sample_interval, num_segments=args.num_segments,
                                test_num_clips=args.test_num_clips, test_num_crops=args.test_num_crops,
                                crop_size=args.crop_size, random_shift=args.random_shift)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, writer)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            top1_acc = validate(val_loader, model, criterion, optimizer, epoch, writer)
            is_best = top1_acc > best_top1_acc
            best_top1_acc = max(top1_acc, best_top1_acc)

        lr_scheduler.step()

        save_checkpoint({
                'epoch': epoch + 1,
                'backbone': args.backbone,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_top1_acc': best_top1_acc,
            }, is_best, epoch + 1, dir_name)

def train(train_loader, model, criterion, optimizer, epoch, writer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    last_time = time.time()

    for i, (data, label) in enumerate(train_loader):
        global iteration
        iteration += 1
        # data loading time
        data_time.update(time.time() - last_time)
        data = data.cuda()  # (N, T, C, H, W)
        label = label.cuda()

        n, t, c, h, w = data.shape
        data = data.view(-1, c, h, w)

        output = model(data)
        loss = criterion(output, label)

        top1_acc, top5_acc = accuracy(output.data, label, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(top1_acc.item(), data.size(0))
        top5.update(top5_acc.item(), data.size(0))

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - last_time)
        last_time = time.time()

        message = ('Epoch: [{}][{}/{}], lr: {lr: .5f}\t'
                   'Batch Training Time: {batch_time.value: .3f}\t'
                   'Data Loading Time: {data_time.value: .3f}\t'
                   'Loss: {loss.avg: .4f} ({loss.value: .4f})\t'
                   'Acc@1: {top1.avg: .3f} ({top1.value: .3f})\t'
                   'Acc@5: {top5.avg: .3f} ({top5.value: .3f})\t'
        ).format(epoch, i, len(train_loader), lr=optimizer.param_groups[-1]['lr'],
                 batch_time=batch_time, data_time=data_time, loss=losses,
                 top1=top1, top5=top5
        )
        logger.info(message)

        writer.add_scalar('loss/train/iteration', losses.avg, iteration)
        writer.add_scalar('top1_acc/train/iteration', top1.avg, iteration)
        writer.add_scalar('top5_acc/train/iteration', top5.avg, iteration)

    writer.add_scalar('loss/train/epoch', losses.avg, epoch)
    writer.add_scalar('top1_acc/train/epoch', top1.avg, epoch)
    writer.add_scalar('top5_acc/train/epoch', top5.avg, epoch)
    writer.add_scalar('learning rate', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    last_time = time.time()

    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            data = data.cuda()
            label = label.cuda()

            n, t, c, h, w = data.shape
            data = data.view(-1, c, h, w)

            output = model(data)
            loss = criterion(output, label)

            top1_acc, top5_acc = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(top1_acc.item(), data.size(0))
            top5.update(top5_acc.item(), data.size(0))

    message = ('Testing Results: Acc@1 {top1.avg:.3f}\t Acc@5 {top5.avg:.3f}\t Loss {loss.avg:.5f}'.format(top1=top1, top5=top5, loss=losses))
    logger.info("================================================================")
    logger.info(message)

    writer.add_scalar('loss/val/epoch', losses.avg, epoch)
    writer.add_scalar('top1_acc/val/epoch', top1.avg, epoch)
    writer.add_scalar('top5_acc/val/epoch', top5.avg, epoch)

    return top1.avg


def load_state_dict(model, state_dict_path):
    logger.info("Loading pretrained model from {}".format(state_dict_path))
    pretrained_model = torch.load(state_dict_path)

    model_names = list(model.state_dict())
    pretrained_model_names = list(pretrained_model['state_dict'].keys())

    tmp = model.state_dict()
    # expect the final fully connected layer's weight and bias
    for i in range(len(pretrained_model_names) - 2):
        tmp[model_names[i]] = pretrained_model['state_dict'][pretrained_model_names[i]]

    model.load_state_dict(tmp)

    return model


def save_checkpoint(state, is_best, epoch, dir_name):
    file_name = "{}_{}_{}_{}_{}_{}.pth.tar".format(args.backbone, epoch, args.sample_strategy, args.num_frames, args.sample_interval, args.num_segments)
    torch.save(state, os.path.join(args.output_dir, dir_name, "checkpoints", file_name))
    logger.info("{} has been saved in {}".format(file_name, os.path.join(args.output_dir, dir_name, "checkpoints", file_name)))
    if is_best:
        shutil.copyfile(os.path.join(args.output_dir, dir_name, "checkpoints", file_name), os.path.join(args.output_dir, dir_name, "checkpoints", 'best.pth.tar'))
        logger.info("Saved best acc@1 checkpoint")
    logger.info("================================================================")


if __name__ == "__main__":
    main()