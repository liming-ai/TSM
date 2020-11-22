import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")

parser.add_argument('--data_path', type=str, default="./data/rawframes")
parser.add_argument('--train_anno_path', type=str, default="./data/ucf101_train_split_1_rawframes.txt")
parser.add_argument('--val_anno_path', type=str, default="./data/ucf101_val_split_1_rawframes.txt")
parser.add_argument('--output_dir', type=str, default="./output_dir")

parser.add_argument('--sample_strategy', type=str, default="sparse")
parser.add_argument('--num_frames', type=int, default=1)
parser.add_argument('--sample_interval', type=int, default=1)
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--test_num_clips', type=int, default=2)
parser.add_argument('--test_num_crops', type=int, default=3)
parser.add_argument('--random_shift', type=bool, default=True)
parser.add_argument('--crop_size', type=int, default=224)

parser.add_argument('--backbone', type=str, default="resnet50_tsm")
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--consensus_type', type=str, default="avg")
parser.add_argument('--shift_div', type=int, default=8)
parser.add_argument('--shift_mode', type=str, default="residual")
parser.add_argument('--num_classes', type=int, default=101)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr_type', type=str, default='step')
parser.add_argument('--lr_steps', type=float, default=[50, 100])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=4)