# Introduction
This is the code for [TSM: Temporal Shift Module for Efficient Video Understanding](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html) on CVPR 2019 and [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859) on ECCV 2016.

We only support for [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) rawframe dataset, but you can modify the [dataset.py](dataset.py) to add more dataset.

This repository is designed to help beginners better understand video classification, about how to write the latest models and understand the data pipelines.

**So we added as much detailed code comments as possible**, the basic code is from [official TSM implementation](https://github.com/mit-han-lab/temporal-shift-module), we only rewrite the code for brevity and easier understanding.

# Prerequisites
You should install all the required dependencies:
* [PyTorch](https://pytorch.org/) 1.0 or higher
* [TensorboardX](https://github.com/lanpa/tensorboardX)
* [tqdm](https://github.com/tqdm/tqdm)
* [scikit-learn](https://scikit-learn.org/stable/)

# Data Preparation
Here we only write code for rawframes of UCF101, please see [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ucf101/README.md) to prepare these frames.

The folder structure should be like this:
```
TSM
├── backbones
├── dataset.py
├── logger.py
├── opts.py
├── README.md
├── test.py
├── train.py
├── transforms.py
├── tsn.py
├── utils.py
├── data
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g01_c01
│   │   │   │   ├── ...
│   │   │   │   ├── v_YoYo_g25_c05

```

# Use
Before use the code, you should create a folder to store all the training logs and checkpoints:
```bash
mkdir output_dir
```

You can find all the needed parameters in [ots.py](opts.py), set the parameters you want and run the code for training directly:
```bash
python train.py
```
or run the code for testing:
```bash
python test.py
```
You can set these parameters in terminal too:
```bash
python train.py --backbone "resnet50_tsm" --batch_size 32 --epochs 25
```

Please read the code in [ots.py](opts.py) for more details.