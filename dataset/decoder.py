import os
import torch
import random
import numpy as np
from decord import VideoReader
from decord import cpu, gpu

def decode(video_name, num_clips, cfg):
    """
    Decord the video and return `num_clips` clips.

    Args:
        video_name (string): Video file name.
        num_clips (int): Number of clips to sample.
        cfg (cfgNode): configs

    Returns:
        torch.Tensor: (num_clips * cfg.num_frames * height * width * 3)
    """
    video_path = os.path.join(cfg.root_path, video_name)
    vr = VideoReader(video_path, ctx=gpu(0))

    fps = int(vr.get_avg_fps())  # current fps

    # number of frames of a clip
    # 一个 clip 中帧的数目
    if cfg.target_fps is not None:
        clip_size = cfg.sampling_interval * cfg.num_frames / cfg.target_fps * fps
    else:
        clip_size = cfg.sampling_interval * cfg.num_frames

    if cfg.mode in ["train", "val"]:
        frame_inds = get_train_clips(vr, num_clips, clip_size)
    elif cfg.mode in ["test"]:
        frame_inds = get_test_clips(vr, num_clips, clip_size)

    # frame_inds is start indexes of each clip, now get all the selected frames' indexes.
    frame_inds = frame_inds[:, None] + \
                 np.arange(cfg.num_frames)[None, :] * cfg.sampling_interval
    frame_inds = np.concatenate(frame_inds)
    # Adding temporal jitter
    if cfg.temporal_jitter:
        perframe_offsets = np.random.randint(cfg.sampling_interval, size=len(frame_inds))
        frame_inds += perframe_offsets

    # frame_inds shape: (num_clips, num_frames)
    frame_inds = frame_inds.reshape((-1, cfg.num_frames))

    # Deal with frame index out of bounds
    safe_inds = frame_inds < len(vr)
    unsafe_inds = 1 - safe_inds
    last_ind = np.max(safe_inds * frame_inds, axis=1)
    new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
    frame_inds = new_inds

    clips = [(vr.get_batch(clip_indices)).asnumpy() for clip_indices in frame_inds]

    return torch.from_numpy(np.stack(clips))

def get_train_clips(record, num_clips, clip_size):
    """
    For sparse sampling, `cfg.num_frames` and `cfg.sampling_interval` are equal to 1, means each clip sample 1 frame, so return `num_clips` frames' indices.

    For dense sampling, return `num_clips` indices, each indices is the start index of a clip, each clip has `cfg.num_frames` * `cfg.sampling_interval` frames.

    Args:
        record (VideoReader): Video container.
        num_clips (int): Number of clips to sample from the video.
        clip_size (int): Number of frames per clip.

    Return:
        np.ndarray: Sampled frame indices in train mode.
    """
    video_length = len(record)
    # number of frames between two clips
    average_duration = (video_length - clip_size + 1) // num_clips


    if average_duration > 0:
        indices = np.multiply(list(range(num_clips)), average_duration)
        indices += np.random.randint(average_duration, size=num_clips)
    # video_length > clip_size, number of different clips < num_clips
    elif video_length > max(num_clips, clip_size):
        indices = np.sort(np.random.randint(
            video_length - clip_size + 1, size=num_clips
        ))
    # video_length == clip_size NOTE: think why???
    elif average_duration == 0:
        ratio = (video_length - clip_size + 1.0) / num_clips
        indices = np.round(np.arange(num_clips) * ratio)
    else:
        indices = np.zeros((num_clips, ))

    return indices

def get_test_clips(record, num_clips, clip_size):
    """
    For sparse sampling, `cfg.num_frames` and `cfg.sampling_interval` are equal to 1, means each clip sample 1 frame, so return `num_clips` frames' indices.

    For dense sampling, return `num_clips` indices, each indices is the start index of a clip, each clip has `cfg.num_frames` * `cfg.sampling_interval` frames.

    Args:
        record (VideoReader): Video container.
        num_clips (int): Number of clips to sample from the video.
        clip_size (int): Number of frames per clip.

    Returns:
        numpy.array: Indices of clips
    """
    video_length = len(record)
    average_duration = (video_length - clip_size + 1) / float(num_clips)

    if video_length > clip_size - 1:
        base_indices = np.arange(num_clips) * average_duration
        indices = (base_indices + average_duration / 2.0).astype(np.int)
    else:
        indices = np.zeros((num_clips, ), dtype=np.int)

    return indices
