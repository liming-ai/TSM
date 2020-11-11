import numpy as np

from decord import VideoReader, cpu

frame_inds = np.array([0, 3, 7, 10, 14, 17, 20, 24, 27, 31])
print(frame_inds)

frame_inds = frame_inds[:, None] + np.arange(8)[None, :] * 8
print(frame_inds)

frame_inds = np.concatenate(frame_inds)
print(frame_inds)

frame_inds = frame_inds.reshape((-1, 8))
print(frame_inds)

print(frame_inds.shape)


vr = VideoReader("data/videos/Bowling/v_Bowling_g01_c07.avi", ctx=cpu(0))
print(len(vr))

safe_inds = frame_inds < len(vr)
unsafe_inds = 1 - safe_inds
last_ind = np.max(safe_inds * frame_inds, axis=1)
new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
frame_inds = new_inds

print(frame_inds)
print(frame_inds.shape)

clips = [(vr.get_batch(clip_indices)).asnumpy() for clip_indices in frame_inds]
cc = np.stack(clips)
print(cc.shape)

import pdb; pdb.set_trace()

