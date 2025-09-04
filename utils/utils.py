import os
import random
import torch
import numpy as np
import skimage as ski
from torchvision.transforms.functional import resize
import json

label_map = {
    0: 'RCM',  1: 'DCM', 2: 'ARVC', 3: 'NDLVC', 4: 'HCM'
}


def video_sample(view_path, num_frames):
    stride = 2
    imgs_path = os.listdir(view_path)

    def sort_fn(x):
        try:
            return int(x.split(".")[0])
        except:
            print(view_path, x)
            return 500
    imgs_path = sorted(imgs_path, key=sort_fn)
    frame_t0 = 0
    frame_t1 = len(imgs_path) - 1
    if frame_t1 - frame_t0 < num_frames * stride:
        idxes = np.arange(frame_t0, frame_t1, stride)
    else:
        space = frame_t1 - num_frames * stride
        start_idx = random.randint(0, space)
        idxes = np.arange(start_idx, frame_t1, stride)[:16]
    imgs = []
    for idx in idxes:
        img_path = os.path.join(view_path, imgs_path[idx])
        if not os.path.exists(img_path):
            raise RuntimeError(f"{img_path} Not fond file！")
        img = ski.io.imread(img_path)
        if (len(img.shape) == 2):
            img = torch.from_numpy(img)[None, ...]
            img = img.repeat((3, 1, 1))
        else:
            img = torch.from_numpy(img)
            img = img.permute([2, 0, 1])
        imgs.append(img)
    if len(imgs) < num_frames:
        for _ in range(num_frames - len(imgs)):
            imgs.append(torch.zeros_like(img))
    return torch.stack(imgs)


def load_view(view_data_path, num_frames=16):
    mean = 27.9297816
    std = 49.908389
    view_data = video_sample(view_data_path, num_frames)
    view_data = view_data.float().permute([1, 0, 2, 3])
    view_data = resize(view_data, [224, 224], antialias=None)
    view_data.sub_(mean).div_(std)
    return view_data[None, ...]
