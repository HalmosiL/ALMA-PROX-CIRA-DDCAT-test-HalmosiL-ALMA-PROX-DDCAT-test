import os
import time
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn

from model.pspnet import PSPNet_DDCAT
from util import dataset, transform, config

cv2.ocl.setUseOpenCL(False)

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

mean_origin = [0.485, 0.456, 0.406]
std_origin = [0.229, 0.224, 0.225]

def get_args():
    return {
        "classes": 19,
        "crop_h": 449,
        "crop_w": 449
    }

def attack_runer(input, target, model, attack):
    return attack(model=model, inputs=image, labels=attack_label_arr[k], targeted=False)

def net_process(model, image, target, mean, std=None, attack=None):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    target = torch.from_numpy(target).long()

    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)

    input = input.unsqueeze(0)
    target = target.unsqueeze(0)

    if True:
        flip = False
    else:
        flip = True

    if flip:
        input = torch.cat([input, input.flip(3)], 0)
        target = torch.cat([target, target.flip(2)], 0)

    if attack is not None:
        adver_input = attack_runer(
            input,
            target,
            model,
            attack
        )

        with torch.no_grad():
            output = model(adver_input)
    else:
        with torch.no_grad():
            output = model(input)

    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)

    output = F.softmax(output, dim=1)

    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]

    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    
    return output


def scale_process(model, image, target, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3, attack=None):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)

    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
        target = cv2.copyMakeBorder(target, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w, _ = image.shape

    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))

    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)

    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)

    count_crop = np.zeros((new_h, new_w), dtype=float)

    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()

            target_crop = target[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, target_crop, mean, std, attack)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction

def prediction(model, input, target, attack=None):
    args = get_args()

    input = np.squeeze(input.numpy(), axis=0)
    target = np.squeeze(target.numpy(), axis=0)

    image = np.transpose(input, (1, 2, 0))
    image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)

    h, w, _ = image.shape
    prediction = np.zeros((h, w, classes), dtype=float)

    for scale in scales:
        long_size = round(scale * base_size)
        new_h = long_size
        new_w = long_size
        if h > w:
            new_w = round(long_size/float(h)*w)
        else:
            new_h = round(long_size/float(w)*h)

        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        target_scale = cv2.resize(target.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        prediction += scale_process(
            model=model,
            image_scale=image_scale,
            target_scale=target_scale,
            classes=args.classes,
            crop_h=args.crop_h,
            crop_w=args.crop_w,
            h=h,
            w=w,
            mean=mean,
            std=std,
            attack=attack
        )

    return prediction