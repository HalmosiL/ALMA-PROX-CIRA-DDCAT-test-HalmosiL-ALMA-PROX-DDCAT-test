from forward import prediction
from model import load_model

import transforms as transform
from dataset import SemDataSplit

import numpy as np
import torch

def get_cityscapes_resized(root="", size=None, split="", num_images=None, batch_size=1):
    val_transform = transform.Compose(
        [transform.ToTensor()]
    )

    image_list_path = root + "/" + split + ".txt"

    loader = torch.utils.data.DataLoader(   
        dataset=SemDataSplit(
            split=split,
            data_root=root,
            data_list=image_list_path,
            transform=val_transform,
            num_of_images=num_images
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=True
    )

    return loader

device = "cpu"
model = load_model("../../../pretrain/cityscapes/pspnet/no_defense/train_epoch_400.pth", device)

loader = get_cityscapes_resized(
    root="/home/developer/Desktop/CIRA-AP-Kutatás/CitySpace/Data/ok/",
    size=None,
    split="val",
    num_images=None,
    batch_size=1
)

input, target = next(iter(loader))

input = input[0]
target = target[0]

pred = prediction(
    model=model,
    input=input,
    target=target,
    attack=None
)