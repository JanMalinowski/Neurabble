import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms, utils
from skimage import io, transform
import numpy as np
import cv2
import albumentations as A
from .dataset import DobbleDataset
from .engine import Engine
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tfms = A.Compose(
        [   A.Rotate(p=1.0),
            A.Blur(p=0.2),
            A.CoarseDropout(max_holes=20, max_height=70, max_width=70),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.HueSaturationValue(),
            A.Resize(1024, 1024, always_apply=True),
        ]
    )


    valid_tfms = A.Compose(
        [
            A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            A.Resize(1024, 1024, always_apply=True),
        ]
    )

    trainset = DobbleDataset(
        pkl_file="input/detecting_common_element/train_pairs.pkl",
        root_dir="input/images/",
        transform=train_tfms,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )

    validset = DobbleDataset(
        pkl_file="input/detecting_common_element/test_pairs.pkl",
        root_dir="input/images",
        transform=valid_tfms,
    )
    validloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 57)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loss = Engine.train(trainloader, model, optimizer, device)
    torch.save(model.state_dict(), "models/detecting_common_element/neurabble.pth")

    valid_preds, valid_targets = Engine.evaluate(validloader, model, device)

    category_pred = list()
    for i in range(len(valid_preds)):
        category_pred.append(valid_preds[i].index(max(valid_preds[i])))

    print("Validation set accuracy is: ", accuracy_score(category_pred, valid_targets))
