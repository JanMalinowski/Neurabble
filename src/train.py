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
from wtfml.utils import EarlyStopping

if __name__ == "__main__":
    # Defining augmentations
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # Training augmentations
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

    # Validation augmentations
    valid_tfms = A.Compose(
        [
            A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            A.Resize(1024, 1024, always_apply=True),
        ]
    )
    # Training set
    trainset = DobbleDataset(
        pkl_file="input/detecting_common_element/train_pairs.pkl",
        root_dir="input/images/",
        transform=train_tfms,
    )
    # Train loader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )
    # Validation set
    validset = DobbleDataset(
        pkl_file="input/detecting_common_element/test_pairs.pkl",
        root_dir="input/images",
        transform=valid_tfms,
    )
    # Validation loader
    validloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    
    # Initializing the model
    model = models.resnet18(pretrained=True)
    # Since we need to classify 57 categories we change
    # the last layer
    model.fc = nn.Linear(512, 57)

    # If the model already exists we load it in.
    if os.path.isfile('models/detecting_common_element/neurabble.pth'):
        model.load_state_dict(torch.load('models/detecting_common_element/neurabble.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    es = EarlyStopping(patience=5, mode="max")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Since we use early we the number for the range should
    #  be pretty big in order to be sure that the model can
    # fit properly
    for epoch in range(50):
        # Engine is a utility class for training and evaluating the 
        # model
        train_loss = Engine.train(trainloader, model, optimizer, device)

        valid_preds, valid_targets = Engine.evaluate(validloader, model, device)

        # Calculating the accuracy of our model
        category_pred = list()
        for i in range(len(valid_preds)):
            category_pred.append(valid_preds[i].index(max(valid_preds[i])))
        accuracy = accuracy_score(category_pred, valid_targets)

        print(f"Epoch: {epoch} accuracy: {accuracy}")

        # Scheduler step
        scheduler.step(accuracy)
        # Early stopping step
        es(accuracy, model, model_path=f"models/detecting_common_element/neurabble.pth")
        
        if es.early_stop:
            print("Early stopping")
            break

