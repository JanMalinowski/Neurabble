from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms, utils
import random
import cv2
import os
import torch
import pandas as pd
import numpy as np
import cv2
import random
import albumentations as A


class DobbleDataset(Dataset):
    """Dobble dataset."""

    def __init__(self, pkl_file, root_dir, transform=None):
        """
        Args:
            pkl_file (string): Path to the pkl dataframe.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_pickle(pkl_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name1 = os.path.join(self.root_dir, self.df.loc[idx, "image1"])
        img_name2 = os.path.join(self.root_dir, self.df.loc[idx, "image2"])
        target = self.df.loc[idx, "common_element"]

        img1 = read_img(img_name1)
        img2 = read_img(img_name2)

        if self.transform:
            img1 = self.transform(image=img1)["image"]
            img2 = self.transform(image=img2)["image"]

        if random.randint(0, 10000) % 2 == 0:
            img = cv2.hconcat([img1, img2])
        else:
            img = cv2.vconcat([img1, img2])

        img = A.Resize(224, 224)(image=img)["image"]
        img = torchvision.transforms.ToTensor()(img)

        sample = {"image": img, "target": target}

        return sample
