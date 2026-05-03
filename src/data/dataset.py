import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms.functional as TF
import random

class CityscapesDiTDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, augment=False):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        self.files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, img_name))

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        if self.transform:
            image = self.transform(image)

        mask_tensor = torch.from_numpy(np.array(mask)).long()
        return image, mask_tensor