import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class CityscapesDiTDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) 

        if self.transform:
            image = self.transform(image)
    
        mask_tensor = torch.from_numpy(np.array(mask)).long()

        return image, mask_tensor