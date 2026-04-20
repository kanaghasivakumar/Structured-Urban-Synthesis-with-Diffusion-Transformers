import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import wandb
import os

from data.dataset import CityscapesDiTDataset
from models.dit import DiT

def train():
    default_config = {
        "batch_size": 2,    
        "lr": 1e-4,
        "epochs": 1,
        "patch_size": 8,
        "depth": 6,
        "num_heads": 4
    }

    with wandb.init(
        project="structured-urban-synthesis", 
        config=default_config
    ):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_ds = CityscapesDiTDataset(root_dir="processed_data", split='train', transform=transform)
        train_loader = DataLoader(
            train_ds, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )

        model = DiT(
            img_size=128,
            patch_size=config.patch_size,
            in_channels=3,
            num_classes=19,
            depth=config.depth,
            num_heads=config.num_heads
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for step, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device)
                
                t = torch.randint(0, 1000, (images.shape[0],), device=device).long()
                noise = torch.randn_like(images)
                
                noisy_images = images + noise
                
                predicted_noise = model(noisy_images, t, masks)
                loss = F.mse_loss(predicted_noise, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "step": epoch * len(train_loader) + step
                })

            avg_epoch_loss = epoch_loss / len(train_loader)
            wandb.log({"epoch_loss": avg_epoch_loss})

if __name__ == "__main__":
    train()