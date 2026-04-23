import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import wandb
import os
import argparse

from data.dataset import CityscapesDiTDataset
from models.dit import DiT

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    args = parser.parse_args()

    sweep_config = vars(args)

    with wandb.init(
        project="structured-urban-synthesis", 
        config=sweep_config
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
            num_workers=8,
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

        scaler = torch.amp.GradScaler('cuda')

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
                
                with torch.amp.autocast('cuda'):
                    predicted_noise = model(noisy_images, t, masks)
                    loss = F.mse_loss(predicted_noise, noise)
                
                optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
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