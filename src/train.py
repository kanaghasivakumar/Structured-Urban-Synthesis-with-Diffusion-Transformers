import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import wandb
import argparse

from data.dataset import CityscapesDiTDataset
from models.dit import DiT

def make_ddpm_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return {
        'betas': betas,
        'sqrt_alpha_cumprod': torch.sqrt(alpha_cumprod),
        'sqrt_one_minus_alpha_cumprod': torch.sqrt(1.0 - alpha_cumprod),
    }


def q_sample(x0, t, schedule):
    """Forward diffusion: add noise according to timestep t."""
    s1 = schedule['sqrt_alpha_cumprod'][t].view(-1, 1, 1, 1)
    s2 = schedule['sqrt_one_minus_alpha_cumprod'][t].view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    return s1 * x0 + s2 * noise, noise

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    args = parser.parse_args()

    with wandb.init(project="structured-urban-synthesis", config=vars(args)):
        config = wandb.config
        lr_str, bs_str = config.lr_batch.split(',')
        config.update({"lr": float(lr_str), "batch_size": int(bs_str)}, allow_val_change=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_ds = CityscapesDiTDataset(root_dir="processed_data", split='train', transform=transform)
        val_ds   = CityscapesDiTDataset(root_dir="processed_data", split='val',   transform=transform)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

        model = DiT(
            img_size=128,
            patch_size=config.patch_size,
            in_channels=3,
            num_classes=19,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            depth=config.depth,
        ).to(device)

        schedule = make_ddpm_schedule(T=1000, device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

        total_steps = config.epochs * len(train_loader)
        warmup_steps = config.warmup_epochs * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler('cuda')

        best_val_loss = float('inf')
        patience, patience_counter = 10, 0
        global_step = 0

        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
                images = images.to(device)
                masks  = masks.to(device)

                t = torch.randint(0, 1000, (images.shape[0],), device=device).long()
                noisy, noise = q_sample(images, t, schedule)

                with torch.amp.autocast('cuda'):
                    pred_noise = model(noisy, t, masks)
                    loss = F.mse_loss(pred_noise, noise)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                wandb.log({
                    "batch_loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                })
                global_step += 1

            avg_train_loss = epoch_loss / len(train_loader)

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                    images = images.to(device)
                    masks  = masks.to(device)

                    t = torch.randint(0, 1000, (images.shape[0],), device=device).long()
                    noisy, noise = q_sample(images, t, schedule)

                    with torch.amp.autocast('cuda'):
                        pred_noise = model(noisy, t, masks)
                        loss = F.mse_loss(pred_noise, noise)

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            wandb.log({
                "epoch_loss": avg_train_loss,
                "val_loss":   avg_val_loss,
                "epoch":      epoch,
            })

            print(f"Epoch {epoch} | train={avg_train_loss:.4f} | val={avg_val_loss:.4f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stop at epoch {epoch}.")
                    break


if __name__ == "__main__":
    train()