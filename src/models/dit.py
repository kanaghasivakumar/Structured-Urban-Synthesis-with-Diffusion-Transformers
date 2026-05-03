import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torch.utils.checkpoint import checkpoint

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class MaskEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.num_classes = num_classes

    def forward(self, mask):
        valid = (mask >= 0) & (mask < self.num_classes)  
        one_hot = F.one_hot(mask.clamp(0, self.num_classes - 1), num_classes=self.num_classes).float()
        one_hot[~valid] = 0.0  
        pooled = one_hot.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).float().unsqueeze(-1).clamp(min=1)
        return self.proj(pooled)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        norm_x = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(1) * attn_out

        norm_x = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x)
        return x

class DiT(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_channels=3,
        num_classes=19,
        head_dim=64,      
        num_heads=6,
        depth=6,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        hidden_size = num_heads * head_dim
        self.hidden_size = hidden_size

        self.x_embedder = nn.Conv2d(in_channels + num_classes, hidden_size, kernel_size=patch_size, stride=patch_size)

        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(self._sinusoidal_pos_embed(num_patches, hidden_size), requires_grad=False)

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_shift_scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

        self._init_weights()

    @staticmethod
    def _sinusoidal_pos_embed(num_patches, hidden_size):
        pos = torch.arange(num_patches).unsqueeze(1).float()
        dim = torch.arange(0, hidden_size, 2).float()
        angles = pos / torch.pow(10000, dim / hidden_size)
        pe = torch.zeros(1, num_patches, hidden_size)
        pe[0, :, 0::2] = torch.sin(angles)
        pe[0, :, 1::2] = torch.cos(angles)
        return pe

    def _init_weights(self):
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(self, x, t, mask):

        safe_mask = mask.clamp(0, self.num_classes - 1)
        mask_onehot = F.one_hot(safe_mask, num_classes=self.num_classes).float()
        mask_onehot = mask_onehot.permute(0, 3, 1, 2) 
        ignore = (mask < 0) | (mask >= self.num_classes)
        mask_onehot[ignore.unsqueeze(1).expand_as(mask_onehot)] = 0.0
        x_input = torch.cat([x, mask_onehot], dim=1)
        x_seq = rearrange(self.x_embedder(x_input), 'b c h w -> b (h w) c')
        x_seq = x_seq + self.pos_embed

        c = self.t_embedder(t)

        for block in self.blocks:
            if self.training:
                x_seq = checkpoint(block, x_seq, c, use_reentrant=False)
            else:
                x_seq = block(x_seq, c)

        shift, scale = self.final_shift_scale(c).chunk(2, dim=1)
        x_seq = self.final_norm(x_seq) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x_seq = self.final_linear(x_seq)

        h = w = self.img_size // self.patch_size
        p = self.patch_size
        return rearrange(x_seq, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p)
    
    @torch.no_grad()

    def p_sample(self, mask, device, T=1000, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        b = mask.shape[0]
        x = torch.randn(b, self.in_channels, self.img_size, self.img_size, device=device)

        for t_val in reversed(range(T)):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)

            pred_noise = self(x, t, mask)

            ac  = alpha_cumprod[t_val]
            acp = alpha_cumprod_prev[t_val]
            beta = betas[t_val]
            alpha = alphas[t_val]

            x0_pred = (x - (1 - ac).sqrt() * pred_noise) / ac.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            mean = (acp.sqrt() * beta / (1 - ac)) * x0_pred + \
                (alpha.sqrt() * (1 - acp) / (1 - ac)) * x

            if t_val == 0:
                x = mean
            else:
                variance = beta * (1 - acp) / (1 - ac)
                x = mean + variance.sqrt() * torch.randn_like(x)

        return x  