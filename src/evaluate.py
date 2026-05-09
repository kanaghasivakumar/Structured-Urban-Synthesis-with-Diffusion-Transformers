import torch
import numpy as np
from PIL import Image
import os, sys, argparse
from scipy import linalg
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.dit import DiT

def get_inception_features(img_dir, model, transform, device, limit=None):
    features = []
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    if limit:
        files = files[:limit]
    for fname in files:
        img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(x).squeeze(-1).squeeze(-1).cpu().numpy()
        features.append(feat)
    return np.concatenate(features, axis=0)

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--gen_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    real_feats = get_inception_features(args.real_dir, inception, transform, device, args.limit)
    gen_feats = get_inception_features(args.gen_dir, inception, transform, device, args.limit)

    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(0), np.cov(gen_feats, rowvar=False)

    fid = compute_fid(mu_r, sigma_r, mu_g, sigma_g)
    print(f"FID: {fid:.4f}")

if __name__ == "__main__":
    main()