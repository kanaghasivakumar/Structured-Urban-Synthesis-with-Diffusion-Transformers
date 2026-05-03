import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.dit import DiT

CITYSCAPES_COLORS = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156],
    [190,153,153], [153,153,153], [250,170, 30], [220,220,  0],
    [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60],
    [255,  0,  0], [  0,  0,142], [  0,  0, 70], [  0, 60,100],
    [  0, 80,100], [  0,  0,230], [119, 11, 32],
], dtype=np.uint8)

def colorize_mask(mask_np):
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(19):
        color[mask_np == cls_id] = CITYSCAPES_COLORS[cls_id]
    return color

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="inference_out")
    parser.add_argument("--model_path", type=str, default="best_model.pt")
    parser.add_argument("--data_root", type=str, default="processed_data")
    parser.add_argument("--ddim_steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DiT(
        img_size=128, patch_size=8, in_channels=3,
        num_classes=19, head_dim=64, num_heads=12, depth=12
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded {args.model_path}")

    mask_dir = os.path.join(args.data_root, args.split, "masks")
    img_dir  = os.path.join(args.data_root, args.split, "images")
    os.makedirs(args.out_dir, exist_ok=True)
    indiv_dir = os.path.join(args.out_dir, "generated")
    os.makedirs(indiv_dir, exist_ok=True)

    fnames = sorted(os.listdir(mask_dir))[:args.n]

    for fname in fnames:

        mask_pil = Image.open(os.path.join(mask_dir, fname)).convert("L")
        mask_pil = TF.resize(mask_pil, [128, 128], interpolation=TF.InterpolationMode.NEAREST)
        mask_np  = np.array(mask_pil)
        mask_t   = torch.from_numpy(mask_np).long().unsqueeze(0).to(device)

        sample = model.p_sample(mask_t, device=device, ddim_steps=args.ddim_steps)
        gen_np = (sample.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        gen_np = (gen_np * 255).astype(np.uint8)

        color_mask = colorize_mask(mask_np)

        real_path = os.path.join(img_dir, fname)
        if os.path.exists(real_path):
            real_pil = Image.open(real_path).convert("RGB").resize((128, 128), Image.BILINEAR)
            real_np  = np.array(real_pil)
            side_by_side = np.concatenate([color_mask, real_np, gen_np], axis=1)
            label = "mask | real | generated"
        else:
            side_by_side = np.concatenate([color_mask, gen_np], axis=1)
            label = "mask | generated"

        Image.fromarray(side_by_side).save(os.path.join(args.out_dir, fname))
        Image.fromarray(gen_np).save(os.path.join(indiv_dir, fname))
        print(f"  saved ({label})")


if __name__ == "__main__":
    main()