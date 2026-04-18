import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

ID_MAP = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 
          22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 
          32: 17, 33: 18}

def test_single_image():
    raw_img_path = "data/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
    raw_mask_path = "data/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png"
    
    if not os.path.exists(raw_img_path):
        print(f"FAILED: Could not find image at {raw_img_path}")
        return

    print("SUCCESS: Found image and mask.")

    img = Image.open(raw_img_path).convert('RGB').resize((128, 128), Image.BILINEAR)
    
    mask = Image.open(raw_mask_path)
    mask_np = np.array(mask)
    mapped_mask = np.vectorize(lambda x: ID_MAP.get(x, 255))(mask_np).astype(np.uint8)
    mask_resized = Image.fromarray(mapped_mask).resize((128, 128), Image.NEAREST)

    os.makedirs("test_out", exist_ok=True)
    img.save("test_out/test_img.png")
    mask_resized.save("test_out/test_mask.png")
    print("DONE: Check the 'test_out' folder for the resized results.")

if __name__ == "__main__":
    test_single_image()