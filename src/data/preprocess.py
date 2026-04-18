import os
import time
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import numpy as np

IGNORE_ID = 255
ID_MAP = {0: IGNORE_ID, 1: IGNORE_ID, 2: IGNORE_ID, 3: IGNORE_ID, 4: IGNORE_ID, 
          5: IGNORE_ID, 6: IGNORE_ID, 7: 0, 8: 1, 9: IGNORE_ID, 10: IGNORE_ID, 
          11: 2, 12: 3, 13: 4, 14: IGNORE_ID, 15: IGNORE_ID, 16: IGNORE_ID, 
          17: 5, 18: IGNORE_ID, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 
          25: 12, 26: 13, 27: 14, 28: 15, 29: IGNORE_ID, 30: IGNORE_ID, 
          31: 16, 32: 17, 33: 18, -1: IGNORE_ID}

def process_single_pair(img_path, mask_path, output_dir, size=(128, 128)):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(size, Image.BILINEAR)
        
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        mapped_mask = np.vectorize(ID_MAP.get)(mask_np, IGNORE_ID).astype(np.uint8)
        mask = Image.fromarray(mapped_mask)
        mask = mask.resize(size, Image.NEAREST) # Nearest for labels!

        base_name = Path(img_path).stem.replace('_leftImg8bit', '')
        img.save(os.path.join(output_dir, 'images', f"{base_name}.png"))
        mask.save(os.path.join(output_dir, 'masks', f"{base_name}.png"))
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def parallel_preprocess(raw_data_dir, output_dir, workers=8):
    start_time = time.time()
    
    end_time = time.time()
    print(f"PROFILING: Processed {count} images in {end_time - start_time:.2f}s using {workers} workers.")