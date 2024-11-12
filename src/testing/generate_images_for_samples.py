from pathlib import Path

import tqdm
import numpy as np
import cv2


root_in_dir = './data/samples'
out_ext = '.jpg'



root_in_dir = Path(root_in_dir)
root_out_dir = root_in_dir.with_name('sampled_images')

for in_path in tqdm.tqdm(list(root_in_dir.rglob('*.npy'))):

    out_path = root_out_dir / in_path.relative_to(root_in_dir).with_suffix(out_ext)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(in_path)
    data = np.clip(255 * data, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), data)
