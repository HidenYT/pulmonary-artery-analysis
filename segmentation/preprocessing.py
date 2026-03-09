import torch
import numpy as np
from skimage.transform import resize
from utils.cv import normalize_image, window_level


WINDOW_LEVEL = (350, 40)

def preprocess_for_segmentation(slice_img: np.ndarray):
    # window, level = WINDOW_LEVEL
    window, level = 2000, 0
    slice_img = window_level(slice_img, window=window, level=level)
    slice_img = normalize_image(slice_img, min=level-window/2, max=level+window/2)
    slice_img = resize(slice_img, (256,256), mode='constant', anti_aliasing=True)

    t = torch.from_numpy(slice_img).float()

    t = t.unsqueeze(0).unsqueeze(0)

    return t