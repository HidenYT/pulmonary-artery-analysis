import numpy as np


def window_level(img: np.ndarray, window: int, level: int):
    low = level - window/2
    high = level + window/2
    return img.clip(low, high)


def normalize_image(img, min=None, max=None):
    if min is None:
        min = img.min()
    if max is None:
        max = img.max()
    return (img-min)/(max-min)