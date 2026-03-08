import numpy as np


def window_level(img: np.ndarray, window: int, level: int):
    low = level - window/2
    high = level + window/2
    return img.clip(low, high)


def normalize_image(img):
    return (img-img.min())/(img.max()-img.min())