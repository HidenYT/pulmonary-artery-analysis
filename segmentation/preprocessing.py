import torch
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize
from utils.cv import normalize_image, window_level


WINDOW_LEVEL = (350, 40)

def preprocess_for_segmentation(slice_img: np.ndarray):

    # slice_img = window_level(slice_img, *WINDOW_LEVEL)
    slice_img = normalize_image(slice_img)
    slice_img = resize(slice_img, (256,256), mode='constant', anti_aliasing=True)

    t = torch.from_numpy(slice_img).float()

    # t = F.interpolate(
    #     t,
    #     size=(256,256),
    #     mode="constant",
    #     align_corners=True
    # )

    t = t.unsqueeze(0).unsqueeze(0)

    return t