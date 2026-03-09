import torch
import torch.nn.functional as F


def preprocess_for_classifier(slice_img):
    t = torch.from_numpy(slice_img).float()
    t = t.unsqueeze(0).unsqueeze(0)
    t = F.interpolate(
        t,
        size=(128,128),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    t.clamp_(-1000, 1000).div_(1000.0)
    return t