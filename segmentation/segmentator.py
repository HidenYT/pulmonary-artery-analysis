import torch
import numpy as np
import torch.nn.functional as F
from segmentation.preprocessing import preprocess_for_segmentation


def segment_slices(volume, indices, model, device):

    masks = []

    with torch.no_grad():

        for idx in indices:

            slice_img = volume[idx]

            inp = preprocess_for_segmentation(slice_img).to(device)

            pred = model(inp)

            pred = torch.sigmoid(pred)

            pred = pred.cpu()

            mask = F.interpolate(
                pred,
                size=slice_img.shape,
                mode="bilinear",
                align_corners=False
            )

            mask = mask.squeeze().numpy()

            mask = (mask > 0.5).astype(np.uint8)

            masks.append((idx, mask))

    return masks