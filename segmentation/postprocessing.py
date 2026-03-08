import numpy as np
from skimage import morphology, measure


def clean_mask(mask):

    mask = morphology.remove_small_objects(mask.astype(bool), 100)

    labeled = measure.label(mask)

    if labeled.max() == 0:
        return mask

    regions = measure.regionprops(labeled)

    largest = max(regions, key=lambda r: r.area)

    cleaned = labeled == largest.label

    return cleaned.astype(np.uint8)


def find_largest_mask(masks):

    best = None
    best_area = 0
    best_index = None

    for idx, mask in masks:

        area = mask.sum()

        if area > best_area:
            best_area = area
            best = mask
            best_index = idx

    return best_index, best