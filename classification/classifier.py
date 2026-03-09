import torch
from classification.preprocessing import preprocess_for_classifier
import numpy as np

from classification.utils import keep_longest_run


def classify_slices(volume, model, device, threshold=0.5):
    positive_indices = []
    predictions = []
    with torch.no_grad():
        for slice_img in volume:
            inp = preprocess_for_classifier(slice_img).to(device)
            logits = model(inp)
            pred = torch.sigmoid(logits) > threshold
            predictions.append(bool(pred))

    positive_indices = keep_longest_run(torch.tensor(predictions))
    return torch.argwhere(positive_indices)