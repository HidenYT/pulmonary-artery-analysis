import torch
from classification.preprocessing import preprocess_for_classifier
import matplotlib.pyplot as plt


def classify_slices(volume, model, device, threshold=0.5):

    positive_indices = []

    with torch.no_grad():

        for i, slice_img in enumerate(volume):
            inp = preprocess_for_classifier(slice_img).to(device)

            logits = model(inp)
            pred = torch.sigmoid(logits) > threshold

            if pred:
                positive_indices.append(i)

    return positive_indices