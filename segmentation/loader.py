from segmentation.ml_model import SegNet
import torch.nn as nn
import torch

def load_segnet(weights_path: str, device):
    model_weight = torch.load(weights_path, weights_only=True)
    model = SegNet(1, 1)
    model.load_state_dict(model_weight)
    model.eval()
    model.to(device)
    return model