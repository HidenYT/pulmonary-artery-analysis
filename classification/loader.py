from torchvision.models import resnet50
import torch.nn as nn
import torch

def load_resnet50(weights_path: str, device):
    model_weight = torch.load(weights_path, weights_only=True)
    model = resnet50()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(model_weight)
    model.eval()
    model.to(device)
    return model