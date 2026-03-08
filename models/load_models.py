import torch


def load_classifier(model, weights_path, device):

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model


def load_segmentator(model, weights_path, device):

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model