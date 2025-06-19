import torch
from torch.nn import Module

from .tiny import TinyRoMa
from .xfeat_model import XFeatModel


def tiny_roma_v1_outdoor_model() -> Module:
    xfeat = XFeatModel()
    model = TinyRoMa(xfeat, freeze_xfeat=False, exact_softmax=False)
    model.load_state_dict(torch.load("weights/tiny_roma_v1_outdoor.pth"))
    return model
