import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    def __init__(self, mask, p):
        super().__init__()
        self.mask = mask
        self.p = p

    def forward(self, x):
        return 1. / self.p * x * self.mask.squeeze()
