import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

class TextConditioner(nn.Module):
    def __init__(
        self,
        *,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x
