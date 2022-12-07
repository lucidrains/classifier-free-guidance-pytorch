import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from classifier_free_guidance_pytorch.open_clip import OpenClipAdapter
from classifier_free_guidance_pytorch.t5 import t5_encode_text

class TextConditioner(nn.Module):
    def __init__(
        self,
        *,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x
