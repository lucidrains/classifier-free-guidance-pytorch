from collections import namedtuple

from beartype import beartype
from typing import List

import torch
from torch import nn, einsum
import torch.nn.functional as F

import open_clip
from classifier_free_guidance_pytorch.tokenizer import tokenizer

EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings'])

def l2norm(t):
    return F.normalize(t, dim = -1)

class OpenClipAdapter():
    def __init__(
        self,
        name = 'ViT-B/32',
        pretrained = 'laion400m_e32'
    ):
        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained = pretrained)

        self.clip = clip
        self.tokenizer = tokenizer

        self.eos_id = 49407

        text_attention_final = self.find_layer('ln_final')
        self._dim_latent = text_attention_final.weight.shape[0]

        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.handle()

    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def max_text_len(self):
        return 77

    @torch.no_grad()
    @beartype
    def embed_text(
        self,
        text: List[str]
    ):
        text = self.tokenizer.tokenize(text)

        text = text[..., :self.max_text_len]

        is_eos_id = (text == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        assert not self.cleared

        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float())
