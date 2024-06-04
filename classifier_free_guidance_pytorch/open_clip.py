from beartype import beartype
from typing import List

import torch
from torch import nn, einsum
import torch.nn.functional as F

import open_clip
from classifier_free_guidance_pytorch.tokenizer import tokenizer

# constants

DEFAULT_CLIP_NAME = 'ViT-B-32'
DEFAULT_PRETRAINED_CLIP = 'laion400m_e32'

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

# adapter

class OpenClipAdapter():
    def __init__(
        self,
        name = DEFAULT_CLIP_NAME,
        pretrained = DEFAULT_PRETRAINED_CLIP,
        text_embed_pad_value = 0.
    ):
        name = default(name, DEFAULT_CLIP_NAME)
        pretrained = default(pretrained, DEFAULT_PRETRAINED_CLIP)

        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained = pretrained)

        self.clip = clip
        clip.eval()

        self.tokenizer = tokenizer
        self.text_embed_pad_value = text_embed_pad_value

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
        texts: List[str],
        return_text_encodings = False,
        output_device = None
    ):
        texts, max_length = self.tokenizer.tokenize(texts)
        texts = texts[..., :self.max_text_len]

        text_embeds = self.clip.encode_text(texts)

        texts = texts[..., :max_length]

        if not return_text_encodings:
            return l2norm(text_embeds).to(output_device)

        is_eos_id = (texts == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        text_mask = text_mask & (texts != 0)

        assert not self.cleared

        text_encodings = self.text_encodings[:, :max_length]
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], self.text_embed_pad_value)
        del self.text_encodings

        return text_encodings.float().to(output_device)
