from typing import List
from beartype import beartype

import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config

transformers.logging.set_verbosity_error()

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# config

MAX_LENGTH = 256

DEFAULT_T5_NAME = 'google/t5-v1_1-base'

T5_CONFIGS = {}

# singleton globals

def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer

def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model

def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model

# encoding text

def t5_encode_text(texts, name = DEFAULT_T5_NAME, output_device = None):
    t5, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()

    if not exists(output_device):
        return encoded_text, attn_mask

    encoded_text.to(output_device)
    attn_mask.to(output_device)

    return encoded_text, attn_mask

class T5Adapter():
    def __init__(
        self,
        name,
        text_embed_pad_value = 0.
    ):
        name = default(name, DEFAULT_T5_NAME)
        t5, tokenizer = get_model_and_tokenizer(name)

        if torch.cuda.is_available():
            t5 = t5.cuda()

        self.name =  name
        self.t5 = t5
        self.tokenizer = tokenizer
        self.text_embed_pad_value = text_embed_pad_value

    @property
    def dim_latent(self):
        return get_encoded_dim(self.name)

    @property
    def max_text_len(self):
        return MAX_LENGTH

    @torch.no_grad()
    @beartype
    def embed_text(
        self,
        texts: List[str],
        return_text_encodings = False,
        output_device = None
    ):
        device = next(self.t5.parameters()).device

        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors = "pt",
            padding = 'longest',
            max_length = MAX_LENGTH,
            truncation = True
        )

        input_ids = encoded.input_ids.to(device)
        attn_mask = encoded.attention_mask.to(device)

        self.t5.eval()

        with torch.no_grad():
            output = self.t5(input_ids = input_ids, attention_mask = attn_mask)
            encoded_text = output.last_hidden_state.detach()

        attn_mask = attn_mask.bool()

        encoded_text.masked_fill_(~attn_mask[..., None], self.text_embed_pad_value)

        if not return_text_encodings:
            numer = encoded_text.sum(dim = -2)
            denom = attn_mask.sum(dim = -1)[..., None]
            numer.masked_fill_(denom == 0, 0.)
            mean_encodings = numer / denom.clamp(min = 1e-3)
            return mean_encodings

        return encoded_text.to(output_device)
