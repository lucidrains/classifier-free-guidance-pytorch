from __future__ import annotations

from collections import namedtuple
from functools import wraps, partial, cache

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack

from beartype.door import is_bearable
from beartype.typing import Callable, Tuple, List, Literal, Dict, Any

from inspect import signature

from classifier_free_guidance_pytorch.typing import typecheck, beartype_isinstance

from classifier_free_guidance_pytorch.t5 import T5Adapter
from classifier_free_guidance_pytorch.open_clip import OpenClipAdapter
from classifier_free_guidance_pytorch.attend import Attend
from classifier_free_guidance_pytorch.bge import BGEAdapter

# constants

COND_DROP_KEY_NAME = 'cond_drop_prob'

TEXTS_KEY_NAME = 'texts'
TEXT_EMBEDS_KEY_NAME = 'text_embeds'
TEXT_CONDITIONER_NAME = 'text_conditioner'
CONDITION_FUNCTION_KEY_NAME = 'cond_fns'

TextCondReturn = namedtuple('TextCondReturn', [
    'embed',
    'mask'
])

# helper functions

def exists(val):
    return val is not None

def is_empty(l):
    return len(l) == 0

def default(*values):
    for value in values:
        if exists(value):
            return value
    return None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# tensor helpers

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# classifier free guidance with automatic text conditioning

@typecheck
def classifier_free_guidance(
    fn: Callable,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    fn_params = signature(fn).parameters

    auto_handle_text_condition = texts_key_name not in fn_params and text_embeds_key_name not in fn_params

    @wraps(fn)
    def inner(
        self,
        *args,
        cond_scale: float = 1.,
        rescale_phi: float = 0.,
        return_unconditioned: bool = False,
        cfg_routed_kwargs: Dict[str, Tuple[Any, Any]] = dict(),   # to pass in separate arguments to forward and nulled forward calls (for handling caching when using CFG on transformer decoding)
        **kwargs
    ):
        @wraps(fn)
        def fn_maybe_with_text(self, *args, **kwargs):
            if auto_handle_text_condition:
                texts = kwargs.pop('texts', None)
                text_embeds = kwargs.pop('text_embeds', None)

                assert not (exists(texts) and exists(text_embeds))

                raw_text_cond = cond_fns = None

                text_conditioner = getattr(self, text_conditioner_name, None)

                cond_drop_prob = kwargs.pop(cond_drop_prob_keyname, None)

                assert not exists(cond_drop_prob) or 0. <= cond_drop_prob <= 1.

                # auto convert texts -> conditioning functions

                if exists(texts) ^ exists(text_embeds):

                    assert is_bearable(texts, List[str] | None), f'keyword `{texts_key_name}` must be a list of strings'

                    assert exists(text_conditioner) and is_bearable(text_conditioner, Conditioner), 'text_conditioner must be set on your network with the correct hidden dimensions to be conditioned on'

                    text_condition_input = dict(texts = texts) if exists(texts) else dict(text_embeds = text_embeds)

                    cond_fns, raw_text_cond = text_conditioner(**text_condition_input, cond_drop_prob = cond_drop_prob)

                elif isinstance(text_conditioner, NullConditioner):
                    assert cond_drop_prob == 0., 'null conditioner has nothing to dropout'

                    cond_fns, raw_text_cond = text_conditioner()

                if 'cond_fns' in fn_params:
                    kwargs.update(cond_fns = cond_fns)

                if 'raw_text_cond' in fn_params:
                    kwargs.update(raw_text_cond = raw_text_cond)

            return fn(self, *args, **kwargs)

        # main classifier free guidance logic

        if self.training:
            assert cond_scale == 1, 'you cannot do condition scaling when in training mode'

            return fn_maybe_with_text(self, *args, **kwargs)

        assert cond_scale >= 1, 'invalid conditioning scale, must be greater or equal to 1'

        kwargs_without_cond_dropout = {**kwargs, cond_drop_prob_keyname: 0.}
        kwargs_with_cond_dropout = {**kwargs, cond_drop_prob_keyname: 1.}

        # handle kwargs to be routed to forward and nulled forward separately
        # for handling caching of both calls

        fn_kwargs = {k: v[0] for k, v in cfg_routed_kwargs.items()}
        null_fn_kwargs = {k: v[1] for k, v in cfg_routed_kwargs.items()}

        # non-null forward

        outputs = fn_maybe_with_text(self, *args, **fn_kwargs, **kwargs_without_cond_dropout)

        if cond_scale == 1:
            return outputs

        logits, *rest = cast_tuple(outputs)

        # nulled forward

        null_outputs = fn_maybe_with_text(self, *args, **null_fn_kwargs, **kwargs_with_cond_dropout)

        null_logits, *null_rest = cast_tuple(null_outputs)

        zipped_rest = tuple(zip(rest, null_rest))

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescale_phi <= 0:
            logit_output = scaled_logits
        else:
            # proposed in https://arxiv.org/abs/2305.08891
            # as a way to prevent over-saturation with classifier free guidance
            # works both in pixel as well as latent space as opposed to the solution from imagen

            dims = tuple(range(1, logits.ndim - 1))
            rescaled_logits = scaled_logits * (logits.std(dim = dims, keepdim = True) / scaled_logits.std(dim = dims, keepdim= True))
            logit_output = rescaled_logits * rescale_phi + scaled_logits * (1. - rescale_phi)

        # can return unconditioned prediction
        # for use in CFG++ https://arxiv.org/abs/2406.08070

        output = logit_output

        if return_unconditioned:
            output = (output, null_logits)

        # handle multiple outputs from original function

        if is_empty(zipped_rest):
            return output

        return (output, *zipped_rest)

    return inner

# class decorator

@typecheck
def classifier_free_guidance_class_decorator(
    orig_class,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    assert issubclass(orig_class, Module)

    # decorate init

    orig_init = orig_class.__init__

    @wraps(orig_init)
    @typecheck
    def __init__(
        self,
        *args,
        text_condition_type: Literal['film', 'attention', 'null', 'raw'] = 'film',
        text_condition_model_types: Tuple[str, ...] = ('t5',),
        text_condition_hidden_dims: Tuple[int, ...],
        text_condition_cond_drop_prob: float,
        **kwargs
    ):
        orig_init(self, *args, **kwargs)

        if text_condition_type == 'film':
            condition_klass = TextConditioner
        elif text_condition_type == 'attention':
            condition_klass = AttentionTextConditioner
        elif text_condition_type == 'raw':
            condition_klass = TextEmbeddingReturner
        else:
            condition_klass = NullConditioner

        self.text_conditioner = condition_klass(
            model_types = text_condition_model_types,
            hidden_dims = text_condition_hidden_dims,
            cond_drop_prob = text_condition_cond_drop_prob
        )

    orig_class.__init__ = __init__

    # decorate forward

    decorated_forward = classifier_free_guidance(
        orig_class.forward,
        cond_drop_prob_keyname = cond_drop_prob_keyname,
        texts_key_name = texts_key_name,
        text_embeds_key_name = text_embeds_key_name,
        cond_fns_keyname = cond_fns_keyname,
        text_conditioner_name = text_conditioner_name
    )

    orig_class.forward = decorated_forward

    # forward `embed_texts` to the `text_conditioner.embed_texts`

    @typecheck
    def embed_texts(self, texts: List[str]):
        return self.text_conditioner.embed_texts(texts)

    @property
    @cache
    def max_cond_text_len(self):
        total_cond_text_len = sum([text_model.max_text_len for text_model in self.text_conditioner.text_models])
        return total_cond_text_len

    if not hasattr(orig_class, 'max_cond_text_len'):
        orig_class.max_cond_text_len = max_cond_text_len

    if not hasattr(orig_class, 'embed_texts'):
        orig_class.embed_texts = embed_texts

    orig_class.__decorated_with_cfg = True
    return orig_class

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_context = None,
        norm_context = False,
        num_null_kv = 0,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attend = Attend(flash = flash)        

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        mask = None
    ):
        b = x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b = b).unbind(dim = 0)
            k = torch.cat((null_k, k), dim = -2)
            v = torch.cat((null_v, v), dim = -2)

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# dimension adapters

def rearrange_channel_last(fn):
    @wraps(fn)
    def inner(hiddens):
        hiddens, ps = pack_one(hiddens, 'b * d')
        conditioned = fn(hiddens)
        return unpack_one(conditioned, ps, 'b * d')
    return inner

def rearrange_channel_first(fn):
    """ will adapt shape of (batch, feature, ...) for conditioning """

    @wraps(fn)
    def inner(hiddens):
        hiddens, ps = pack_one(hiddens, 'b d *')
        hiddens = rearrange(hiddens, 'b d n -> b n d')
        conditioned =  fn(hiddens)
        conditioned = rearrange(conditioned, 'b n d -> b d n')
        return unpack_one(conditioned, ps, 'b d *')

    return inner

# conditioning modules

class FiLM(Module):
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditions, hiddens):
        scale, shift = self.net(conditions).chunk(2, dim = -1)
        assert scale.shape[-1] == hiddens.shape[-1], f'unexpected hidden dimesion {hiddens.shape[-1]} used for conditioning'
        scale, shift = map(lambda t: rearrange(t, 'b d -> b 1 d'), (scale, shift))
        return hiddens * (scale + 1) + shift

class CrossAttention(Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        heads = 8,
        dim_head = 64,
        flash = False
    ):
        super().__init__()
        self.attn = Attention(
            dim = hidden_dim,
            dim_context = dim,
            norm_context = True,
            num_null_kv = 1,
            dim_head = dim_head,
            heads = heads,
            flash = flash
        )

    def forward(
        self,
        condition,
        hiddens,
        mask = None
    ):
        return self.attn(hiddens, condition, mask = mask) + hiddens

# film text conditioning

CONDITION_CONFIG = dict(
    t5 = T5Adapter,
    clip = OpenClipAdapter,
    bge = BGEAdapter
)

MODEL_TYPES = CONDITION_CONFIG.keys()

class Conditioner(Module):
    pass

# null conditioner

class Identity(Module):
    def forward(self, t, *args, **kwargs):
        return t

class NullConditioner(Conditioner):
    @typecheck
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        **kwargs
    ):
        super().__init__()
        num_null_conditioners = len(hidden_dims)
        self.cond_fns = tuple(Identity() for _ in range(num_null_conditioners))

        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    @typecheck
    def embed_texts(self, texts: List[str]):
        assert False, 'null conditioner cannot embed text'

    def forward(self, *args, **kwarg):
        return self.cond_fns, None

# text conditioner with FiLM

class TextConditioner(Conditioner):
    @typecheck
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        model_types = 't5',
        model_names = None,
        cond_drop_prob = 0.,
        hiddens_channel_first = True,
        text_embed_stem_dim_mult = 2,
        text_embed_pad_value = 0.
    ):
        super().__init__()
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))

        assert len(model_types) == len(model_names)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        text_models = []

        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name, text_embed_pad_value = text_embed_pad_value)
            text_models.append(model)

        self.text_models = text_models
        self.latent_dims = [model.dim_latent for model in text_models]

        self.conditioners = ModuleList([])

        self.hidden_dims = hidden_dims
        self.num_condition_fns = len(hidden_dims)
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns) # whether hiddens to be conditioned is channel first or last

        assert len(self.hiddens_channel_first) == self.num_condition_fns

        self.cond_drop_prob = cond_drop_prob

        total_latent_dim = sum(self.latent_dims)

        self.dim_latent = total_latent_dim

        mlp_stem_output_dim = total_latent_dim * text_embed_stem_dim_mult

        self.text_embed_stem_mlp = nn.Sequential(
            nn.Linear(total_latent_dim, mlp_stem_output_dim),
            nn.SiLU()
        )

        for hidden_dim in hidden_dims:
            self.conditioners.append(FiLM(mlp_stem_output_dim, hidden_dim))

        self.null_text_embed = nn.Parameter(torch.randn(total_latent_dim))

        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    @typecheck
    def embed_texts(self, texts: List[str]):
        device = self.device

        text_embeds = []
        for text_model in self.text_models:
            text_embed = text_model.embed_text(texts)
            text_embeds.append(text_embed.to(device))

        return torch.cat(text_embeds, dim = -1)

    @typecheck
    def forward(
        self,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None,
        repeat_batch = 1,               # for robotic transformer edge case
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
    ]:

        assert exists(texts) ^ exists(text_embeds)

        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        if exists(texts):
            batch = len(texts)
        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            null_text_embeds = rearrange(self.null_text_embed, 'd -> 1 d')

            text_embeds = torch.where(
                prob_keep_mask,
                text_embeds,
                null_text_embeds
            )

        # text embed mlp stem, as done in unet conditioning in guided diffusion

        text_embeds = self.text_embed_stem_mlp(text_embeds)

        # prepare the conditioning functions

        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        cond_fns = []

        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioners, self.hiddens_channel_first, repeat_batch):
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r = cond_repeat_batch)
            cond_fn = partial(cond, cond_text_embeds)

            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

            cond_fns.append(wrapper_fn(cond_fn))

        return tuple(cond_fns), TextCondReturn(text_embeds, None)

# cross attention text conditioner

@typecheck
class AttentionTextConditioner(Conditioner):
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        model_types = 't5',
        model_names = None,
        cond_drop_prob = 0.,
        hiddens_channel_first = True,
        dim_latent = None,
        attn_dim_head = 64,
        attn_heads = 8,
        flash = True,
        text_embed_pad_value = 0.
    ):
        super().__init__()
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))

        assert len(model_types) == len(model_names)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        text_models = []

        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name, text_embed_pad_value = text_embed_pad_value)
            text_models.append(model)

        self.text_models = text_models

        self.to_latent_dims = ModuleList([])

        dim_latent = default(dim_latent, max([model.dim_latent for model in text_models]))

        self.dim_latent = dim_latent

        for model in text_models:
            self.to_latent_dims.append(nn.Linear(model.dim_latent, dim_latent))

        self.conditioners = ModuleList([])

        self.hidden_dims = hidden_dims
        self.num_condition_fns = len(hidden_dims)
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns) # whether hiddens to be conditioned is channel first or last

        assert len(self.hiddens_channel_first) == self.num_condition_fns

        self.text_embed_pad_value = text_embed_pad_value
        self.cond_drop_prob = cond_drop_prob

        for hidden_dim in hidden_dims:
            self.conditioners.append(CrossAttention(dim_latent, hidden_dim, flash = flash))

        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def embed_texts(self, texts: List[str]):
        device = self.device

        text_embeds = []

        for text_model, to_latent in zip(self.text_models, self.to_latent_dims):
            text_embed = text_model.embed_text(texts, return_text_encodings = True)

            text_embed = text_embed.to(device)

            mask = (text_embed != self.text_embed_pad_value).any(dim = -1)

            text_embed = to_latent(text_embed)
            text_embed = text_embed.masked_fill(~mask[..., None], self.text_embed_pad_value)

            text_embeds.append(text_embed)

        return torch.cat(text_embeds, dim = -2)

    @typecheck
    def forward(
        self,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None,
        repeat_batch = 1,  # for robotic transformer edge case
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
    ]:

        assert exists(texts) or exists(text_embeds)

        if exists(text_embeds) and exists(texts):
            # text embed takes precedence
            # for eventual caching of text embeds during the first epoch
            texts = None

        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        if exists(texts):
            batch = len(texts)

        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        mask = (text_embeds != self.text_embed_pad_value).any(dim = -1)

        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            mask = mask & prob_keep_mask

        # prepare the conditioning functions

        repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        cond_fns = []

        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioners, self.hiddens_channel_first, repeat_batch):
            cond_text_embeds = repeat(text_embeds, 'b ... -> (b r) ...', r = cond_repeat_batch)
            cond_mask = repeat(mask, 'b ... -> (b r) ...', r = cond_repeat_batch) if exists(mask) else None

            cond_fn = partial(cond, cond_text_embeds, mask = cond_mask)

            wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

            cond_fns.append(wrapper_fn(cond_fn))

        return tuple(cond_fns), TextCondReturn(text_embeds, mask)

# return raw text embedding

class TextEmbeddingReturner(Conditioner):
    @typecheck
    def __init__(
        self,
        *,
        dim_latent = None,
        hidden_dims: Tuple[int, ...] = (),
        model_types = 't5',
        model_names = None,
        model_kwargs: dict = dict(),
        cond_drop_prob = 0.,
        text_embed_pad_value = 0.
    ):
        super().__init__()
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))
        model_kwargs = cast_tuple(model_kwargs, length = len(model_types))

        assert len(model_types) == len(model_names) == len(model_kwargs)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        text_models = []

        for model_type, model_name, model_kwarg in zip(model_types, model_names, model_kwargs):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name, text_embed_pad_value = text_embed_pad_value, **model_kwarg)
            text_models.append(model)

        self.text_models = text_models
        self.text_embed_pad_value = text_embed_pad_value

        self.to_latent_dims = ModuleList([])

        dim_latent = default(dim_latent, max([model.dim_latent for model in text_models]))

        self.dim_latent = dim_latent

        for model in text_models:
            self.to_latent_dims.append(nn.Linear(model.dim_latent, dim_latent))

        self.conditioners = ModuleList([])

        self.cond_drop_prob = cond_drop_prob

        for hidden_dim in hidden_dims:
            self.conditioners.append(nn.Identity())

        self.register_buffer('_device_param', torch.tensor(0), persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    @typecheck
    def embed_texts(self, texts: List[str]):
        device = self.device

        text_embeds = []

        for text_model, to_latent in zip(self.text_models, self.to_latent_dims):
            text_embed = text_model.embed_text(texts, return_text_encodings = True)

            text_embed = text_embed.to(device)

            mask = (text_embed != self.text_embed_pad_value).any(dim = -1)

            text_embed = to_latent(text_embed)
            text_embed = text_embed.masked_fill(~mask[..., None], self.text_embed_pad_value)

            text_embeds.append(text_embed)

        return torch.cat(text_embeds, dim = -2)

    @typecheck
    def forward(
        self,
        texts: List[str] | None = None,
        text_embeds: Tensor | None = None,
        cond_drop_prob = None
    ) -> Tuple[
        Tuple[Callable, ...],
        TextCondReturn
    ]:

        assert exists(texts) ^ exists(text_embeds)

        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'

        if exists(texts):
            batch = len(texts)

        elif exists(text_embeds):
            batch = text_embeds.shape[0]

        if not exists(text_embeds):
            text_embeds = self.embed_texts(texts)

        mask = (text_embeds != self.text_embed_pad_value).any(dim = -1)

        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob, device = self.device)
            mask = mask & prob_keep_mask

        return tuple(self.conditioners), TextCondReturn(text_embeds, mask)
