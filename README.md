## Classifier Free Guidance - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2207.12598">Classifier Free Guidance</a> in Pytorch, with emphasis on text conditioning, and flexibility to include multiple text embedding models, as done in <a href="https://deepimagination.cc/eDiff-I/">eDiff-I</a>

It is clear now that text guidance is the ultimate interface to models. This repository will leverage some python decorator magic to make it easy to incorporate SOTA text conditioning to any model.

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the generous sponsorship, as well as my other sponsors out there

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their amazing transformers library. The text conditioning module will use T5 embeddings, as latest research recommends

- <a href="https://github.com/mlfoundations/open_clip">OpenCLIP</a> for providing SOTA open sourced CLIP models. The eDiff model sees immense improvements by combining the T5 embeddings with CLIP text embeddings


## Install

```bash
$ pip install classifier-free-guidance-pytorch
```

## Usage

```python
import torch
from classifier_free_guidance_pytorch import TextConditioner

text_conditioner = TextConditioner(
    model_types = 't5',    
    hidden_dims = (256, 512),
    hiddens_channel_first = False,
    cond_drop_prob = 0.2  # conditional dropout 20% of the time, must be greater than 0. to unlock classifier free guidance
).cuda()

# pass in your text as a List[str], and get back a List[callable]
# each callable function receives the hiddens in the dimensions listed at init (hidden_dims)

first_condition_fn, second_condition_fn = text_conditioner(['a dog chasing after a ball'])

# these hiddens will be in the direct flow of your model, say in a unet

first_hidden = torch.randn(1, 16, 256).cuda()
second_hidden = torch.randn(1, 32, 512).cuda()

# conditioned features

first_conditioned = first_condition_fn(first_hidden)
second_conditioned = second_condition_fn(second_hidden)
```

## Todo

- [x] complete film conditioning, without classifier free guidance (used <a href="https://github.com/lucidrains/robotic-transformer-pytorch/blob/main/robotic_transformer_pytorch/robotic_transformer_pytorch.py">here</a>)
- [x] add classifier free guidance for film conditioning

- [ ] complete cross attention conditioning
- [ ] complete auto perceiver resampler

## Citations

```bibtex
@article{Ho2022ClassifierFreeDG,
    title   = {Classifier-Free Diffusion Guidance},
    author  = {Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2207.12598}
}
```

```bibtex
@article{Balaji2022eDiffITD,
    title   = {eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers},
    author  = {Yogesh Balaji and Seungjun Nah and Xun Huang and Arash Vahdat and Jiaming Song and Karsten Kreis and Miika Aittala and Timo Aila and Samuli Laine and Bryan Catanzaro and Tero Karras and Ming-Yu Liu},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2211.01324}
}
```
