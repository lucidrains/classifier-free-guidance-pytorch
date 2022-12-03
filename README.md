## Classifier Free Guidance - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2207.12598">Classifier Free Guidance</a> in Pytorch, with emphasis on text conditioning, and flexibility to include multiple text embedding models, as done in <a href="https://deepimagination.cc/eDiff-I/">eDiff-I</a>

It is clear now that text guidance is the ultimate interface to models. This repository will leverage some python decorator magic to make it easy to incorporate SOTA text conditioning to any model.

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
