# CoFi-CLM: A Coarse-to-Fine Channel Language Model for Finite-Bit CSI Feedback

[![arXiv](https://img.shields.io/badge/arXiv-2609.15865-b31b1b.svg)](https://arxiv.org/abs/2609.15865)

This repository accompanies the paper
[**"CoFi-CLM: A Coarse-to-Fine Channel Language Model for Finite-Bit CSI Feedback"**](https://arxiv.org/abs/2609.15865), which is currently under review.

> **Release status.** This is a preview release of the core CoFi-CLM framework.
> It contains the model architecture and the training objective described in the
> paper. The complete training and evaluation pipeline, experiment
> configurations, dataset, and pretrained checkpoints will be released after the
> paper is accepted.

## Overview

CoFi-CLM formulates finite-bit CSI recovery as conditional fine-token generation
from a limited coarse-token context.

- **UE side:** a lightweight coarse tokenizer maps the angle–delay CSI to a
  4×4 grid of coarse-codebook indices and reports a budget-dependent subset.
- **BS side:** the channel language model (CLM) predicts categorical
  distributions over the fine codebook for all 8×8 fine positions in a single
  Transformer forward pass. The probability-weighted fine embeddings are decoded
  by a frozen fine decoder, and a dual-path fusion module combines this
  generated reconstruction with the direct coarse reconstruction.

Concentrating prediction and fusion at the BS allows the CLM capacity to scale
without increasing UE computation or storage.

## Repository Structure

```text
cofi_clm/
  models/
    tokenizer.py        # Coarse/fine CSI tokenizers with EMA vector quantization
    language_model.py   # One-pass coarse-to-fine fine-token prediction (CLM)
    fusion.py           # Dual-path gated residual fusion
    clm.py              # CoFi-CLM: UE feedback and BS reconstruction interfaces
    layers.py           # Shared convolutional and positional layers
  losses.py             # Training objective
  metrics.py            # Global power-ratio NMSE
```

## Requirements

Python 3.10 or later and PyTorch 2.6 or later.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{zhang2026cofi,
  title={CoFi-CLM: A Coarse-to-Fine Channel Language Model for Finite-Bit CSI Feedback},
  author={Zhang, Chao and Luo, Cheng and Xiang, Luping and Yang, Kun},
  journal={arXiv preprint arXiv:2609.15865},
  year={2026}
}
```
