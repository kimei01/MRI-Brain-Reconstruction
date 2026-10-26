# MRI-Brain-Reconstruction
# Real- vs Complex-Valued Vision Transformers for Undersampled MRI Reconstruction

## What We're Building

A controlled comparison of two Vision Transformers (ViTs) that reconstruct MRI images from undersampled k-space data

**Research question:** Does the complex-valued advantage seen in CNN-based MRI reconstruction also hold for attention-based models?

**Gap we fill:** Existing MRI transformers (Lin & Heckel, SwinMR) are real-valued. Complex-valued ViTs exist only for k-space *classification* (Rempe et al.). No controlled real-vs-complex ViT comparison for *reconstruction* has been published, to our knowledge.

**Keep controlled:** same data, masks, loss, optimizer, epochs, seeds, depth, and patch size. Only the data representation changes. Width is adjusted so both models have about the same parameter count.

## Pipeline

```
fastMRI brain multi-coil k-space
  -> virtual single-coil complex image (coil compression / sensitivity-weighted combination)
  -> crop to fixed size, normalize, store compact dataset
  -> equispaced Cartesian undersampling masks (4x and 8x)
  -> IFFT -> undersampled complex image
  -> [real ViT (2 channels)]  or  [complex ViT]
  -> reconstruction head -> magnitude image
  -> compare against fastMRI reference reconstruction (reconstruction_rss)
```

## Timeline (Oct 5 to Dec 13, 2026)

| Task | Weeks | Dates (Monday of week) |
|---|---|---|
| A. Finalize literature review & methodology, set up repo + PyTorch env | W1-2 | Oct 5, Oct 12 |
| B. Download & check fastMRI brain data | W1-2 | Oct 5, Oct 12 |
| C. Virtual single-coil conversion & preprocessing | W2-3 | Oct 12, Oct 19 |
| D. Undersampling masks (4x, 8x) & data loaders | W3 | Oct 19 |
| E. Real-valued ViT baseline | W3-5 | Oct 19 to Nov 2 |
| **Checkpoint: compute (laptop vs Colab Pro)** | end of W4 | **Nov 1** |
| F. Complex-valued ViT | W4-6 | Oct 26 to Nov 9 |
| G. Controlled training (4x and 8x) | W6-8 | Nov 9 to Nov 23 |
| H. Evaluation & analysis | W8-9 | Nov 23, Nov 30 |
| I. Final report & presentation | W9-10 | Nov 30, Dec 7 |

**Compute checkpoint (Nov 1):** time one training epoch of the baseline on the RTX 3070 Ti. If full training won't fit in Weeks 6-8, move the training runs to Colab Pro.

## Repo Layout (suggested)

```
.
├── data/            # fastMRI subset + processed dataset (git-ignored)
├── src/
│   ├── data/        # preprocessing, masks, loaders
│   ├── models/      # real_vit.py, complex_vit.py, complex layers
│   ├── train.py
│   └── evaluate.py
├── configs/         # shared hyperparameters for both models
├── notebooks/       # exploration, Colab runs
├── results/         # metrics, figures
└── README.md
```


Data: request and download fastMRI brain data from https://fastmri.med.nyu.edu/ and place it under `data/`. Do not commit data files.