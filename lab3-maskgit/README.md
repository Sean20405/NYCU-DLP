# Lab 3: MaskGIT for Image Inpainting

MaskGIT inpainting with a pre-trained VQGAN and a bidirectional transformer. Stage 1 is the VQGAN, which the course provides. Stage 2 trains the transformer to predict masked image tokens. Inpainting then fills the masked tokens in several iterative decoding steps.

## Setup

```bash
conda env create -f environment.yml
conda activate maskgit
```

You must add these files from the course:

- The VQGAN checkpoint in `models/VQGAN/checkpoints/`.
- The LPIPS weights `vgg.pth` in `models/VQGAN/vgg_lpips/`.
- The dataset and the masks. Edit the paths in `config/MaskGit.yml`.
- The `faster-pytorch-fid` tool, which computes the FID score.

## Run

```bash
python training_transformer.py
python inpainting.py
```

The `script/` folder has the tools for the plots in the report: `plot_fid.py`, `create_comparison.py`, `combine_images.py`, and `find_sweet_spot.py`.

## Results

My best FID score is 29.1414. The cosine mask schedule gave the best results of the schedules that I compared. `results/func_comparison.png` plots the mask schedule functions.

Two choices improved the FID score:

- Sampling a token from the predicted distribution instead of `argmax`, as in the MaskGIT paper.
- Gradient accumulation over 10 steps. The GPU limited the batch size to 16, so the effective batch size was 160.

The full analysis is in [report/report.pdf](report/report.pdf).
