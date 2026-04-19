# Methodology

## Pipeline

The repository is organized around two tasks:

1. **SAR autofocus / image reconstruction**
2. **Terrain classification**

The code in `src/` is a cleaned and modularized version inspired by the original notebook workflow.

## Recommended Workflow

1. Prepare dataset folders and update paths in `configs/`
2. Train the autofocus model
3. Train the classifier on focused image data
4. Save figures and metrics under `results/`

## Suggested Future Improvements

- Add a dedicated inference script
- Add confusion matrix generation
- Add SSIM / PSNR batch evaluation
- Add model checkpoint tracking
- Add reproducibility seed control
