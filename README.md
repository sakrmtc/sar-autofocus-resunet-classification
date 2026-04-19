# SAR Autofocus and Terrain Classification using ResU-Net and CNN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#installation)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository presents a research-oriented implementation of a deep-learning pipeline for **synthetic aperture radar (SAR) image autofocus** and **terrain classification**.

The project is based on the published paper:

> **Enhanced synthetic aperture radar image autofocus and classification using 2D SARNet framework**  
> Mohamed Sakr, Ahmed Saleh, Fathy AbdElkader, Ghada Amer, and Mohamed AboElenean  
> *Journal of Applied Remote Sensing*, 2024

## Overview

The proposed pipeline is composed of two main stages:

1. **SAR autofocus / image formation** using an encoder–decoder architecture:
   - U-Net
   - ResU-Net
2. **Terrain classification** using a CNN-based classifier for:
   - Mountain
   - Sand
   - Sea / water-like scenes

The main goal is to support **near-real-time SAR processing** by replacing heavy traditional post-processing with a deep-learning-based workflow.

## Key Contributions

- Research implementation aligned with the 2D SARNet paper
- ResU-Net based SAR autofocus pipeline
- CNN-based terrain classification
- Reproducible project structure for academic presentation
- Clean repository layout suitable for GitHub and portfolio use

## Repository Structure

```text
sar-autofocus-resunet-classification/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
│
├── paper/
│   └── JARS-240130G_online.pdf
│
├── notebooks/
│   ├── res_u_net_paper.ipynb
│   └── classifier_sea.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_resunet.py
│   ├── train_autofocus.py
│   ├── train_classifier.py
│   ├── evaluate.py
│   └── utils.py
│
├── configs/
│   ├── autofocus_config.yaml
│   └── classifier_config.yaml
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── sample_outputs/
│
├── docs/
│   ├── methodology.md
│   └── dataset_notes.md
│
└── assets/
    └── repo_banner.png
```

## Methodology

### 1) Autofocus Model
The autofocus model is based on **U-Net / ResU-Net**.  
Residual blocks are added to strengthen feature extraction and improve reconstruction quality.

### 2) Classification Model
The classifier is a compact CNN that predicts terrain categories from focused SAR images.

### 3) Data Preparation
The original workflow uses SAR raw data and focused targets generated from conventional SAR processing.  
This repository keeps the implementation modular so the preprocessing, training, and evaluation steps are easier to reproduce and extend.

## Results Summary

Reported paper highlights include:

- Strong autofocus performance using **ResU-Net**
- High-quality reconstruction measured with **SSIM** and **PSNR**
- CNN-based terrain classification with strong reported accuracy
- A near-real-time oriented processing pipeline

> For exact quantitative values, please refer to the paper in `paper/JARS-240130G_online.pdf`.

## Installation

### Option 1: pip
```bash
git clone https://github.com/sakrmtc/sar-autofocus-resunet-classification.git
cd sar-autofocus-resunet-classification
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Option 2: conda
```bash
conda env create -f environment.yml
conda activate sar-autofocus
```

## Usage

### Train autofocus model
```bash
python src/train_autofocus.py --config configs/autofocus_config.yaml
```

### Train classifier
```bash
python src/train_classifier.py --config configs/classifier_config.yaml
```

### Evaluate predictions
```bash
python src/evaluate.py
```

## Notes on Data

- The original paper uses **ERS SAR data** and focused reference outputs.
- Large datasets are intentionally **not included** in this repository.
- Update dataset paths through the YAML config files in `configs/`.

## Recommended GitHub Setup

To make the repository look professional on GitHub:

- Add a short project description
- Pin this repository on your profile
- Add sample outputs under `results/sample_outputs/`
- Add training curves under `results/figures/`
- Create a first release such as `v1.0`

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{sakr2024sarnet,
  title={Enhanced synthetic aperture radar image autofocus and classification using 2D SARNet framework},
  author={Sakr, Mohamed and Saleh, Ahmed and AbdElkader, Fathy and Amer, Ghada and AboElenean, Mohamed},
  journal={Journal of Applied Remote Sensing},
  year={2024},
  doi={10.1117/1.JRS.18.026507}
}
```

## License

This repository is released under the **MIT License**. See `LICENSE` for details.
