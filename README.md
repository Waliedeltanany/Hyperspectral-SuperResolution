# Hyperspectral-SuperResolution
Deep Learning based Super-Resolution for Hyperspectral Images (HSI) using PyTorch.  This repository includes data preprocessing, patch extraction, model training, and evaluation (PSNR/SSIM).

This repository contains an implementation of a Deep Learning pipeline for **Hyperspectral Image (HSI) Super-Resolution** using PyTorch.

## Features
- Data loading from `.mat` files (e.g., PaviaU).
- Automatic conversion to `.npy` format.
- Patch extraction for training.
- Lightweight CNN model for super-resolution.
- Training loop with PSNR and SSIM evaluation.
- Saving results and reconstructed images.

## Dataset
- Tested on **PaviaU dataset** (hyperspectral data).
- You can download more datasets from [Kaggle](https://www.kaggle.com) or [IEEE Dataport](https://ieee-dataport.org/).

## Results
- Achieved **PSNR > 30 dB** on PaviaU after training.
- Qualitative comparison shows clear improvement from LR → SR → HR.

## Usage
```bash
python DL.py
