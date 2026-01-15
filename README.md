# Prokudin-Gorskii Color Image Reconstruction

**Student:** Omar A.M. Issa  
**ID:** 220212901  
**University:** OSTIM Technical University

## Why this code exists
Sergey Prokudin-Gorskii captured early color photographs by taking three separate
exposures through blue, green, and red filters. The resulting scans are stacked
vertically in a single grayscale image. This project reconstructs a modern color
photo by splitting, aligning, and enhancing those three exposures.

## What the code does
The pipeline for each input image is:
- Load a stacked grayscale plate and split it into **B**, **G**, **R** channels.
- Align **G** and **R** to **B** using **NCC** or **SSD** (optionally with a pyramid).
- Merge the aligned channels into an RGB image.
- Enhance contrast and sharpness (histogram equalization, gamma, unsharp mask).
- Auto-crop black borders produced by the alignment step.

## Code structure (and why each part exists)
- `code/main.py`  
  Entry point and CLI. Orchestrates the full pipeline, saves outputs, and prints
  a summary table.
- `code/alignment.py`  
  Alignment metrics (SSD/NCC) and two search strategies: brute-force and
  multi-scale pyramid alignment for faster, large-shift correction.
- `code/enhancement.py`  
  Image enhancement steps that improve visual quality without altering alignment.
- `code/utils.py`  
  I/O helpers, channel splitting, channel shifting, RGB composition, and
  automatic border cropping.

## Repository layout
- `code/` Source code.
- `data/` Input stacked images (`.jpg` or `.tif`).
- `results/` Output images (`_unaligned`, `_aligned`, `_enhanced`).
- `OmarIssa_Goruntu_Isleme_Rapor.pdf` Project report.

## Requirements
- Python **3.10+** (uses `str | Path` type hints)
- Packages:
  - `numpy`
  - `opencv-python`

Install dependencies:
```bash
python -m pip install numpy opencv-python
```

## How to run
From the `code/` folder:
```bash
python main.py --input ../data --output ../results --metric ncc --pyramid
```

Run on a single image:
```bash
python main.py --input ../data/00106v.jpg --output ../results --metric ssd
```

## Inputs and outputs
- **Input format:** A single grayscale image with three exposures stacked
  top-to-bottom in the order **B**, **G**, **R**.
- **Supported extensions:** `.jpg`, `.tif`
- **Outputs:** For each input image, three files are written to `results/`:
  - `*_unaligned.jpg` (raw RGB stack, no alignment)
  - `*_aligned.jpg` (aligned channels)
  - `*_enhanced.jpg` (aligned + enhancement + auto-crop)

## Notes
- Use `--pyramid` for faster and more robust alignment on large images.
- The crop step removes black borders introduced by shifting channels.
