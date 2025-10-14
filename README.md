# Prostate T2 Relaxometry using Convolutional Neural Networks

Code to develop and evaluate neural networks for parameter estimation in prostate T2 relaxometry.

This repository contains all code and models from:
**"Improved Quantitative Parameter Estimation for Prostate T2 Relaxometry using Convolutional Neural Networks"**  
https://doi.org/10.1101/2023.01.11.23284194

## Overview

You can use this code to:
- Generate synthesized training/testing data
- Train neural network models for T2 parameter estimation
- Run inference on synthetic and in-vivo data
- Reproduce all figures from the paper

You can use this code to generate the synthesized data, train the models, run inference, and create all the figures in the paper. However, this process is not fully automated - you will need to do this in parts, with some manual steps. 

## Requirements

### System Requirements
- Python 3.8+ with virtual environment support
- Several GB of disk space for datasets and models
- GPU recommended for training (CPU works but is slower)

### Python Environment
```bash
pip install -r requirements.txt
```

**Known Issue**: The requirements.txt may not install perfectly. If using Spyder, install version 5.2.2 first, then upgrade to latest to resolve missing packages.

## Setup Instructions

### 1. Configure Paths
Edit `utility_functions.py` to update hardcoded paths for your system:
- Base directory (currently `/home/pbolan/prostate_t2map`)
- ImageNet data location

### 2. Download ImageNet Data
Download the full ImageNet dataset (165GB) from:
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

**Note**: Only the validation dataset (50k images, 6.4GB) is used, but the full download is required.
Validation data location: `ILSVRC/Data/CLS-LOC/val/`
First file should be: `ILSVRC2012_val_00000001.JPEG` (sea snake on beach)

### 3. Download In-Vivo Data
Download from: https://conservancy.umn.edu/handle/11299/243192

Extract `images.tgz` and organize as follows:
```
$BASE/datasets/
├── invivo_set1/images/    # Development set (first 9 files from imagesTr)
├── invivo_set2/images/    # Training set (all files from imagesTr)  
└── invivo_set3/images/    # Test set (all files from imagesTs)
```

**Convert 3D to 2D slices**:
```bash
python extract_slices_from_invivo_datasets.py
```

### 4. Generate Synthetic Data
```bash
python build_synthetic_datasets.py
```
This creates both large (10k) training sets and smaller (1k) development sets.

## Usage

### Training Models (Optional)
Train your own models (takes several days):
```bash
python train_1d.py      # 1D neural networks
python train_cnn.py     # 2D convolutional networks
```
Models saved to: `$BASE/models/*.pt`

**Alternative**: Use pre-trained models from GitHub releases.

### Running Inference
```bash
python inference.py
```

Inference is organized in three parts:
- **Part A**: Synthetic data evaluation
- **Part B**: In-vivo data evaluation  
- **Part C**: In-vivo data with added noise

Results saved to: `$BASE/predictions/` (~7GB)

### Generating Figures

**Figure 1, 3 (top), S1**:
```bash
python make_demo_figure.py
```
(Change switch value and re-run for different plots)

**Figure 3**:
```bash
python plot_example_partA.py
```

**Figures 4-5** (synthetic data evaluation, takes several hours):
```bash
python make_plots_partA.py
```

**Figure 6**:
```bash
python plot_example_partB.py
```

**Figure 7**:
```bash
python plot_example_partC.py
```

**Figure 8**:
```bash
python evaluate_partC_byslice.py
```

## Directory Structure
```
prostate_t2map/
├── datasets/           # Training and test data
├── models/            # Trained model files
├── predictions/       # Inference results
├── figures/           # Generated plots (you create this)
└── [source files]     # Python scripts
```

## Notes
- Graphics rendering: Development used `%matplotlib inline`, paper used Qt backend
- Plots saved as PNG and SVG for manual figure assembly
- Some analysis scripts need to be run multiple times with different parameters

## Citation
If you use this code, please cite our paper:
Bolan PJ, Saunders SL, Kay K, Gross M, Akcakaya M, Metzger GJ. Improved quantitative parameter estimation for prostate T2 relaxometry using convolutional neural networks. MAGMA. 2024 Aug;37(4):721-735. doi: 10.1007/s10334-024-01186-3. Epub 2024 Jul 23. PMID: 39042205; PMCID: PMC11417079. https://link.springer.com/article/10.1007/s10334-024-01186-3

This code actually reproduces figures from the preprint, which has more information and comparisons than the final paper:
Improved Quantitative Parameter Estimation for Prostate T2 Relaxometry using Convolutional Neural Networks
Patrick J. Bolan, Sara L. Saunders, Kendrick Kay, Mitchell Gross, Mehmet Akcakaya, Gregory J. Metzger
medRxiv 2023.01.11.23284194; doi: https://doi.org/10.1101/2023.01.11.23284194

## Updates
- **Oct 2025**: Updated the code for MONAI 1.5 compatibility and improved inference efficiency


