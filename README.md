AVPDN: Learning Motion-Robust and Scale-Adaptive Representations for Polyp Detection in Dynamic Colonoscopy Frames

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://github.com/xiaochen925/AVPDN/blob/main/intro.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-red.svg)](https://pytorch.org/)

Our code has been uploaded to GitHub at the link provided in the manuscript: https://github.com/xiaochen925/AVPDN. This repository is set to public access—no permission or login is required for anyone to view and download the code (users can directly click the "Code" button to download all files). In addition, we have also uploaded the final version of the code as supporting material to the submission system of the journal.
As shown in the figure below, the code can be downloaded by clicking the Code button without logging in.
<img width="1893" height="826" alt="image" src="https://github.com/user-attachments/assets/b6c7b4c0-c927-4519-ace6-59738d4b47d7" />

This repository is the official PyTorch implementation of **AVPDN (Adaptive Video Polyp Detection Network)**, a motion-robust and scale-adaptive framework for accurate polyp detection in dynamic colonoscopy videos. The proposed method addresses the core challenges of colonoscopy video detection: motion blur, specular reflections, and severe scale variations caused by rapid camera movement during clinical examinations.

## Overview
Colorectal cancer (CRC) is the third most commonly diagnosed cancer and the second leading cause of cancer-related mortality worldwide. Accurate polyp detection during colonoscopy is critical for the early diagnosis and intervention of CRC. However, dynamic colonoscopy frames suffer from severe motion artifacts, transient interference, and multi-scale variation, which significantly degrade the performance of existing detection models.

To tackle these challenges, we propose AVPDN, built upon the real-time RT-DETR detection framework, with two core innovative components:
- **Adaptive Feature Interaction and Augmentation (AFIA) Module**: A dual-branch self-attention architecture that balances global context modeling (Dense Self-Attention, DSA) and noise suppression (Sparse Self-Attention, SSA), with channel shuffle operations to enhance inter-branch information exchange.
- **Scale-Aware Context Integration (SACI) Module**: A multi-scale feature aggregation module with dilated convolutions of varying receptive fields, which captures both coarse global context and fine-grained local details to stabilize feature representation under drastic scale changes.

AVPDN achieves state-of-the-art performance on two public colonoscopy video benchmarks, while maintaining real-time inference speed suitable for clinical auxiliary diagnosis.

## Key Contributions
1. We propose **AVPDN**, a dedicated end-to-end framework for robust multi-scale polyp detection in dynamic colonoscopy videos, which directly addresses the motion artifacts and scale variation challenges in clinical colonoscopy scenarios.
2. We design the **Adaptive Feature Enhancement (AFE)** block, which integrates the AFIA and SACI modules to adaptively refine feature representations, suppress task-irrelevant noise, and capture the spatial distribution of polyps across different scales.
3. Extensive experiments on public benchmarks demonstrate that AVPDN consistently outperforms state-of-the-art object detection and specialized polyp detection methods, with significant improvements in detection accuracy and generalization capability.

## Method Overview
The overall architecture of AVPDN is shown below:
1. **Backbone**: A pre-trained ResNet-50 is used for hierarchical feature extraction from input colonoscopy frames.
2. **Adaptive Feature Enhancement (AFE) Blocks**: A series of AFE blocks are applied to multi-scale feature maps from the backbone, each consisting of the AFIA module and SACI module for noise-robust and scale-adaptive feature refinement.
3. **Transformer Decoder**: A deformable attention-based decoder inherited from RT-DETR, with cross-scale query selection and detection heads for end-to-end polyp localization and classification.

Detailed network architecture and methodology can be found in:
- [intro.pdf](https://github.com/xiaochen925/AVPDN/blob/main/intro.pdf): Full paper with methodology, experiments, and ablation studies
- [frame.pdf](https://github.com/xiaochen925/AVPDN/blob/main/frame.pdf): Detailed network framework diagram
- [vis.pdf](https://github.com/xiaochen925/AVPDN/blob/main/vis.pdf): Visualization results of detection performance

## Experimental Results
AVPDN is evaluated on two challenging public colonoscopy video datasets: **LDPolypVideo** and **CVC-VideoClinicDB**, and outperforms all compared state-of-the-art methods.

### Quantitative Comparison with SOTA Detectors
| Dataset | Model | AP (%) | Precision (%) | Recall (%) | F1-Score (%) | FPS |
|---------|-------|--------|---------------|------------|--------------|-----|
| **LDPolypVideo** | Faster R-CNN | 73.2 | 77.2 | 69.6 | 73.2 | 43.1 |
| | YOLOv8 | 92.9 | 93.1 | 91.6 | 92.3 | 54.0 |
| | YOLOv12 | 92.5 | 92.6 | 91.6 | 92.0 | 53.9 |
| | RT-DETR (Baseline) | 94.2 | 94.4 | 92.3 | 93.3 | 52.1 |
| | **AVPDN (Ours)** | **96.6** | **96.8** | **95.0** | **95.8** | **53.2** |
| **CVC-VideoClinicDB** | Faster R-CNN | 88.2 | 84.6 | 98.2 | 90.9 | 43.1 |
| | YOLOv8 | 90.9 | 92.2 | 87.8 | 89.9 | 54.0 |
| | YOLOv12 | 93.1 | 93.9 | 93.3 | 93.5 | 53.9 |
| | RT-DETR (Baseline) | 93.1 | 94.1 | 93.9 | 93.9 | 52.1 |
| | **AVPDN (Ours)** | **95.7** | **95.9** | **94.9** | **95.3** | **53.2** |

### Ablation Study Results (LDPolypVideo Dataset)
We validate the effectiveness of each core component of AVPDN through ablation experiments:

| AFIA Module | | | SACI Module | AP (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------------|---|---|-------------|--------|---------------|------------|--------------|
| CS | DSA | SSA | | | | | |
| - | - | - | - | 94.2 | 94.4 | 92.3 | 93.3 |
| ✓ | - | - | - | 94.6 | 94.6 | 92.7 | 93.6 |
| - | ✓ | - | - | 95.5 | 95.9 | 94.0 | 94.9 |
| - | - | ✓ | - | 95.2 | 96.0 | 93.6 | 94.7 |
| ✓ | ✓ | ✓ | - | 96.3 | 96.4 | 94.7 | 95.5 |
| ✓ | ✓ | ✓ | ✓ | **96.6** | **96.8** | **95.0** | **95.8** |

## Prerequisites
- Linux OS (tested on Ubuntu 20.04)
- Python 3.8+
- PyTorch 2.11+
- CUDA 11.7+ (for GPU acceleration)
- NVIDIA GeForce RTX 4090D (or equivalent GPU with ≥24GB VRAM)
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone this repository:
```bash
git clone https://github.com/xiaochen925/AVPDN.git
cd AVPDN
```

2. Create and activate a conda virtual environment:
```bash
conda create -n avpdn python=3.9
conda activate avpdn
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start
### 1. Data Preparation
We use two public colonoscopy video datasets for training and evaluation:
- **LDPolypVideo**: [Official Repository](https://github.com/dashishi/LDPolypVideo-Benchmark)
- **CVC-VideoClinicDB**: [Kaggle Download Link](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)

Download the datasets and organize them into the `./data/` directory following the structure below:
```
data/
├── LDPolypVideo/
│   ├── train/
│   ├── val/
│   └── test/
└── CVC-VideoClinicDB/
    ├── train/
    ├── val/
    └── test/
```

### 2. Inference with Pre-trained Model
1. Download the pre-trained model weights from the release page (or contact the author for reasonable requests) and place them in `./pretrained_models/`.
2. Modify the configuration file in `./configs/` to set the input path and model weight path.
3. Run inference on colonoscopy images/videos:
```bash
# Single image inference
python tools/inference.py --config configs/avpdn_infer.yaml --input_path ./data/test/ --output_path ./results/

# Video inference
python tools/inference_video.py --config configs/avpdn_video_infer.yaml --input_path ./data/test_video.mp4 --output_path ./results/
```

### 3. Training
1. Modify the training configuration file in `./configs/avpdn_train.yaml` to set dataset paths, batch size, learning rate, and other hyperparameters.
2. Start training:
```bash
python tools/train.py --config configs/avpdn_train.yaml
```
- Training logs and model checkpoints will be saved in `./output/`.
- An early stopping mechanism is enabled by default to prevent overfitting.

### 4. Evaluation
To evaluate the model on the test set:
```bash
python tools/evaluate.py --config configs/avpdn_eval.yaml --dataset LDPolypVideo --weights ./pretrained_models/avpdn_best.pth
```
The evaluation script will output AP, Precision, Recall, F1-Score, and FPS metrics.

## Project Structure
```
AVPDN/
├── configs/                # Configuration files for training, inference, and evaluation
├── src/                    # Core source code
│   ├── models/             # AVPDN network architecture, AFE/AFIA/SACI module implementations
│   ├── datasets/           # Dataset loaders and preprocessing pipelines
│   └── utils/              # Utility functions (metrics, logging, visualization, etc.)
├── tools/                  # Executable scripts for training, inference, and evaluation
├── data/                   # Dataset directory
├── pretrained_models/      # Pre-trained model weights
├── results/                # Inference output results
├── output/                 # Training logs and checkpoints
├── frame.pdf               # Network architecture diagram
├── intro.pdf               # Full paper of AVPDN
├── vis.pdf                 # Visualization results
├── requirements.txt        # Dependencies list
└── README.md               # This file
```

## Citation
If you use this code or our method in your research, please cite our work:
```bibtex
@article{chen2025avpdn,
  title={AVPDN: Learning Motion-Robust and Scale-Adaptive Representations for Polyp Detection in Dynamic Colonoscopy Frames},
  author={Chen, Zilin and Lu, Shengnan},
  journal={},
  year={2025}
}
```

## Data Availability
- LDPolypVideo dataset: https://github.com/dashishi/LDPolypVideo-Benchmark
- CVC-VideoClinicDB database: https://www.kaggle.com/datasets/balraj98/cvcclinicdb

## Acknowledgements
This work is funded by the 2025 Graduate Innovation Fund Project of Xi’an Shiyou University (Grant No. YCX2513160). We also thank the contributors of the LDPolypVideo and CVC-VideoClinicDB datasets for their open-source efforts.

## Contact
For any questions or issues, please contact Zilin Chen at chenzilin0925@163.com, or open an issue in this repository.
