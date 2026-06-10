This repository contains code for **Differentially Private 2D Human Pose Estimation (CVPR 2026)**:
> **Differentially Private 2D Human Pose Estimation**  
> Paper link:(https://openaccess.thecvf.com/content/CVPR2026/papers/Sivangi_Differentially_Private_2D_Human_Pose_Estimation_CVPR_2026_paper.pdf)

The codebase provides differential privacy baselines and feature-projective DP method for human pose estimation on MPII, and HumanArt datasets.

## Repository Structure

```text
.
├── DP_base.py
├── feature_DP.py
├── Feature_projective_DP.py
├── requirements.txt
├── utils/
├── datautils/
├── TinyVit_mod.py
└── README.md
```

### Main Scripts

| Script | Description |
| --- | --- |
| `DP_base.py` | Differential privacy baseline for human pose estimation. |
| `feature_DP.py` | Feature-based differential privacy benchmark for human pose estimation. |
| `Feature_projective_DP.py` | Main Feature Projective  DP method. |

The paths inside these scripts should be updated before running. In particular, update paths for datasets, config files, pretrained checkpoints, output checkpoints, and evaluation output directories.

Config files are located in the `utils/` folder.
## Installation

Create an environment with Python and install the project requirements:

```bash
pip install -r requirements.txt
```

## Data Preparation

### COCO and MPII

Prepare COCO and MPII following the official HRNet Human Pose Estimation repository:

[HRNet Human Pose Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)

### HumanArt

Prepare HumanArt using the official HumanArt repository:

[HumanArt](https://github.com/idea-research/humanart)

The HumanArt preparation process is similar to COCO-style human pose estimation data preparation.

After preparing the datasets, update the dataset root paths inside the training scripts.

## Checkpoints

The COCO pretrained checkpoint is provided at [https://drive.google.com/file/d/17WLCncC3kEA0Mgzxvsf9MQuKHrYLx7P2/view?usp=sharing].

Before training, update the checkpoint path inside the relevant script:

```python
pretrained_checkpoint = "INSERT_PATH_TO_TINYVIT_PRETRAINED_CHECKPOINT"
```

Update the output checkpoint path as well:

```python
output_checkpoint = "INSERT_PATH_TO_OUTPUT_CHECKPOINT"
```

## Usage

Update the paths inside the script you want to run:

```python
config_file = "INSERT_PATH_TO_CONFIG_FILE"
dataset_root = "INSERT_PATH_TO_DATASET_ROOT"
pretrained_checkpoint = "INSERT_PATH_TO_PRETRAINED_CHECKPOINT"
output_checkpoint = "INSERT_PATH_TO_OUTPUT_CHECKPOINT"
eval_output_dir = "INSERT_PATH_TO_EVAL_OUTPUT_DIR"
```

### DP Baseline

```bash
python DP_base.py
```

This runs the differential privacy baseline benchmark.

### Feature DP Benchmark

```bash
python feature_DP.py
```

This runs the feature differential privacy benchmark.

### Projective Feature DP Method

```bash
python Feature_projective_DP.py
```

This runs the main feature projective DP method.

## Notes

- Dataset paths and config paths must be updated before running.
- Config files are located in `utils/`.
- The scripts assume HumanArt-style COCO-format annotations for HumanArt experiments.
- CUDA device settings may need to be changed depending on your system.
- Batch size, privacy budget, clipping norm, and projection settings can be adjusted inside each script.

## References

This repository builds on code and ideas from the following projects. 

- https://github.com/HRNet/HRNet-Human-Pose-Estimation
- https://github.com/idea-research/humanart
- https://github.com/wkcn/TinyViT
- https://github.com/meta-pytorch/opacus
- 
## Citation
```bibtex
@inproceedings{sivangi2026differentially,
  title={Differentially private 2D human pose estimation},
  author={Sivangi, Kaushik Bhargav and Henderson, Paul and Deligianni, Fani},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21143--21153},
  year={2026}
}
```
