# Multimodal and Sparse Recurrent Transformers for Real-Time Object Detection

**Master's thesis · MSc Robotics, Systems and Control · ETH Zürich, Center for Project-Based Learning (PBL), D-ITET · 2025**

[![Paper](https://img.shields.io/badge/arXiv-2607.18747-b31b1b.svg)](https://arxiv.org/abs/2607.18747)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB.svg)](setup_env.sh)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg)](setup_env.sh)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Real-time detection of small, fast drones by fusing an **event camera** stream with **RGB frames**.
The event branch is a Scene Adaptive Sparse Transformer (SAST), the RGB branch a YOLOX backbone;
both feature pyramids are fused before a shared YOLOX detection head. The code builds on the official
[SAST](https://github.com/Peterande/SAST) implementation (CVPR 2024) and extends it with multimodal
inputs, new detection heads and ONNX / TensorRT export for NVIDIA Jetson.

<p align="center">
  <img src="figures/eventrgb.png" width="750" alt="RGB + event fusion architecture: SAST and CSPDarknet branches fused at the FPN, YOLOX head">
  <br>
  <em>RGB + event fusion: SAST (events) and CSPDarknet (RGB) feature pyramids are summed and decoded by a YOLOX head.</em>
</p>

## TL;DR

- **Multimodal fusion beats both single modalities.** On [SkyEV](https://arxiv.org/abs/2607.18747) the fused
  SAST + RGB model reaches **45.8 mAP** (42.3 AP<sub>small</sub>), versus 35.5 mAP for RGB-only YOLOX and
  43.4 mAP for event-only SAST.
- **Sparse by design.** SAST keeps only the informative windows and tokens of the event stream; the thesis also
  explores reusing these sparsity masks to prune the tokens of an [LW-DETR](https://github.com/Atten4Vis/LW-DETR) detector.
- **Deployable.** Models export to ONNX and TensorRT (`export.py`) and were benchmarked for NVIDIA Jetson.
- **Published baseline.** The fusion architecture is the detection baseline of the SkyEV dataset paper
  (Mandula, Heusinger, Moosmann, Vogt, Magno; arXiv:2607.18747, 2026).

## Results

Detection results on the SkyEV test set (from the SkyEV paper, Table 3). All models are trained with this code base.
On a Jetson AGX Orin the fused model runs in **5.3 ms** per frame with TensorRT (31.3 M parameters, thesis Section 6.5).

| Model              | Input        | `model=`   | AP<sub>small</sub> | mAP<sub>50-95</sub> |
|--------------------|--------------|------------|-------------------:|--------------------:|
| YOLOX-RGB          | RGB          | `rgb`      | 32.9               | 35.5                |
| YOLOX-Events       | events       | `rgb`\*    | 32.2               | 37.3                |
| SAST               | events       | `rnndet`   | 40.7               | 43.4                |
| **SAST + RGB**     | RGB + events | `eventrgb` | **42.3**           | **45.8**            |

\* YOLOX fed with event histograms instead of RGB frames.

<details>
<summary>Earlier thesis evaluation, including the LW-DETR variants</summary>
<br>

Evaluation from the thesis on the *StStephan* subset of SkyEV. The thesis refers to SkyEV by its working title
*F-UAV-D*. These numbers are not directly comparable with the published table above.

<p align="center">
  <img src="figures/performance.png" width="750" alt="Earlier thesis results table">
</p>

</details>

### Datasets

- **[SkyEV](https://arxiv.org/abs/2607.18747)** — synchronized, uncompressed RGB + event recordings of eight drone
  types with strong camera ego-motion and very small targets (median box 41 px). The thesis and the preprocessing
  scripts use its working title *F-UAV-D* (`preprocessing/armasuisse.py`); *StStephan* is one of its subsets.
  The dataset is not publicly released yet; see the paper for details.
- **[NeRDD](https://github.com/MagriniGabriele/NeRDD)** — Neuromorphic RGB-Event Drone Detection dataset, used as a
  second benchmark.

Both are converted to a common on-disk format by the scripts in [`preprocessing/`](preprocessing/README.md)
(event histograms, RGB frames and labels as paired `.npz` files per sequence).

## Architectures

| `model=`   | Description                                                                              | Input        |
|------------|------------------------------------------------------------------------------------------|--------------|
| `rnndet`   | Base SAST: sparse recurrent transformer backbone + YOLOX FPN/head                        | events       |
| `rgb`      | Base YOLOX (CSPDarknet + PAFPN + head)                                                   | RGB (or events) |
| `eventrgb` | **Fusion**: SAST branch + CSPDarknet branch, summed at the FPN, one YOLOX head (figure above) | RGB + events |
| `lwdetr`   | LW-DETR whose ViT tokens are sparsified with the masks predicted by SAST (figure below)  | events       |

<p align="center">
  <img src="figures/lwdetrsast.png" width="750" alt="SAST sparsity masks used to prune LW-DETR tokens">
  <br>
  <em>SAST as a mask generator: its window/token selection sparsifies the LW-DETR transformer.</em>
</p>

## Getting started

### Installation

```bash
./setup_env.sh        # creates the conda env "sast" (Python 3.9, PyTorch 2.0, CUDA 11.8, Lightning, Hydra)
conda activate sast
```

Detectron2 is optional but speeds up the COCO-style evaluation.
The LW-DETR deformable-attention CUDA ops are built by the setup script (`models/detection/lwdetr/ops`).

### Pre-trained checkpoints

| YOLOX | SAST | LW-DETR |
|-------|------|---------|
| [YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX) | [1 Mpx](https://github.com/Peterande/SAST) | [LWDETR_tiny_30e_objects365](https://github.com/Atten4Vis/LW-DETR) |

Set the checkpoint paths in the corresponding `config/model/*.yaml`.

### Training

Configuration is handled by [Hydra](https://hydra.cc); every value below can be overridden on the command line.

```bash
# batch size, GPUs, learning rate and DATA_DIR
source set_envs.sh

# fusion model on SkyEV / NeRDD (preprocessed, dataset=eventrgb)
python train.py model=eventrgb dataset=eventrgb dataset.path="${DATA_DIR}" \
  +experiment/arma="base.yaml" \
  batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
  hardware.gpus=[${GPUS}] training.learning_rate=${lr} \
  wandb.project_name=SkyEV wandb.group_name=eventrgb

# RGB-only YOLOX baseline with a pretrained backbone
python train.py model=rgb dataset=eventrgb dataset.path="${DATA_DIR}" \
  +experiment/arma="base.yaml" fpn.ckpt=yolox_s.pth \
  batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
  hardware.gpus=[${GPUS}] training.learning_rate=${lr}
```

`+experiment/gen4=...` and `+experiment/gen1=...` select the Prophesee Gen4 / Gen1 settings inherited from SAST;
`+experiment/lwdetr=...` configures the LW-DETR variants. Training logs go to Weights & Biases.

### Evaluation

```bash
python validation.py model=eventrgb dataset=eventrgb dataset.path="${DATA_DIR}" \
  checkpoint=<path/to/ckpt> hardware.gpus=${GPUS} use_test_set=True   # single GPU only
```

### Deployment: ONNX and TensorRT

```bash
python export.py          # exports the model from config/detect_lwdetr.yaml to ONNX (onnxsim) and builds a TensorRT engine
python run_onnx.py        # latency benchmark of model.onnx with onnxruntime (CUDA execution provider)
```

The [`tensorrt`](https://github.com/Heusini/SAST/tree/tensorrt) branch contains the model changes needed for a
clean TensorRT conversion of SAST, plus benchmark scripts.

## Repository layout

```
config/          Hydra configs: dataset, model, experiment presets
data/            Dataset classes (event_rgb, genx), representations, augmentation
preprocessing/   Converters for SkyEV and NeRDD into the paired .npz format (+ README)
models/
  detection/
    event_rgb/   Fusion detector (SAST + CSPDarknet → YOLOX head)
    lwdetr/      LW-DETR with SAST-driven token sparsification
    rgb_yolo/    YOLOX RGB detector
    recurrent_backbone/  SAST recurrent backbone
  layers/SAST/   Scene Adaptive Sparse Transformer layers
modules/         PyTorch Lightning modules (one *_step.py per model type)
utils/, util/    Evaluation (Prophesee / COCO), optimizers, timers, helpers
train.py, validation.py, export.py, run_onnx.py
```

## Citation

```bibtex
@mastersthesis{heusinger2025multimodal,
  author = {Sebastian Heusinger},
  title  = {Multimodal and Sparse Recurrent Transformers for Real-Time Object Detection},
  school = {ETH Z{\"u}rich, Center for Project-Based Learning, D-ITET},
  year   = {2025},
}

@article{mandula2026skyev,
  author  = {Jakub Mandula and Sebastian Heusinger and Julian Moosmann and Christian Vogt and Michele Magno},
  title   = {SkyEV: RGB-Event UAV Detection and Tracking Dataset and Baseline},
  journal = {arXiv preprint arXiv:2607.18747},
  year    = {2026},
}
```

## Built on

This repository started as a modified copy of [SAST](https://github.com/Peterande/SAST)
(Peng et al., *Scene Adaptive Sparse Transformer for Event-based Object Detection*, CVPR 2024) and uses code from:

- [SAST](https://github.com/Peterande/SAST) — SAST architecture
- [RVT](https://github.com/uzh-rpg/RVT) — recurrent vision transformer implementation and data pipeline
- [timm](https://github.com/huggingface/pytorch-image-models) — original MaxViT layers
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) — CSPDarknet backbone, PAFPN and detection head
- [LW-DETR](https://github.com/Atten4Vis/LW-DETR) — lightweight DETR detector and export tooling

Licensed under the [MIT License](LICENSE).
