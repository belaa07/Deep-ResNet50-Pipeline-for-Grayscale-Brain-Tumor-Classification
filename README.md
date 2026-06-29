# Grayscale ResNet50 — 3-Class MRI Classification (512×512)

A TensorFlow/Keras pipeline for classifying grayscale MRI images into three classes using a transfer-learned ResNet50 backbone adapted for single-channel (512×512×1) input.

---

## Overview

Standard ResNet50 is pretrained on ImageNet with 3-channel (RGB) input. This project adapts that backbone for **grayscale medical images** by collapsing the first convolutional layer's weights across the channel axis, preserving as much pretrained knowledge as possible while supporting 1-channel input at full 512×512 resolution.

---

## Project Structure

```
project/
├── data/
│   ├── 1/          # Class 1 — .npy image arrays
│   ├── 2/          # Class 2 — .npy image arrays
│   └── 3/          # Class 3 — .npy image arrays
└── train.py        # Main training script
```

Each sample is stored as a `.npy` file containing a float32 array of shape `(512, 512)` or `(512, 512, 1)`.

---

## Requirements

```
tensorflow >= 2.10
numpy
scikit-learn
```

Install with:

```bash
pip install tensorflow numpy scikit-learn
```

A CUDA-compatible GPU is strongly recommended. The script checks for GPU availability at startup and prints `nvidia-smi` output if found.

---

## Dataset Format

- **Directory layout:** one subfolder per class, named `1`, `2`, `3`
- **File format:** `.npy` arrays, each of shape `(512, 512)` or `(512, 512, 1)`, dtype `float32`
- **Set `DATA_DIR`** in the script to the root folder containing the three class subfolders

---

## Pipeline

### 1. Data Discovery & Stratified Splitting

All `.npy` file paths are collected and split into:

| Split      | Proportion |
|------------|-----------|
| Train      | 70%       |
| Validation | 15%       |
| Test       | 15%       |

Splits are stratified by class label and seeded at `random_state=42` for reproducibility.

### 2. Lazy Loading via `tf.py_function`

Images are loaded on-the-fly using `np.load` wrapped in a `tf.py_function`, avoiding loading the full dataset into memory. Each image is cast to `float32` and assigned a fixed shape of `(512, 512, 1)`.

### 3. Data Augmentation (training only)

Applied after batching on the training set:

| Transform           | Parameters                      |
|---------------------|---------------------------------|
| Random Horizontal Flip | —                            |
| Random Rotation     | ±15% of 2π, reflect fill        |
| Random Translation  | ±5% height and width            |
| Random Zoom         | ±10%                            |
| Random Brightness   | ±15%                            |
| Random Contrast     | ±15%                            |

### 4. Model Architecture

**Backbone:** ResNet50 with pretrained ImageNet weights, adapted for grayscale:

- The RGB first-conv kernel `(7, 7, 3, 64)` is averaged across the channel axis → `(7, 7, 1, 64)`, preserving learned edge/texture detectors
- All other layer weights copied directly from the RGB checkpoint
- Last 15 layers unfrozen for fine-tuning; earlier layers frozen

**Classification Head:**

```
GlobalAveragePooling2D  ─┐
                          ├─► Concatenate → Dense(256, ReLU) → BN → Dropout(0.4) → Dense(3, Softmax)
GlobalMaxPooling2D      ─┘
```

The dual-pooling concatenation captures both mean activation (global context) and peak activation (salient features).

### 5. Training Configuration

| Parameter        | Value                          |
|------------------|-------------------------------|
| Optimizer        | Adam                          |
| Learning rate    | 1e-4                          |
| Loss             | Sparse Categorical Crossentropy |
| Batch size       | 32                            |
| Epochs           | 25                            |
| Input shape      | 512 × 512 × 1                 |

---

## Usage

1. Set `DATA_DIR` to the root of your dataset directory.
2. Run the script:

```bash
python train.py
```

3. Training and validation metrics are logged per epoch. After training, final test accuracy is printed.

---

## Key Design Decisions

**Why no resizing to 224×224?**  
Medical images contain fine-grained spatial detail (e.g., tissue boundaries, lesion morphology) that is lost at lower resolutions. The pipeline retains the native 512×512 resolution throughout.

**Why average the first conv weights instead of training from scratch?**  
Averaging RGB → grayscale preserves the pretrained low-level feature detectors (edges, textures) while making them applicable to a single input channel, giving a better initialization than random weights.

**Why GAP + GMP concatenation?**  
Global Average Pooling captures distributed activations; Global Max Pooling highlights the most strongly activated spatial location. Their concatenation gives the classifier richer summarization of the feature maps.

---

## Notes

- Ensure sufficient GPU VRAM (≥8 GB recommended) for 512×512 inputs at batch size 32; reduce `BATCH_SIZE` if OOM errors occur
- `tf.keras.backend.clear_session()` and `gc.collect()` are called at startup to avoid stale graph accumulation in notebook environments
- The script is written for use in a Colab/Jupyter environment but runs as a plain Python script with minor adaptation (remove `!nvidia-smi`)
