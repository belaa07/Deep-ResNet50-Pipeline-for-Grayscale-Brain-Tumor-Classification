# Brain Tumor MRI Classification — ResNet50 (Grayscale, 512×512)

A TensorFlow/Keras pipeline for classifying brain tumor MRI scans into three types — **Meningioma**, **Glioma**, and **Pituitary** — using a transfer-learned ResNet50 backbone adapted for single-channel (512×512×1) input.

---

## Dataset

This project uses the **Figshare Brain Tumor Dataset**, originally created by **Jun Cheng**. It comprises 3,064 T1-weighted contrast-enhanced MRI slices collected from 233 patients, covering three tumor types: Meningioma, Glioma, and Pituitary.

- **Source:** [https://figshare.com/articles/dataset/brain_tumor_dataset/1512427](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- **Imaging modality:** T1-weighted contrast-enhanced MRI
- **Patients:** 233
- **Total slices:** 3,064

If you use this dataset in your work, please cite the original creator:

> Jun Cheng. (2017). *brain_tumor_dataset*. figshare. Dataset. https://doi.org/10.6084/m9.figshare.1512427

```
project/
├── dataset/
│   ├── Meningioma/        # 708 .npy MRI arrays
│   ├── Glioma/            # 1,426 .npy MRI arrays
│   └── Pituitary/         # 930 .npy MRI arrays
├── results/
│   ├── confusion_matrix    # Saved confusion matrix plot
│   └── result_csv          # accuracy and f1 scores
└── brain_tumor_classification.ipynb   # Full pipeline notebook
```

**Total dataset size:** 3,064 samples across 3 classes

> **Class imbalance note:** Glioma (1,426) is ~2× more frequent than Meningioma (708). The stratified splitting strategy preserves this ratio across train/val/test sets.

---

## Classes

| Label | Tumor Type   | Sample Count |
|-------|--------------|-------------|
| 0     | Meningioma   | 708         |
| 1     | Glioma       | 1,426       |
| 2     | Pituitary    | 930         |

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

A CUDA-compatible GPU is strongly recommended. The notebook checks for GPU availability at startup and prints `nvidia-smi` output if one is found.

---

## Dataset Format

- **File format:** `.npy` arrays, each of shape `(512, 512)` or `(512, 512, 1)`, dtype `float32`
- **Directory layout:** one subfolder per tumor class under `dataset/`
- Set `DATA_DIR` in the notebook to the path of the `dataset/` folder before running

---

## Pipeline

### 1. Data Discovery & Stratified Splitting

All `.npy` file paths are collected and split into:

| Split      | Proportion | Approx. Samples |
|------------|-----------|----------------|
| Train      | 70%       | ~2,145         |
| Validation | 15%       | ~460           |
| Test       | 15%       | ~460           |

Splits are stratified by class label and seeded at `random_state=42` for reproducibility.

### 2. Lazy Loading via `tf.py_function`

Images are loaded on-the-fly using `np.load` wrapped in a `tf.py_function`, so the full dataset is never held in memory simultaneously. Each image is cast to `float32` with a fixed shape of `(512, 512, 1)`.

### 3. Data Augmentation 

Applied after batching on the training set:

| Transform              | Parameters                 |
|------------------------|----------------------------|
| Random Rotation        | ±15% of 2π, reflect fill   |
| Random Translation     | ±5% height and width       |
| Random Zoom            | ±10%                       |
| Random Brightness      | ±15%                       |
| Random Contrast        | ±15%                       |

### 4. Model Architecture

**Backbone:** ResNet50 with pretrained ImageNet weights, adapted for grayscale input:

- The RGB first-conv kernel `(7, 7, 3, 64)` is averaged across the channel axis → `(7, 7, 1, 64)`, preserving pretrained edge and texture detectors
- All other layer weights are copied directly from the ImageNet checkpoint
- Last 15 layers are unfrozen for fine-tuning; all earlier layers are frozen

**Classification Head:**

```
GlobalAveragePooling2D  ─┐
                          ├─► Concatenate → Dense(256, ReLU) → BatchNorm → Dropout(0.4) → Dense(3, Softmax)
GlobalMaxPooling2D      ─┘
```

The dual-pooling concatenation captures both distributed activation (GAP) and peak salient features (GMP).

### 5. Training Configuration

| Parameter        | Value                           |
|------------------|---------------------------------|
| Learning rate    | 1e-4                            |
| Loss             | Sparse Categorical Crossentropy |
| Batch size       | 32                              |
| Epochs           | 25                              |
| Input shape      | 512 × 512 × 1                   |

---

## Usage

1. Open `brain_tumor_classification.ipynb` in Jupyter or Google Colab.
2. Set `DATA_DIR` to the path of your `dataset/` folder
3. Run all cells. Training and validation metrics are logged per epoch.

---

## Key Design Decisions

**Why average the first conv weights instead of random initialization?**
Averaging the RGB pretrained kernel across channels preserves the low-level feature detectors (edges, gradients, textures) learned on ImageNet, giving a substantially better initialization than random weights for a single-channel input.

**Why GAP + GMP concatenation in the head?**
Global Average Pooling summarizes distributed activations across the spatial map; Global Max Pooling captures the single most strongly activated location. Their concatenation gives the classifier a richer feature summary than either alone.

---
