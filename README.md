# Brain Tumor Classification — Parallel Deep Feature Extraction

**Parallel Deep Feature Extraction Using MobileNetV2, VGG16, and DenseNet121 with SVM Fusion for Brain Tumor Classification**

This repository contains the code, figures, and results for a parallel deep learning framework that classifies brain tumors from MRI scans. Instead of running convolutional neural networks sequentially, **MobileNetV2**, **VGG16**, and **DenseNet121** extract deep features **concurrently** via multi-threaded GPU execution. The three feature vectors are fused into a single 2,816-dimensional representation and classified using a **Support Vector Machine (SVM)** with an RBF kernel. The proposed method achieves **94.81% accuracy** and a **6.49× speedup** over a sequential baseline on a four-class MRI dataset.

---

## Table of Contents

- [Research Highlights](#research-highlights)
- [Proposed Architecture](#proposed-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Comparison with Existing Literature](#comparison-with-existing-literature)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [Contact / Author](#contact--author)

---

## Research Highlights

- **Parallel feature extraction:** MobileNetV2, VGG16, and DenseNet121 run concurrently using a `ThreadPoolExecutor` (max_workers=3) across two NVIDIA Tesla T4 GPUs, instead of the sequential pipelines used in prior work.
- **Feature-level fusion:** Deep features from all three backbones (1,280 + 512 + 1,024 dimensions) are concatenated via `numpy.hstack()` into a unified 2,816-dimensional representation, preserving complementary spatial and semantic information.
- **SVM classification:** An RBF-kernel SVM (C = 10, class weight = balanced) classifies the fused, standardized feature vectors across four tumor classes.
- **Significant speedup:** The parallel pipeline reduces total feature extraction time from 2025.84s (sequential) to 312.30s — a **6.49× speedup** — with no loss in classification accuracy.
- **Comparable accuracy with added efficiency:** Achieves 94.81% accuracy on a more complex four-class problem using frozen transfer-learning weights, explicitly quantifying the computational trade-off that most prior studies do not report.

---

## Proposed Architecture

**Pipeline:** MRI image → Resize (224×224×3) → Model-specific preprocessing → Parallel feature extraction (MobileNetV2 / VGG16 / DenseNet121) → GlobalAveragePooling2D → Feature fusion (concatenation) → Feature scaling (StandardScaler) → Label encoding → RBF-kernel SVM → Classification (Glioma / Meningioma / Pituitary Tumor / No Tumor)

**Key design details:**
- Each MRI scan is resized to 224×224×3 and processed with model-specific normalization: MobileNetV2 scales pixels to [-1, 1], VGG16 subtracts ImageNet channel means, and DenseNet121 normalizes using the ImageNet mean and standard deviation.
- All three backbones use **frozen ImageNet weights** with a `GlobalAveragePooling2D` layer to convert 3D feature maps into 1D vectors (1,280-D, 512-D, and 1,024-D respectively).
- Sequential execution time is the **sum** of individual model times; parallel execution time is bounded by the **slowest** model (`max` instead of `sum`), which is the core source of the speedup.

See `fig/proposed.png` for the full architecture diagram.

---

## Dataset

- **Source (Kaggle):** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total images:** 7,200
- **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor

| Class | Training Images | Testing Images | Total Images |
|---|---|---|---|
| Glioma | 1,400 | 400 | 1,800 |
| Meningioma | 1,400 | 400 | 1,800 |
| Pituitary Tumor | 1,400 | 400 | 1,800 |
| No Tumor | 1,400 | 400 | 1,800 |
| **Total** | **5,600** | **1,600** | **7,200** |

Sample images for each class are available in `fig/sample.png`.

---

## Results

### Classification Performance (Parallel DL + SVM)

| Metric | Score |
|---|---|
| Accuracy | 94.81% |
| Precision (weighted avg) | 95.17% |
| Recall (weighted avg) | 94.81% |
| F1-Score (weighted avg) | 94.70% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma | 98.49% | 81.50% | 89.19% |
| Meningioma | 88.76% | 98.75% | 93.49% |
| No Tumor | 93.68% | 100.00% | 96.74% |
| Pituitary | 99.75% | 99.00% | 99.37% |

### Timing Comparison — Sequential vs Parallel

| Mode | MobileNetV2 | VGG16 | DenseNet121 | Total | Speedup |
|---|---|---|---|---|---|
| Sequential | 655.43s | 674.02s | 696.38s | 2025.84s | 1.00× |
| **Parallel** | 184.37s | 55.41s | 301.35s | **312.30s** | **6.49×** |

Confusion matrix and per-class precision/recall/F1 bar chart are available in `results/confusion.png` and `results/graph.png`.

---

## Comparison with Existing Literature

| Study | Method | Classes | Execution | Speed Up | Accuracy |
|---|---|---|---|---|---|
| Badža et al. | Custom CNN | 3 | Sequential | Not reported | 96.56% |
| Irmak et al. | Grid-Optimized Multi-CNN | 5 | Sequential | Not reported | 92.66% |
| Gunasekara et al. | Cascade CNN | 2 | Sequential | Not reported | 92.31% |
| Lamrani et al. | Custom CNN | 2 | Sequential | Not reported | 96.33% |
| Rahman et al. | Parallel Deep CNN | 4 | Parallel | Not reported | 98.12% |
| Remzan et al. | RadImageNet CNN | 4 | Sequential | Not reported | 97.71% |
| Sadr et al. | Custom Deep CNN | 3 | Parallel | Not reported | 97.27% |
| **Proposed** | **MobileNetV2 + VGG16 + DenseNet121 (parallel) + SVM** | **4** | **Parallel (T4×2)** | **6.49×** | **94.81%** |

This work is the only study in the comparison that explicitly measures and reports a quantified speedup for its parallel implementation under controlled, same-hardware conditions.

---

## Repository Structure

```
Brain-Tumor(Parallel)/
│
├── code/
│   ├── parallel.ipynb        # Proposed parallel feature extraction pipeline (MobileNetV2 + VGG16 + DenseNet121)
│   ├── proposed.ipynb        # Full proposed methodology: fusion + SVM classification
│   └── sequential.ipynb      # Sequential baseline pipeline for timing comparison
│
├── fig/
│   ├── sample.png            # Sample MRI images (Glioma, Meningioma, Pituitary, No Tumor)
│   └── proposed.png          # Proposed architecture diagram
│
├── results/
│   ├── confusion.png         # Confusion matrix (Parallel DL + SVM)
│   └── graph.png             # Per-class precision, recall, and F1-score bar chart
│
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.12
- TensorFlow/Keras 2.x
- scikit-learn 1.x
- Jupyter Notebook / JupyterLab
- GPU (NVIDIA CUDA-enabled) recommended for parallel execution; experiments were run on 2× NVIDIA Tesla T4 GPUs

### Windows

```bash
git clone https://github.com/<your-username>/Brain-Tumor-Parallel.git
cd Brain-Tumor-Parallel
python -m venv venv
venv\Scripts\activate
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn jupyter
jupyter notebook
```

### Linux

```bash
git clone https://github.com/<your-username>/Brain-Tumor-Parallel.git
cd Brain-Tumor-Parallel
python3 -m venv venv
source venv/bin/activate
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn jupyter
jupyter notebook
```

1. Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle and place it according to the path structure expected in the notebooks.
2. Run `code/sequential.ipynb` to reproduce the sequential baseline timings.
3. Run `code/parallel.ipynb` to reproduce the parallel feature extraction pipeline.
4. Run `code/proposed.ipynb` to reproduce feature fusion, SVM training, and final classification results.

---

## Limitations

- The three CNN backbones are used with **frozen ImageNet weights**; no task-specific fine-tuning was applied, which may limit peak classification accuracy compared to fully fine-tuned models.
- The framework was evaluated on a **single publicly available dataset**; performance on other clinical or multi-institutional MRI datasets was not tested.
- The reported speedup (6.49×) is hardware-specific, measured on a dual NVIDIA Tesla T4 configuration on Kaggle's cloud platform, and may vary on different hardware setups.
- The study addresses a four-class classification problem; tumor sub-typing, grading, or segmentation were not within scope.

---

## Future Work

- Apply data augmentation and task-driven fine-tuning of unfrozen backbone weights to further improve classification accuracy.
- Evaluate the framework on additional, diverse clinical MRI datasets.
- Extend the parallel ensemble with additional CNN architectures.
- Explore clinical deployment pathways for the proposed pipeline.

---

## Citation

If you use this work, please cite the paper (citation details will be updated once officially published):

```
C. M. Abir, S. Nagadevi, P. Das, N. Singh, and S. Biswas,
"Parallel Deep Feature Extraction Using MobileNetV2, VGG16, and DenseNet121
with SVM Fusion for Brain Tumor Classification," (accepted, in press).
```

---

## Contact / Author

**Chayan Mondal Abir**
Department of Computer Science and Engineering
SRM Institute of Science and Technology, Kattankulathur, India
📧 abirchayan2000@gmail.com

For questions, collaborations, or issues related to this repository, feel free to reach out via email or open an issue on this repository.
