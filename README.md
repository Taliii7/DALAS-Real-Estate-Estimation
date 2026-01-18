# 🏠 DALAS: Multimodal Real Estate Valuation via Deep Learning

> A State-of-the-Art Hybrid Architecture combining **Computer Vision (DINOv2)**, **NLP (CamemBERT)**, and **Gradient Boosting (XGBoost)** to estimate real estate prices in France with 97% accuracy.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU_Hist-red)](https://xgboost.readthedocs.io/)
[![Computer Vision](https://img.shields.io/badge/CV-DINOv2_%26_CLIP-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## 📄 Technical Report
This project is based on extensive research involving the scraping of 675k listings and 3M+ images. For a detailed explanation of the **Masked Multi-Task Loss** and the **Visual Premium** analysis, please refer to the full report:

👉 **[Read the Full Technical Report (PDF)](./docs/Rapport.pdf)**

---

## 🚀 Project Overview
Traditional Automated Valuation Models (AVMs) rely heavily on tabular data (surface, location). DALAS bridges the semantic gap by integrating **unstructured data** (images and descriptions) to capture the "intrinsic condition" of a property.

**Core Innovation: The Two-Stage Hybrid Pipeline**
1.  **Neural Feature Extraction:** A custom backbone fuses visual features (via **ConvNeXt/DINOv2**) and textual features (via **CamemBERT**) to learn a high-dimensional latent representation of the property.
2.  **Gradient Boosting Regressor:** These embeddings are fed into an **XGBoost** model (trained with `gpu_hist`) to handle non-linear geographic interactions and minimize error.


## 👥 Team & Credits
Project developed at **Sorbonne Université** (Master of Computer Science).
- [**Maël**](https://github.com/Mael-lcn)
- [**Ali**](https://github.com/Taliii7)


## 📊 Key Results
We achieved State-of-the-Art performance on the French rental market, quantitatively proving that **visual features reduce estimation error by ~30%**.

| Market Segment | Model Architecture | $R^2$ Score | MAE (Mean Absolute Error) |
| :--- | :--- | :--- | :--- |
| **Rental (Location)** | **Hybrid (Ours)** | **0.973** | **36.96 €** |
| Rental (Location) | Tabular Baseline | 0.954 | 53.40 € |
| **Sales (Achat)** | **Hybrid (Ours)** | **0.760** | **~64k €** |

---




## 🛠 Repository Structure

The codebase is organized to separate data acquisition, analysis, and modeling logic:

```text
DALAS/
├── analyse/                  # 📊 Exploratory Data Analysis (EDA)
│   ├── analyse_bivariee.py   # Correlation matrices & ANOVA tests
│   ├── analyse_multivariee.py# PCA & Dimensionality reduction analysis
│   └── tools.py              # Statistical utility functions
├── data_acquisition/         # 🕷️ Distributed Scraping Pipeline
│   ├── get_image.py          # Asynchronous image downloader
│   └── dataset_stat.py       # Data volume monitoring
├── images_process/           # 🖼️ Computer Vision Pipeline
│   ├── ai_part.py            # DINOv2 Feature Extraction logic
│   └── filter_images.py      # Zero-Shot Semantic Filtering (CLIP)
├── model/                    # 🧠 Deep Learning & Training Core
│   ├── model.py              # PyTorch Backbone definition (Multi-modal)
│   ├── train.py              # Training loop with Masked Multi-Task Loss
│   ├── my_xgboost.py         # Stage 2: Boosting Regressor implementation
│   ├── data_loader.py        # Custom Dataset class with Robust Scaling
│   └── eval.py               # Inference & Metrics calculation
├── dataset/                  # Dataset generation scripts
└── environment.yml           # Conda environment configuration
