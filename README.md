# Deep Learning for Perceptual Image Similarity in Dermatology Image Regions

## Overview

This repository contains the code and experiments for the diploma thesis **"Deep Learning for Perceptual Image Similarity in Dermatology Image Regions"**.

The main goal of this thesis is to investigate **perceptual image similarity in dermatology** by leveraging **deep learning models**, including **convolutional neural networks (CNNs)** (such as VGG16 and ResNet50) and **Vision Transformers (ViT)**.

This work focuses on analyzing and comparing **dermoscopic and clinical skin image regions**, using modern perceptual similarity metrics such as **DISTS** and **LPIPS**. The objective is to study how these similarity measures behave on dermatology images and to evaluate how well they align with **expert dermatologists’ judgments**.

Beyond perceptual metrics, the thesis also includes the **development, fine-tuning, and evaluation** of deep learning architectures for dermatology image understanding, providing both quantitative results and tools (including an interactive GUI) to explore similarity between regions of interest.


The models are trained and evaluated using:
- The **ISIC** skin lesion dataset.
- An additional **clinical dataset from the University Hospital of Ioannina**, containing real-world dermatology images.

## Objectives

- Develop and fine-tune deep learning models (CNNs and transformers) for dermatology image analysis.
- Study **perceptual image similarity** in dermatology image regions and its relation to skin lesion severity.
- Evaluate and compare different architectures (ResNet50, VGG16, Vision Transformers, and classical ML baselines) on ISIC and hospital data.
- Provide an experimental framework that can be reused or extended by other researchers.

## Datasets

### ISIC Dataset

- Public dermoscopic image dataset with multiple skin lesion classes.
- Used for training, validation and testing of deep learning models.
- The dataset itself is **not included** in this repository. Please download it from the official ISIC sources and place it under `datasets/` (or follow the instructions in the corresponding notebook/script).

### University Hospital of Ioannina Dataset

- Private dataset of dermatology images collected at the **University Hospital of Ioannina**.
- Used for additional training/validation and for more realistic evaluation.
- Due to privacy and ethical constraints, these images **cannot be shared** in this repository.

## Methods and Models

This project explores multiple families of models and techniques, with a particular focus on **VGG16**, **ResNet50**, and **Vision Transformers (ViT)** as the main backbone architectures:

- **Convolutional Neural Networks (CNNs)**
  - VGG16
  - ResNet50
- **Vision Transformers (ViT)** and other transformer-based vision models
- **Perceptual image similarity metrics**
  - **DISTS** (Deep Image Structure and Texture Similarity)
  - **LPIPS** (Learned Perceptual Image Patch Similarity)
- **Classical machine learning models** using features extracted from deep networks
- Comparison and analysis across different architectures, similarity metrics, and training setups.

In all perceptual similarity experiments, the models produce a **continuous similarity score in the range [0, 1]**, which is interpreted in a way that aims to approximate **human perceptual judgement**:

- Scores **close to 0** correspond to **small distance / high similarity** (images or regions that appear very similar to a human observer).
- Scores **close to 1** correspond to **large distance / low similarity** (images or regions that are perceptually different according to human judgement).

Throughout the thesis, we **systematically explored different architectural and similarity variants**, such as:

- Modifying the backbone networks (e.g. removing or changing layers).
- Changing how similarity is computed or aggregated across layers/features.
- Comparing the behaviour of different backbone architectures and similarity formulations with respect to perceptual consistency.

## Technologies and Tools

The experiments combine several libraries and tools (depending on the folder / experiment):

- **Python**
- **TensorFlow / Keras**
- **PyTorch**
- **Scikit-learn**
- **OpenCV**
- **Jupyter Notebook** and **Google Colab**
- **CUDA / GPU acceleration** (for training speed, when available)

Not all components necessarily use every library above, but these are the core technologies used across the project.

## Repository Structure (High-Level)

The top-level folders in this project are organised so that each name reflects its content:

- `datasets/` – Local copies of ISIC and related datasets, together with Python scripts for dataset preparation, cleaning and pre-processing. **Datasets themselves are not tracked in git**.
- `DISTS-master/` – Upstream / reference code for DISTS (Deep Image Structure and Texture Similarity).
- `LPIPS-master/` – Upstream / reference code for LPIPS (Learned Perceptual Image Patch Similarity).
- `DISTS_FINE_TUNE/` – Fine-tuning experiments for DISTS or related perceptual similarity models on dermatology images.
- `GoogleCollab/` – Jupyter/Colab notebooks, including experiments with training the ViT model in a notebook environment.
- `ResNet50/` – Python programs and experiments using the ResNet50 architecture on ISIC and hospital data.
- `VGG16/` – Python programs and experiments using the Vgg16 architecture.
- `ViT/` – Python programs and experiments using Vision Transformers.
- `Scikit/` – Experiments implemented with scikit-learn (e.g. classical ML models based on deep features or handcrafted features).
- `OpenCV/` – Utilities for image pre-processing, augmentation and visualisation using OpenCV.
- `Papers/` – Research papers and related literature used for this thesis.
- `myWork/` – The main working area of the thesis, containing:
  - scripts for dataset processing and experimentation with different networks and similarity settings,
  - spreadsheets (`.xlsx`) with metrics and results,
  - scripts that compute correlations between model outputs and clinicians' scores on the hospital dataset,
  - images used for testing and visualisation,
  - and the code used to develop the **GUI** for interactive inspection of DISTS/LPIPS scores.

The GUI is a **complementary tool** built on top of the core models and similarity metrics, mainly to visualise and explore the behaviour of DISTS and LPIPS on dermatology images and regions. The primary focus of the thesis remains the **design, training, and analysis of the deep learning and perceptual similarity models themselves**.

## How to Use This Repository

This repository contains **research code** and multiple experimental setups (TensorFlow, PyTorch, classical ML, GUI tools, etc.) developed during an exploratory diploma thesis.  
It is **not a single plug-and-play training pipeline**, but rather a collection of scripts and notebooks that illustrate ideas, architectures, and evaluation workflows.

The code is intended primarily as a **reference and starting point** for researchers or practitioners who would like to apply similar methods to **their own dermatology (or medical) image datasets**.

A typical way to work with this repository is:

1. **Clone the repository** from GitHub:
   ```bash
   git clone https://github.com/stylianos-syrros/deep-learning-similarity-metrics-dermatology.git
   cd deep-learning-similarity-metrics-dermatology
   ```

2. **Create and activate a Python environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   ```

3. **Install the required dependencies** (if a `requirements.txt` or `environment.yml` is provided in a specific folder, use that). Example:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the datasets** (ISIC, hospital data if you have access):
   - Place them under `datasets/` or follow the instructions in the corresponding notebook.
   - Make sure dataset paths in notebooks/scripts match your local structure.

5. **Open the relevant notebooks** (e.g. under `GoogleCollab/` or other folders) in Jupyter or Google Colab and run the cells in order.

6. **Run the interactive GUI tool** (see the section below) to load a dermatology image, select two regions, and obtain DISTS and LPIPS similarity scores for them.

## Interactive GUI Tool for Region Similarity

In addition to the training and evaluation code, the project includes a **graphical user interface (GUI)** (implemented in Python with `tkinter`, `torch`, `lpips`, `DISTS_pytorch`, `Pillow`, etc.) that allows interactive exploration of perceptual similarity in dermatology images.

### Main capabilities

- The user can **load a single image** (e.g. a dermoscopic or clinical skin image).
- The user can **select or mark two regions** within that image (or optionally compare two different images).
- The tool computes and displays the corresponding **DISTS** and **LPIPS** similarity scores between the two selected regions (or images), using deep perceptual similarity models.

This GUI provides an intuitive way for clinicians and researchers to:

- Visually inspect and compare different regions of the same lesion or nearby lesions.
- Understand how perceptual similarity metrics behave on real dermatology cases.
- Demonstrate the underlying models and similarity measures in a more user-friendly setting.

### How to run the GUI (local setup)

1. Make sure you have the necessary Python dependencies installed (in particular):
   - `torch`, `torchvision`
   - `lpips`
   - `DISTS_pytorch`
   - `Pillow`
   - `numpy`, `matplotlib`
   - and the standard library `tkinter` (included with most Python distributions on desktop OS).

2. Open a terminal / PowerShell window and navigate to the folder that contains the GUI scripts (e.g. a directory with `imageSelector.py`).

3. Run the GUI application, for example:
   ```bash
   python imageSelector.py
   ```

4. In the GUI window:
   - Click **"Select Option"**.
   - Choose **"Select Two Different Images"** to compare two separate images, or
   - Choose **"Select One Image with Regions"** to load a single image and interactively mark two regions.

5. After selecting the regions or images, the GUI will compute and display:
   - A **DISTS score** for the pair of regions/images.
   - An **LPIPS score (VGG-based)** for the same pair.

Depending on your system, you may also want to enable GPU support (CUDA) so that the underlying models (used by DISTS and LPIPS) run faster.

## Results (High-Level)

The thesis contains a detailed quantitative and qualitative evaluation of the models, including:

- Classification accuracy and other metrics on ISIC and hospital datasets.
- Comparison of CNN-based models (ResNet50, VGG16) and transformer-based models (ViT).
- Analysis of perceptual similarity metrics and their relevance for dermatology image regions.

For exact metrics (tables, ROC curves, confusion matrices, etc.), please refer to the **thesis document** and any results notebooks included in this repository.

## Ethical and Practical Considerations

- The hospital dataset is **not public** and cannot be shared, due to patient privacy and ethical constraints.
- Models and code in this repository are provided **for research and educational purposes only** and **must not** be used as a standalone medical diagnostic tool.

## Contact

For questions regarding this work, collaboration, or further information about the thesis, feel free to contact me via GitHub or at stylianossyrros@gmail.com provided in the CV / accompanying application materials.
