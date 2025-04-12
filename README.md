# CellularVision

**Analysis and Segmentation of Histopathological Tissues**

CellularVision is a comprehensive repository for performing nuclear segmentation and analysis on the PanNuke dataset. The project implements several deep learning segmentation models alongside traditional post-processing techniques to refine segmentation masks. The main segmentation models include:  

1. Baseline CNN  
2. U-Net  
3. SegNet  
4. PSPNet  
5. Mask R-CNN  

After generating segmentation predictions from these algorithms, the repository applies classical post-processing methods such as Mean Shift, Region Growing, and Spectral Clustering to enhance the final output quality.

---

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Models and Post-Processing](#models-and-post-processing)
- [Installation](#installation)
- [Usage](#usage)

## Features

- **Multiple Deep Learning Models:**  
  Implementations of five segmentation models:
  - **Baseline CNN:** A simple convolutional network used as the initial benchmark.
  - **U-Net:** A popular encoder-decoder architecture well-known in biomedical segmentation.
  - **SegNet:** An efficient segmentation architecture featuring an encoder–decoder structure.
  - **PSPNet (Pyramid Scene Parsing Network):** Uses pyramid pooling to aggregate global context.
  - **Mask R-CNN:** A two-stage detector that generates instance segmentation masks alongside bounding boxes.
  
- **Post-Processing Techniques:**  
  The raw segmentation outputs are refined using classical algorithms:
  - **Mean Shift:** To smooth segment boundaries.
  - **Region Growing:** To correct for missing or over-segmented regions.
  - **Spectral Clustering:** To separate clustered nuclei and improve delineation.

- **Visualization and Analysis Tools:**  
  Script(s) for visualizing predictions and comparisons over test images.

- **Modular Structure:**  
  Easily replace or extend segmentation models and post-processing methods.

---

## Repository Structure

```
CellularVision/
├── cellularvision/        # Main source code for segmentation pipeline
├── figures/               # Figures for documentation and sample results
├── model_weights/         # Pre-trained model weights (if available)
├── training_scripts/      # Scripts for model training and evaluation
├── app.py                 # Application script for running inference/visualization
├── visualization.py       # Script for post-processing and visualizing segmentation outputs
├── requirements.txt       # Python dependencies
├── test_image1.png        # Example input image 1
├── test_image2.png        # Example input image 2
├── .gitignore             # Files to be ignored by Git
└── README.md              # This file
```

---

## Models and Post-Processing

### Segmentation Models

1. **Baseline CNN:**  
   - *Description:* A simple convolutional model to serve as the baseline.  
   - *Usage:* Provides a reference point to compare with more complex models.

2. **U-Net:**  
   - *Description:* An encoder-decoder network that leverages skip connections to capture both low-level details and high-level context.  
   - *Usage:* Particularly effective when training data is limited and precise localization is required.

3. **SegNet:**  
   - *Description:* An efficient architecture featuring an encoder that retains pooling indices and a corresponding decoder for upsampling.
   - *Usage:* Balances model complexity with good segmentation performance in resource-constrained scenarios.

4. **PSPNet:**  
   - *Description:* Incorporates pyramid pooling to aggregate contextual information at multiple scales.  
   - *Usage:* Particularly useful for images with large variations in scale.

5. **Mask R-CNN:**  
   - *Description:* A two-stage network that combines object detection with instance segmentation to produce high-quality segmentation masks.  
   - *Usage:* Well-suited for separating overlapping or clustered nuclei.

### Post-Processing Methods

To enhance the segmentation quality, the repository applies three traditional methods:

- **Mean Shift:**  
  A non-parametric technique that clusters pixel values to smooth the mask boundaries and reduce noise.

- **Region Growing:**  
  Initiates from seed points and aggregates neighboring pixels with similar properties, effectively refining object boundaries.

- **Spectral Clustering:**  
  Utilizes graph-based clustering to separate adjacent objects and handle overlapping regions by analyzing feature similarities.

---

## Installation

### Prerequisites

- **Python 3.7+**  
- Recommended: A system with a CUDA-enabled GPU (for training and high-performance inference)

### Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Irwindeep/CellularVision.git
   cd CellularVision
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Download or place pre-trained model weights**  
   If available, move the pre-trained weights into the `model_weights/` folder.

---

## Usage

### Training

To train the segmentation models on the PanNuke dataset or your custom data, navigate to the `training_scripts/` directory and run the appropriate training script. For example:

```bash
python training_scripts/train_unet.py
```
