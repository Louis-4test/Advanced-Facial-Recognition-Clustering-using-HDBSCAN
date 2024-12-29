# Image Clustering Using HDBSCAN and PCA

## Overview
This project implements an image clustering technique using the HDBSCAN algorithm and Principal Component Analysis (PCA) for dimensionality reduction. The goal is to group similar images based on visual features, manage noise data, and assign new images to identified clusters.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Real-Life Applications](#real-life-applications)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features
- Load and preprocess images from a specified directory.
- Add noise to images to assess model robustness.
- Apply PCA for dimensionality reduction.
- Use HDBSCAN for clustering based on various distance metrics.
- Classify new images and analyze generalization capability.
- Identify and visualize representative images for each cluster.

## Requirements
- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `hdbscan`
  - `opencv-python` (for image processing)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/image-clustering.git
   cd image-clustering
