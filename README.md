# Distributed Reverse Image Search Engine
This project implements a Reverse Image Search Engine that retrieves visually similar images from a large-scale dataset using parallel feature extraction, similarity matching and 
approximate nearest neighbour searchusing feature extraction.

## Dataset
A sample dataset is included in the repository containing a small number of images from each weather category for quick testing.
The full implementation was built and tested on the complete Weather Image Dataset (6,378 images) available on Kaggle: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
To use the full dataset:
1. Download it from Kaggle
2. Extract and place it at dataset/Weather_Dataset/
---

## Setup
Python 3.8+
### Install dependencies
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install scikit-image
pip install pillow
pip install tqdm

## How to Run
python Traditional_Feature_Descriptors.py
### Step 1 — Build the feature database (select option 1)
Run this first before any searches. Extracts features from all images and saves them to features_output/.
### Step 2 — Search similar images (select option 2)
enter the query image path
eg: dataset/Weather_Dataset/fogsmog/4075.jpg
It will display the top 5 similar images.

---
