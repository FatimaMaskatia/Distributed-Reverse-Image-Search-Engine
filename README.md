# Distributed Reverse Image Search Engine
This project implements a Reverse Image Search Engine that retrieves visually similar images from a large-scale dataset using parallel feature extraction, similarity matching and 
approximate nearest neighbour searchusing feature extraction.

## Dataset
A sample dataset is included in the repository containing a small number of images from each weather category for quick testing.
The full implementation was built and tested on the complete Weather Image Dataset (6,378 images) available on Kaggle: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset

### System Requirements:
Python (3.8 or higher)
RAM (8 GB recommended)
GPU (Optional - CUDA for 3rd module)
OS (Windows, Linux or macOS)

## Module 1 — Image Preprocessing & Hashing
Loads all images using a multi-threaded pipeline and computes three perceptual hashes per image.
### Install
```bash
pip install pillow imagehash numpy opencv-python tqdm
```

### Run
```bash
python main.py
```
Output saved to: `hash_results.json`

---

## Module 2 — Deep Learning Feature Extraction
Extracts 2048-dimensional CNN feature vectors using a pre-trained ResNet-50 model.
### Install
pip install torch torchvision pillow numpy

Or

use the requirements file:
pip install -r requirements.txt

### Run
python src/run_extraction.py \
  --image-dir "dataset/Weather_Dataset" \
  --output-dir "outputs" \
  --batch-size 32 \
  --num-workers 2 \
  --model-name resnet50

### Output
outputs/features.npy — float32 matrix of shape [N, 2048] L2-normalized
outputs/metadata.csv — maps each row index to its image path

Notes:
GPU (CUDA) is used automatically if available
Falls back to CPU if no GPU is present
Supported formats: JPG, JPEG, PNG, BMP, WEBP

## Module 3 — Traditional Feature Descriptors
Extracts SIFT, ORB, and color histogram features in parallel using a thread pool.

### Install
pip install opencv-contrib-python numpy tqdm

- Must use opencv-contrib-python not opencv-python — SIFT requires the contrib package.

### Run
python Traditional_Features_Descriptors.py

Select option:
1 -> Build feature database (run this first)
2 -> Search similar images by entering a query image path

### Output
features_output/features.npy - float32 matrix of shape [N, 1184]
features_output/image_paths.npy - corresponding image paths

## Module 4 — Feature Fusion & Baseline Evaluation
Aligns outputs from all three modules, fuses feature vectors, and evaluates retrieval accuracy.

Requirements
Google Colab (recommended) or Jupyter Notebook
All output files from Modules 1, 2, and 3

### Setup
Upload these files to Google Drive under PDC_Project_M1/:
Files:
Source hash_results.json (Module 1 output)
features.npy + image_paths.npy (Module 2 output)
features.npy + metadata.csv (Module 3 output)

### Run
Open notebook in Google Colab
Mount Google Drive when prompted
Run all cells in order

### Output
fused_features.npy: final fused matrix of shape [N, 1184]
master_alignment_map.csv: maps each image to its index in all three feature arrays

### If don't want to run on google colab, remove the Google Drive parts and use local paths instead.
### Remove these lines:
from google.colab import drive
drive.mount('/content/drive')
BASE_PATH = "/content/drive/MyDrive/PDC_Project_M1/"
### Replace with:
BASE_PATH = "./"  # or where your files are

### Install the required libraries:
pip install numpy pandas scikit-learn
