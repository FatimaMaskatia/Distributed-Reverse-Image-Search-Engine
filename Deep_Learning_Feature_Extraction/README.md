# Arham Milestone 1 — Deep Learning Feature Extraction (GPU)

This module implements Arham's Milestone 1 deliverable:
- CNN feature extraction using pre-trained `ResNet`.
- Batch-based GPU inference pipeline.
- Configurable memory/performance knobs (`batch_size`, `num_workers`).
- L2-normalized vectors for downstream fusion and indexing.
- Export contract: `features.npy` and `metadata.csv`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/run_extraction.py --image-dir "path/to/images" --output-dir "outputs" --batch-size 32 --num-workers 2 --model-name resnet50
```

## Pipeline Flow

```text
Input Image Folder
	|
	v
run_extraction.py (CLI entry)
	|
	v
deep_features.py
  - scan + load images
  - preprocess (resize/normalize)
  - batched CNN inference (GPU/CPU)
  - L2 normalize feature vectors
	|
	v
save_outputs(...)
  -> outputs/features.npy
  -> outputs/metadata.csv
```

### File Responsibilities
- `src/run_extraction.py`: Parses CLI arguments, builds config, runs extraction, prints summary.
- `src/deep_features.py`: Contains dataset loading, model inference, normalization, and output-saving logic.

## Output
- `features.npy`: float32 matrix, shape `[N, D]`, row-wise L2 normalized.
- `metadata.csv`: image path mapping (`index,image_path`) aligned with `features.npy` rows.

## Notes
- Uses GPU automatically when CUDA is available.
- Falls back to CPU otherwise.
- Supported input formats: JPG, JPEG, PNG, BMP, WEBP.
