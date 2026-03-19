# Milestone 1 — Parallel Image Loading & Hashing

> Loads and preprocesses all 6,378 images from the Weather Dataset and computes perceptual hashes (aHash, dHash, pHash) for every image.

---

## Dataset Overview

| Category   | Images |
|------------|--------|
| dew        | 214    |
| fogsmog    | 851    |
| frost      | 475    |
| glaze      | 639    |
| hail       | 591    |
| lightning  | 377    |
| rain       | 526    |
| rainbow    | 232    |
| rime       | 1160   |
| sandstorm  | 692    |
| snow       | 621    |
| **Total**  | **6378** |

---

## Files Included

| File | Description |
|------|-------------|
| `image_loader.py` | Loads images using 4 threads, resizes to 224×224, normalizes to [0.0, 1.0] |
| `hasher.py` | Computes aHash, dHash, pHash in parallel using already-loaded images |
| `main.py` | Runs the full pipeline and prints results |
| `save_results.py` | Saves all hashes to `hash_results.json` |

---

## Requirements

```bash
pip install pillow imagehash numpy opencv-python tqdm
```

---

## Usage

Use the modules directly to load images and compute hashes:

Use this if you also need access to the raw image arrays (e.g. for CNN input):

```python
from image_loader import load_images_parallel
from hasher import compute_hashes_parallel

IMAGE_DIR = "path/to/images"  # update to your local path

# Load images
# pil_images   = { path: PIL image }    → for hashing
# array_images = { path: numpy array }  → for CNN / feature fusion
pil_images, array_images = load_images_parallel(IMAGE_DIR, num_workers=4)

# Compute hashes
hash_results = compute_hashes_parallel(pil_images, num_workers=4)

# You now have:
# array_images → normalized float32 numpy arrays (224×224×3)
# hash_results → aHash, dHash, pHash for every image
```

---

## Hash Types

| Hash  | Full Name       | How it works                            | Best for               |
|-------|-----------------|----------------------------------------|------------------------|
| aHash | Average Hash    | Compares pixels to average brightness  | Fast, rough similarity |
| dHash | Difference Hash | Compares adjacent pixel differences    | Detecting edits        |
| pHash | Perceptual Hash | Frequency analysis (like JPEG)         | Most robust similarity |

### Comparing Two Images by Hash Distance

```python
import imagehash

hash1 = imagehash.hex_to_hash(hash_results["image1.jpg"]["pHash"])
hash2 = imagehash.hex_to_hash(hash_results["image2.jpg"]["pHash"])

distance = hash1 - hash2  # lower = more similar, 0 = identical

# Rule of thumb:
# distance = 0       → identical images
# distance < 10      → very similar
# distance 10–20     → somewhat similar
# distance > 20      → different images
```

---

## Image Array Format

The `array_images` dictionary contains numpy arrays with:

- **Shape:** `(224, 224, 3)`
- **Type:** `float32`
- **Values:** `0.0` to `1.0`
- **Channel order:** RGB

This is the standard format for CNN models and feature fusion.

---

## Performance

| Step                             | Time    |
|----------------------------------|---------|
| Loading 6,378 images (4 threads) | ~16 sec |
| Hashing 6,378 images (4 threads) | ~9 sec  |
| **Total pipeline**               | **~25 sec** |

- 0 corrupted images found in the dataset
- Hash consistency validated sequentially and across threads — both passed

---

## Integration Checklist

- [ ] Place `hash_results.json` in your project folder
- [ ] Update `IMAGE_DIR` path to match your machine
- [ ] Use `array_images` for CNN feature extraction
- [ ] Use `hash_results` for feature fusion (`pHash` recommended)
- [ ] Hash distance `< 10` means visually similar images
