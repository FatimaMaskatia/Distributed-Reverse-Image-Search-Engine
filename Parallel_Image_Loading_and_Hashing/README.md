\# Milestone 1 - Parallel Image Loading \& Hashing



\## What this module does

1\. Loads and preprocesses all 6378 images from the Weather Dataset

2\. Computes perceptual hashes (aHash, dHash, pHash) for every image



\---



\## Dataset Info



| Category  | Images |

|-----------|--------|

| dew       | 214    |

| fogsmog   | 851    |

| frost     | 475    |

| glaze     | 639    |

| hail      | 591    |

| lightning | 377    |

| rain      | 526    |

| rainbow   | 232    |

| rime      | 1160   |

| sandstorm | 692    |

| snow      | 621    |

| \*\*Total\*\* | \*\*6378\*\* |



\---



\## Files Included



| File | What it does |

|------|-------------|

| `image\_loader.py` | Loads images using 4 threads, resizes to 224x224, normalizes to \[0.0, 1.0] |

| `hasher.py` | Computes aHash, dHash, pHash in parallel using already-loaded images |

| `main.py` | Runs the full pipeline and prints results |

| `save\_results.py` | Saves all hashes to hash\_results.json |

| `hash\_results.json` | Ready-to-use output file with all 6378 image hashes |



\---



\## Requirements

```

pip install pillow imagehash numpy opencv-python tqdm

```



\---



\## Option 1 — Just use the JSON file (easiest)



No need to run anything. Just load hash\_results.json directly:

```python

import json



with open("hash\_results.json", "r") as f:

&#x20;   hash\_results = json.load(f)



\# hash\_results looks like this:

\# {

\#   "dataset\\\\Weather\_Dataset\\\\dew\\\\2692.jpg": {

\#       "aHash": "3c3c3e7e5e9e0c00",

\#       "dHash": "e9f8e4d4b43458c9",

\#       "pHash": "96856bfa61676e08"

\#   },

\#   ...

\# }



for image\_path, hashes in hash\_results.items():

&#x20;   print(image\_path)

&#x20;   print(hashes\["pHash"])  # perceptual hash — most important

```



\---



\## Option 2 — Use the modules directly (if you need image arrays too)

```python

from image\_loader import load\_images\_parallel

from hasher import compute\_hashes\_parallel



IMAGE\_DIR = "path/to/images"  # change this to your local path



\# Load images

\# pil\_images   = { path: PIL image }   → for hashing

\# array\_images = { path: numpy array } → for CNN / feature fusion

pil\_images, array\_images = load\_images\_parallel(IMAGE\_DIR, num\_workers=4)



\# Compute hashes

hash\_results = compute\_hashes\_parallel(pil\_images, num\_workers=4)



\# You now have:

\# array\_images → normalized float32 numpy arrays (224x224x3)

\# hash\_results → aHash, dHash, pHash for every image

```



\---



\## What the Hashes Mean



| Hash  | Full Name       | How it works                        | Best for               |

|-------|-----------------|-------------------------------------|------------------------|

| aHash | Average Hash    | Compares pixels to average brightness | Fast, rough similarity |

| dHash | Difference Hash | Compares adjacent pixel differences | Detecting edits        |

| pHash | Perceptual Hash | Frequency analysis (like JPEG)      | Most robust similarity |



\### Comparing two images by hash distance:

```python

import imagehash



hash1 = imagehash.hex\_to\_hash(hash\_results\["image1.jpg"]\["pHash"])

hash2 = imagehash.hex\_to\_hash(hash\_results\["image2.jpg"]\["pHash"])



distance = hash1 - hash2  # lower = more similar, 0 = identical



\# Rule of thumb:

\# distance = 0      → identical images

\# distance < 10     → very similar

\# distance 10-20    → somewhat similar

\# distance > 20     → different images

```



\---



\## Image Array Format



The array\_images dictionary contains numpy arrays with:

\- Shape: (224, 224, 3)

\- Type: float32

\- Values: 0.0 to 1.0

\- Channel order: RGB



This is the standard format for CNN models and feature fusion.



\---



\## Performance



| Step                          | Time    |

|-------------------------------|---------|

| Loading 6378 images (4 threads) | \~16 sec |

| Hashing 6378 images (4 threads) | \~9 sec  |

| Total pipeline                | \~25 sec |



\- 0 corrupted images found in dataset

\- Hash consistency validated sequentially and across threads — both passed



\---



\## Integration Checklist



\- \[ ] Place hash\_results.json in your project folder

\- \[ ] Update IMAGE\_DIR path to match your machine

\- \[ ] Use array\_images for CNN feature extraction

\- \[ ] Use hash\_results for feature fusion (pHash recommended)

\- \[ ] Hash distance < 10 means visually similar images

