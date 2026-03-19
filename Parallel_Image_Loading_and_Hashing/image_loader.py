import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

IMAGE_SIZE = (224, 224)

def load_and_preprocess(image_path):
    """
    Loads a single image from disk, resizes to 224x224,
    normalizes pixel values to [0.0, 1.0], and returns
    both the PIL image and numpy array.
    No shared state — safe to call from multiple threads.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        return image_path, img_resized, img_array, None
    except Exception as e:
        return image_path, None, None, str(e)


def load_images_parallel(image_dir, num_workers=4):
    """
    Loads all images from a folder using multiple threads.
    Handles JPEG, PNG, BMP, WEBP formats.
    Traverses subfolders automatically.

    Returns:
        pil_images   -> { path: PIL image }   for hashing
        array_images -> { path: numpy array } for CNN features
    """
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_paths = []

    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, f))

    print(f"Found {len(image_paths)} images. Loading with {num_workers} threads...")

    pil_images   = {}
    array_images = {}
    errors       = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(load_and_preprocess, path): path
            for path in image_paths
        }
        for future in tqdm(as_completed(future_to_path),
                           total=len(image_paths),
                           desc="Loading images"):
            path, pil_img, img_array, error = future.result()
            if error:
                errors.append((path, error))
            else:
                pil_images[path]   = pil_img
                array_images[path] = img_array

    print(f"Successfully loaded: {len(pil_images)} images")
    if errors:
        print(f"Failed to load:     {len(errors)} images")
        for p, e in errors[:5]:
            print(f"  - {os.path.basename(p)}: {e}")

    return pil_images, array_images