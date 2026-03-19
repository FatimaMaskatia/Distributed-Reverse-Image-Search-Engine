import json
import os
from image_loader import load_images_parallel
from hasher import compute_hashes_parallel

IMAGE_DIR   = r"C:\Users\MKT\Desktop\image_search\images"
OUTPUT_FILE = r"C:\Users\MKT\Desktop\image_search\hash_results.json"
NUM_WORKERS = 4

if __name__ == "__main__":

    # Load images — one disk read only
    pil_images, array_images = load_images_parallel(
        IMAGE_DIR, num_workers=NUM_WORKERS
    )

    # Compute hashes using already loaded PIL images
    hash_results = compute_hashes_parallel(
        pil_images, num_workers=NUM_WORKERS
    )

    # Save with relative paths as keys so it works on any machine
    clean_results = {}
    for path, hashes in hash_results.items():
        relative_path = os.path.relpath(path, IMAGE_DIR)
        clean_results[relative_path] = hashes

    with open(OUTPUT_FILE, "w") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nSaved {len(clean_results)} image hashes to:")
    print(f"  {OUTPUT_FILE}")
    print(f"\nExpected: {len(hash_results)} | Saved: {len(clean_results)}")
    if len(clean_results) == len(hash_results):
        print("  No data lost")
    else:
        print("  WARNING: some entries missing")