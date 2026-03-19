import os
import imagehash
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def compute_hashes_from_pil(image_path, pil_img):
    """
    Computes 3 perceptual hashes from an already loaded PIL image.
    No disk reading — image was loaded once in image_loader.py.
    No shared state — safe to call from multiple threads simultaneously.

    aHash: average hash  — based on average pixel brightness
    dHash: difference hash — based on pixel-to-pixel differences
    pHash: perceptual hash — based on frequency patterns, most robust
    """
    try:
        a_hash = imagehash.average_hash(pil_img)
        d_hash = imagehash.dhash(pil_img)
        p_hash = imagehash.phash(pil_img)

        return image_path, {
            "aHash": str(a_hash),
            "dHash": str(d_hash),
            "pHash": str(p_hash)
        }, None

    except Exception as e:
        return image_path, None, str(e)


def compute_hashes_parallel(pil_images, num_workers=4):
    """
    Computes perceptual hashes for all images in parallel.
    Accepts already-loaded PIL images so no double disk reading occurs.

    Returns:
        hash_results -> { path: { aHash, dHash, pHash } }
    """
    print(f"Computing hashes for {len(pil_images)} images "
          f"with {num_workers} threads...")

    hash_results = {}
    errors       = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(compute_hashes_from_pil, path, pil_img): path
            for path, pil_img in pil_images.items()
        }
        for future in tqdm(as_completed(future_to_path),
                           total=len(pil_images),
                           desc="Hashing images"):
            path, hashes, error = future.result()
            if error:
                errors.append((path, error))
            else:
                hash_results[path] = hashes

    print(f"Successfully hashed: {len(hash_results)} images")
    if errors:
        print(f"Failed:             {len(errors)} images")

    return hash_results


def validate_hash_consistency(pil_img, image_path, runs=5):
    """
    Hashes the same image multiple times sequentially.
    Verifies that the output is always identical.
    Proves deterministic output on a single thread.
    """
    print(f"\nSequential consistency check: {os.path.basename(image_path)}")
    hashes_seen = set()

    for i in range(runs):
        _, hashes, _ = compute_hashes_from_pil(image_path, pil_img)
        hashes_seen.add(str(hashes))
        print(f"  Run {i+1}: pHash = {hashes['pHash']}")

    if len(hashes_seen) == 1:
        print(f"  Result: ALL {runs} runs identical")
        return True
    else:
        print(f"  Result: INCONSISTENT")
        return False


def validate_hash_consistency_threaded(image_path, pil_img,
                                        num_workers=4, runs=10):
    """
    Hashes the same image from multiple threads simultaneously.
    Verifies that all threads produce identical output.
    Directly satisfies spec requirement:
    'Validate hash consistency across threads'
    """
    print(f"\nCross-thread consistency check: {os.path.basename(image_path)}")
    print(f"  Submitting {runs} threads simultaneously...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(compute_hashes_from_pil, image_path, pil_img)
            for _ in range(runs)
        ]
        results = [f.result()[1] for f in futures]

    unique_results = set(str(r) for r in results)

    if len(unique_results) == 1:
        print(f"  All {runs} threads produced identical hashes")
        return True
    else:
        print(f"  INCONSISTENT results across threads")
        return False