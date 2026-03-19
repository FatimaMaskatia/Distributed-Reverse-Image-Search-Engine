import os
import time
from image_loader import load_images_parallel
from hasher import (compute_hashes_parallel,
                    validate_hash_consistency,
                    validate_hash_consistency_threaded)

IMAGE_DIR   = r"C:\Users\MKT\Desktop\image_search\images"
NUM_WORKERS = 4

if __name__ == "__main__":

    print("=" * 55)
    print("  Milestone 1 - Parallel Image Loading & Hashing")
    print("=" * 55)

    # ── STEP 1: Load and preprocess ──────────────────────────
    print("\n[STEP 1] Loading & Preprocessing Images...")
    print("  Multi-threaded loader — JPEG, PNG, BMP, WEBP")
    print("  Resize: 224x224 | Normalize: [0.0, 1.0]")
    print("-" * 55)

    start = time.time()
    pil_images, array_images = load_images_parallel(
        IMAGE_DIR, num_workers=NUM_WORKERS
    )
    load_time = time.time() - start
    print(f"  Completed in {load_time:.2f} sec ({NUM_WORKERS} threads)")

    # ── STEP 2: Compute hashes ───────────────────────────────
    print("\n[STEP 2] Computing Perceptual Hashes...")
    print("  aHash + dHash + pHash — no double disk read")
    print("-" * 55)

    start = time.time()
    hash_results = compute_hashes_parallel(
        pil_images, num_workers=NUM_WORKERS
    )
    hash_time = time.time() - start
    print(f"  Completed in {hash_time:.2f} sec ({NUM_WORKERS} threads)")

    # ── STEP 3: Dataset validation ───────────────────────────
    print("\n[STEP 3] Dataset Validation...")
    print("-" * 55)

    category_counts = {}
    for path in pil_images.keys():
        relative = os.path.relpath(path, IMAGE_DIR)
        parts    = relative.split(os.sep)
        category = parts[-2] if len(parts) >= 2 else "root"
        category_counts[category] = category_counts.get(category, 0) + 1

    for category, count in sorted(category_counts.items()):
        print(f"  {category:<25} {count} images")

    failed = len(pil_images) - len(hash_results)
    print(f"\n  Total images:     {len(pil_images)}")
    print(f"  Hashed:           {len(hash_results)}")
    print(f"  Corrupted/failed: {failed}")

    # ── STEP 4: Sample output ────────────────────────────────
    print("\n[STEP 4] Sample Hash Output (first 3 images):")
    print("-" * 55)
    for path, hashes in list(hash_results.items())[:3]:
        print(f"  Image : {os.path.basename(path)}")
        print(f"  aHash : {hashes['aHash']}")
        print(f"  dHash : {hashes['dHash']}")
        print(f"  pHash : {hashes['pHash']}")
        print()

    # ── STEP 5: Sequential consistency ──────────────────────
    print("[STEP 5] Sequential Consistency Check...")
    print("-" * 55)
    test_path = list(pil_images.keys())[0]
    test_img  = pil_images[test_path]
    seq_ok    = validate_hash_consistency(test_img, test_path, runs=5)

    # ── STEP 6: Cross-thread consistency ─────────────────────
    print("\n[STEP 6] Cross-Thread Consistency Check...")
    print("-" * 55)
    thread_ok = validate_hash_consistency_threaded(
        test_path, test_img, num_workers=4, runs=10
    )

    # ── STEP 7: Speedup benchmark ────────────────────────────
    print("\n[STEP 7] Parallel Speedup Analysis...")
    print("-" * 55)
    print(f"  Full dataset loading with 4 threads: {load_time:.2f} sec")
    print(f"  Full dataset hashing with 4 threads: {hash_time:.2f} sec")
    print(f"  Total pipeline time:                 {load_time + hash_time:.2f} sec")
    print()
    print("  Thread efficiency note:")
    print("  Loading is I/O-bound → threads give real speedup.")
    print("  Hashing is CPU-bound → Python GIL limits thread")
    print("  parallelism. Multiprocessing would give further")
    print("  CPU speedup but threads are correct for this pipeline.")

    # ── FINAL SUMMARY ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    print(f"  Total images:              {len(pil_images)}")
    print(f"  Categories in dataset:     {len(category_counts)}")
    print(f"  Successfully hashed:       {len(hash_results)}")
    print(f"  Corrupted/failed:          {failed}")
    print(f"  Load time  (4 threads):    {load_time:.2f} sec")
    print(f"  Hash time  (4 threads):    {hash_time:.2f} sec")
    print(f"  Total pipeline time:       {load_time + hash_time:.2f} sec")
    print(f"  Pipeline design:           I/O parallel (threaded loader)")
    print(f"  Sequential consistency:    {'PASSED' if seq_ok else 'FAILED'}")
    print(f"  Cross-thread consistency:  {'PASSED' if thread_ok else 'FAILED'}")
    print("=" * 55)
    print("\n  Part of Milestone 1 complete")