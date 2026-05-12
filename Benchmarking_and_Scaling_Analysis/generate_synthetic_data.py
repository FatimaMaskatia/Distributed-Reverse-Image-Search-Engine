import numpy as np
import os

# Set up the data directory
DATA_DIR = DATA_DIR = '../data'
os.makedirs(DATA_DIR, exist_ok=True)
DIM = 1184
CHUNK_SIZE = 100_000

def generate_and_save(filename: str, num_vectors: int):
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Generating {num_vectors:,} synthetic vectors into '{filepath}'...")

    # Memory-mapped array to prevent RAM crashes
    fp = np.lib.format.open_memmap(filepath, mode='w+', dtype=np.float32, shape=(num_vectors, DIM))

    for i in range(0, num_vectors, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, num_vectors)
        batch = np.random.randn(end - i, DIM).astype(np.float32)

        # L2 Normalize (CRITICAL for Arham's cosine similarity)
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        fp[i:end] = batch
        print(f"  -> Progress: {end:,} / {num_vectors:,} vectors written")

    del fp # Flush memory map safely
    file_size_gb = os.path.getsize(filepath) / (1024**3)
    print(f"✅ Successfully saved {filename} ({file_size_gb:.2f} GB)\n")

print("=== M3 Synthetic Data Generator ===\n")
# Generate 100K Dataset for Weak Scaling baseline
generate_and_save("synthetic_100k.npy", 100_000)
# Generate 1M Dataset for Strong Scaling tests
generate_and_save("synthetic_1m.npy", 1_000_000)
print("All synthetic data generated. Ready for benchmarking!")