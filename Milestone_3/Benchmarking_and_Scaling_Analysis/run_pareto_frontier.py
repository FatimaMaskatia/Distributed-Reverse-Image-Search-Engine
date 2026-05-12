import numpy as np
import time
import os
import matplotlib.pyplot as plt
from lsh_index import DistributedLSHIndex, LSHConfig, LSHHashTable # <-- Added LSHHashTable
from sharding import IndexSharding

DATA_DIR = '../data'
NUM_QUERIES = 50
K = 50 # We will evaluate Recall@50

print(f"\n{'='*50}\n🚀 RUNNING PARETO FRONTIER ANALYSIS\n{'='*50}")

# 1. Load the Real Validation Data
filepath = os.path.join(DATA_DIR, 'fused_features.npy')
print("Loading real fused features dataset...")
try:
    features = np.load(filepath).astype(np.float32)
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {filepath}. Please ensure it is uploaded!")
    raise

# Ensure vectors are normalized
features /= np.linalg.norm(features, axis=1, keepdims=True)
num_images = features.shape[0]

np.random.seed(42)
query_indices = np.random.choice(num_images, NUM_QUERIES, replace=False)
queries = features[query_indices].copy()

# 2. Establish "Ground Truth" using Exact Math (Brute Force)
print("\nCalculating Exact Ground Truth (100% Accuracy baseline)...")
ground_truth_sets = []
for q_vec in queries:
    # Exact cosine similarity against entire dataset
    similarities = features @ q_vec
    # Get top K indices
    top_k_exact = np.argsort(similarities)[::-1][:K]
    ground_truth_sets.append(set(top_k_exact))

# 3. Define the Grid Search Parameters
table_configs = [5, 10, 20]
hash_widths = [8, 16, 32]

results_latency = []
results_recall = []
labels = []

# --- THE FIXED PATCH FOR COLAB RAM ---
def fast_sequential_build(self, feature_vectors, consistent_hash_ring=None):
    # We must initialize the dimension and tables before inserting!
    self._dim = feature_vectors.shape[1]
    self._consistent_hash_ring = consistent_hash_ring
    
    self.tables = [
        LSHHashTable(
            num_projections=self.config.hash_width,
            vector_dimension=self._dim,
            table_id=i,
        )
        for i in range(self.config.num_tables)
    ]
    
    if consistent_hash_ring:
        self.config.num_nodes = consistent_hash_ring.num_nodes
        
    for i in range(feature_vectors.shape[0]):
        self.insert_image(i, feature_vectors[i])

DistributedLSHIndex.build_from_features = fast_sequential_build
# -----------------------------------

print("\nCommencing Grid Search...")
sharder = IndexSharding(num_nodes=4, virtual_nodes=150)

for tables in table_configs:
    for width in hash_widths:
        print(f"\nTesting: {tables} Tables | {width}-bit Hash")
        
        config = LSHConfig(num_tables=tables, hash_width=width, num_nodes=4, num_threads=1)
        index = DistributedLSHIndex(config)
        index.build_from_features(features, consistent_hash_ring=sharder.ring)
        
        latencies = []
        recalls = []
        
        for idx, q_vec in enumerate(queries):
            start_q = time.time()
            
            # Fetch candidates from LSH (no re-ranking needed for raw LSH recall)
            candidates = index.query(q_vec, k=K)
            cand_ids = set([c[0] for c in candidates] if candidates else [])
            
            latencies.append((time.time() - start_q) * 1000)
            
            # Calculate Recall: How many of the true Top-K did LSH find?
            true_set = ground_truth_sets[idx]
            intersection = cand_ids.intersection(true_set)
            recall = len(intersection) / K
            recalls.append(recall)
            
        avg_lat = np.percentile(latencies, 50) # P50 Latency
        avg_rec = np.mean(recalls) * 100 # Convert to percentage
        
        print(f"  -> Recall@{K}: {avg_rec:.2f}% | P50 Latency: {avg_lat:.2f} ms")
        
        results_latency.append(avg_lat)
        results_recall.append(avg_rec)
        labels.append(f"T={tables}, W={width}")

# 4. Generate the Pareto Frontier Plot
plt.figure(figsize=(10, 7))
plt.scatter(results_latency, results_recall, color='#1f77b4', s=100, zorder=5)

# Annotate points
for i, label in enumerate(labels):
    plt.annotate(label, (results_latency[i], results_recall[i]), 
                 xytext=(8, 5), textcoords='offset points', fontsize=9)

# Draw the Pareto Frontier line
sorted_indices = np.argsort(results_latency)
pareto_lat = []
pareto_rec = []
max_rec = -1

for i in sorted_indices:
    if results_recall[i] > max_rec:
        pareto_lat.append(results_latency[i])
        pareto_rec.append(results_recall[i])
        max_rec = results_recall[i]

plt.plot(pareto_lat, pareto_rec, color='#d62728', linestyle='--', linewidth=2, label='Pareto Frontier', zorder=4)

plt.title(f'Accuracy vs. Latency Pareto Frontier (Recall@{K})', fontsize=14, fontweight='bold')
plt.xlabel('P50 Query Latency (milliseconds) → Lower is Better', fontsize=12)
plt.ylabel(f'Recall@{K} (%) → Higher is Better', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('pareto_frontier_plot.png', dpi=300)
plt.show()

print("\n✅ Pareto Frontier plot generated and saved as 'pareto_frontier_plot.png'!")