import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure Python finds your team's code
sys.path.insert(0, "/content/drive/MyDrive/PDC_Project_M2/src")

from lsh_index import DistributedLSHIndex, LSHConfig
from sharding import IndexSharding
from query_processor import DistributedQueryProcessor
from utils import load_features_and_metadata

BASE_PATH = "/content/drive/MyDrive/PDC_Project_M1/"
M2_SRC_PATH = "/content/drive/MyDrive/PDC_Project_M2/src/"

def evaluate_lsh_tradeoffs():
    print("Loading database for benchmarking...")
    features, image_paths = load_features_and_metadata(
        Path(f"{BASE_PATH}fused_features.npy"), 
        Path(f"{BASE_PATH}metadata.csv")
    )
    
    # Extract categories (e.g., 'dew') for accuracy checking
    categories = [path.replace('\\', '/').split('/')[-2] for path in image_paths]
    
    # Pick 50 random images to act as our benchmark queries
    np.random.seed(42)
    query_indices = np.random.choice(len(features), 50, replace=False)
    
    table_configs = [2, 5, 10, 20]
    results = []

    print("\nStarting Benchmark Suite...")
    for tables in table_configs:
        print(f"\n--- Testing Configuration: {tables} Hash Tables ---")
        
        # 1. Setup the distributed system
        config = LSHConfig(num_tables=tables, hash_width=32, num_nodes=4, virtual_nodes=150)
        sharder = IndexSharding(num_nodes=config.num_nodes, virtual_nodes=config.virtual_nodes)
        sharder.distribute(features)
        
        index = DistributedLSHIndex(config)
        index.build_from_features(features, consistent_hash_ring=sharder.ring)
        
        processor = DistributedQueryProcessor(lsh_index=index, feature_matrix=features, sharder=sharder)
        
        # 2. Run the queries
        total_precision = 0
        latencies = []
        candidates_checked = []
        
        for q_idx in query_indices:
            q_vector = features[q_idx]
            q_category = categories[q_idx]
            
            # Query the system
            top_k_results, stats = processor.query(q_vector, k=5, collect_stats=True)
            
            latencies.append(stats.total_ms)
            candidates_checked.append(stats.num_candidates)
            
            # Calculate precision (Ignore the exact duplicate image, check the other 4)
            valid_results = [r for r in top_k_results if r.image_id != q_idx][:4]
            if not valid_results:
                continue
                
            correct_matches = sum(1 for r in valid_results if categories[r.image_id] == q_category)
            total_precision += (correct_matches / 4.0) # Out of 4 possible matches
            
        # 3. Store the averages
        avg_precision = (total_precision / 50) * 100
        avg_latency = np.mean(latencies)
        avg_candidates = np.mean(candidates_checked)
        
        print(f"Results -> Precision: {avg_precision:.1f}%, Latency: {avg_latency:.2f}ms, Candidates Searched: {avg_candidates:.0f}")
        
        results.append({
            'Tables': tables,
            'Precision': avg_precision,
            'Latency': avg_latency,
            'Candidates': avg_candidates
        })

    # ==========================================
    # PLOTTING THE TRADE-OFF GRAPH
    # ==========================================
    df = pd.DataFrame(results)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Plot Precision (Bar Chart)
    color = 'tab:blue'
    ax1.set_xlabel('Number of LSH Hash Tables')
    ax1.set_ylabel('Accuracy (Precision@5 %)', color=color, fontweight='bold')
    bars = ax1.bar(df['Tables'].astype(str), df['Precision'], color=color, alpha=0.7, width=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)
    
    # Plot Latency (Line Chart on secondary Y-axis)
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Query Latency (ms)', color=color, fontweight='bold')  
    line = ax2.plot(df['Tables'].astype(str), df['Latency'], color=color, marker='o', linewidth=3, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(df['Latency']) * 1.2)
    
    # Add titles and labels
    plt.title('Accuracy vs. Latency Trade-off in Distributed LSH', fontsize=14, fontweight='bold')
    
    # Annotate candidate counts on the bars
    for i, bar in enumerate(bars):
        cand_text = f"{int(df['Candidates'].iloc[i])}\ncandidates"
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 15, cand_text, 
                 ha='center', color='white', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plot_path = f"{BASE_PATH}lsh_tradeoff_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n✅ Saved evaluation graph to {plot_path}")
    plt.show()

# Run the benchmark
from pathlib import Path
evaluate_lsh_tradeoffs()