import matplotlib.pyplot as plt
import numpy as np

# --- 1. STRONG SCALING PLOT ---
nodes_strong = ['1 Node', '2 Nodes', '4 Nodes']
p50_strong = [4.11, 16.32, 16.07]
p95_strong = [5.53, 22.13, 20.42]
p99_strong = [11.15, 26.97, 29.00]

plt.figure(figsize=(10, 6))
plt.plot(nodes_strong, p50_strong, marker='o', linewidth=2, label='P50 Latency', color='#2ca02c')
plt.plot(nodes_strong, p95_strong, marker='s', linewidth=2, label='P95 Latency', color='#ff7f0e')
plt.plot(nodes_strong, p99_strong, marker='^', linewidth=2, label='P99 Latency', color='#d62728')

plt.title('Strong Scaling Analysis (Fixed Load: 250,000 Images)', fontsize=14, fontweight='bold')
plt.xlabel('Cluster Size', fontsize=12)
plt.ylabel('Latency (milliseconds)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('strong_scaling_plot.png', dpi=300)
plt.show()

# --- 2. WEAK SCALING PLOT ---
nodes_weak = ['1 Node\n(62.5k Images)', '4 Nodes\n(250k Images)']
p50_weak = [1.36, 12.46]
p95_weak = [3.17, 16.18]
p99_weak = [4.49, 19.38]

plt.figure(figsize=(8, 6))
plt.plot(nodes_weak, p50_weak, marker='o', linewidth=2, label='P50 Latency', color='#2ca02c')
plt.plot(nodes_weak, p95_weak, marker='s', linewidth=2, label='P95 Latency', color='#ff7f0e')
plt.plot(nodes_weak, p99_weak, marker='^', linewidth=2, label='P99 Latency', color='#d62728')

plt.title('Weak Scaling Analysis (Proportional Load)', fontsize=14, fontweight='bold')
plt.xlabel('Cluster Size & Workload', fontsize=12)
plt.ylabel('Latency (milliseconds)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('weak_scaling_plot.png', dpi=300)
plt.show()

print("✅ Scaling plots generated and saved as PNG files!")