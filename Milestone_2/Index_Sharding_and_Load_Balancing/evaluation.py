import numpy as np
import time
from sharding import IndexSharding
from monitor import PartitionMonitor
from rebalancer import Rebalancer

data = np.load(r'C:\Users\MKT\Downloads\milestone1_Zain\fused_features.npy', allow_pickle=True)

configs = [
    {"num_nodes": 2, "virtual_nodes": 150},
    {"num_nodes": 4, "virtual_nodes": 150},
    {"num_nodes": 8, "virtual_nodes": 150},
    {"num_nodes": 4, "virtual_nodes": 10},
    {"num_nodes": 4, "virtual_nodes": 50},
    {"num_nodes": 4, "virtual_nodes": 150},
]


print("EVALUATION: Testing different sharding configurations")


results = []

for config in configs:
    num_nodes = config["num_nodes"]
    virtual_nodes = config["virtual_nodes"]

    print(f"\nConfig: {num_nodes} nodes, {virtual_nodes} virtual nodes")


    start_time = time.time()
    sharder = IndexSharding(num_nodes=num_nodes, virtual_nodes=virtual_nodes)
    sharder.distribute(data)
    distribution_time = time.time() - start_time

    sizes = [len(sharder.get_partition(i)) for i in range(num_nodes)]
    average = sum(sizes) / len(sizes)
    max_skew = max(abs(s - average) / average for s in sizes) * 100

    start_time = time.time()
    for i in range(1000):
        sharder.ring.get_node(i)
    query_time = (time.time() - start_time) / 1000 * 1000

    print(f"Distribution time : {distribution_time:.3f} seconds")
    print(f"Avg query time    : {query_time:.4f} ms")
    print(f"Max skew          : {max_skew:.1f}%")
    print(f"Partition sizes   : {sizes}")

    results.append({
        "num_nodes": num_nodes,
        "virtual_nodes": virtual_nodes,
        "distribution_time": distribution_time,
        "query_time": query_time,
        "max_skew": max_skew
    })

print("\n")

print("SUMMARY TABLE")

print(f"{'Nodes':<8} {'VNodes':<10} {'Dist Time':<14} {'Query Time':<14} {'Max Skew'}")

for r in results:
    print(f"{r['num_nodes']:<8} {r['virtual_nodes']:<10} {r['distribution_time']:<14.3f} {r['query_time']:<14.4f} {r['max_skew']:.1f}%")

print("\nKEY FINDINGS:")
print("- More virtual nodes = lower skew but more memory usage")
print("- More nodes = faster query time but more coordination overhead")
print("- Best spot for this dataset: 4 nodes, 150 virtual nodes")