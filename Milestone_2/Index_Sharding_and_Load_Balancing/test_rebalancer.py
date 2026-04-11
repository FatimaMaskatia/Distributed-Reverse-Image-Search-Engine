import numpy as np
from sharding import IndexSharding
from monitor import PartitionMonitor
from rebalancer import Rebalancer

data = np.load(r'C:\Users\MKT\Downloads\milestone1_Zain\fused_features.npy', allow_pickle=True)

print("=== Creating unbalanced distribution (5 virtual nodes) ===")
sharder = IndexSharding(num_nodes=4, virtual_nodes=5)
sharder.distribute(data)

monitor = PartitionMonitor(sharder, skew_threshold=0.15)
print("\nBEFORE rebalancing:")
needs_rebalance = monitor.check_skew()

if needs_rebalance:
    rebalancer = Rebalancer(sharder, skew_threshold=0.15)
    rebalancer.rebalance()
    print("\nAFTER rebalancing:")
    monitor.check_skew()

print("\n=== TEST: Ring sync after rebalancing ===")
print("Checking images that were ACTUALLY moved during rebalancing:")

all_correct = True
for image_id in rebalancer.moved_images[:5]:
    ring_says = sharder.get_node_for_image(image_id)
    actually_in = None
    for node_id in range(4):
        if image_id in sharder.partitions[node_id]:
            actually_in = node_id
            break
    match = "CORRECT" if ring_says == actually_in else "WRONG"
    if ring_says != actually_in:
        all_correct = False
    print(f"Image {image_id}: ring says Node {ring_says}, actually on Node {actually_in} → {match}")

print(f"\nOverall: {'All correct, ring is in sync!' if all_correct else 'Some images are out of sync!'}")