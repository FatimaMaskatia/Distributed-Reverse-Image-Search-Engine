import numpy as np
from sharding import IndexSharding
from monitor import PartitionMonitor

data = np.load(r'C:\Users\MKT\Downloads\milestone1_Zain\fused_features.npy', allow_pickle=True)

print("=== TEST 1: Balanced distribution (150 virtual nodes) ===")
sharder1 = IndexSharding(num_nodes=4, virtual_nodes=150)
sharder1.distribute(data)
monitor1 = PartitionMonitor(sharder1, skew_threshold=0.15)
monitor1.check_skew()

print("\n=== TEST 2: Unbalanced distribution (5 virtual nodes) ===")
sharder2 = IndexSharding(num_nodes=4, virtual_nodes=5)
sharder2.distribute(data)
monitor2 = PartitionMonitor(sharder2, skew_threshold=0.15)
monitor2.check_skew()