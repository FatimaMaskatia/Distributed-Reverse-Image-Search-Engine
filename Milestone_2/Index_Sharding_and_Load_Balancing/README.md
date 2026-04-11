# Milestone 2 - Index Sharding & Load Balancing

---

## What this does
This module takes all the image feature vectors from Milestone 1 and splits
them evenly across multiple worker nodes. It also watches if any node gets
too much work and automatically fixes it.

The 7128 images from Milestone 1 each have 1184 features. This module
distributes those across 4 worker nodes so that instead of one machine
searching all 7128 images, 4 workers search in parallel which is much faster.

---

## Files included
- consistent_hashing.py — decides which image goes to which node
- sharding.py — splits the images across nodes
- monitor.py — detects if any node is overloaded
- rebalancer.py — fixes overloaded nodes automatically
- evaluation.py — tests different configurations and documents results

---

## Requirements
Make sure you have numpy installed:

    pip install numpy

---

## Before running anything - update the file path
Every file that loads the feature vectors has this line somewhere:

    np.load('fused_features.npy', allow_pickle=True)

Change the path inside np.load() to wherever your fused_features.npy file is actually saved on your computer.

---

## How to run the test files

    python test_hashing.py      — tests that images are assigned to nodes correctly
    python test_sharding.py     — tests that images are split across nodes
    python test_monitor.py      — tests that overloaded nodes are detected
    python test_rebalancer.py   — tests that overloaded nodes are fixed automatically
    python evaluation.py        — runs all configurations and shows a results table

---

## For LSH Index Construction
The sharder gives you one partition per node. Each partition is a dictionary of image_id to feature_vector. You build your LSH index separately on each partition.

Update the path to fused_features.npy before running:

    from sharding import IndexSharding
    import numpy as np

    data = np.load(r'YOUR PATH TO fused_features.npy HERE', allow_pickle=True)

    sharder = IndexSharding(num_nodes=4)
    sharder.distribute(data)

    # get images assigned to a specific node
    partition = sharder.get_partition(node_id=0)

    # loop through all 4 nodes and build your LSH index per node
    for node_id in range(4):
        partition = sharder.get_partition(node_id)
        image_ids = list(partition.keys())
        feature_vectors = list(partition.values())
        # feed image_ids and feature_vectors into your LSH index here

    # check which node a specific image belongs to
    node = sharder.get_node_for_image(image_id=100)

    # get list of all node IDs
    nodes = sharder.ring.get_all_nodes()
    # returns [0, 1, 2, 3]

---

## For Caching & Node Infrastructure
The sharder object holds all partition data and node assignments. Use it to check node status, monitor load, and trigger rebalancing when needed.

    from sharding import IndexSharding
    from monitor import PartitionMonitor
    from rebalancer import Rebalancer
    import numpy as np

    data = np.load(r'YOUR PATH TO fused_features.npy HERE', allow_pickle=True)

    sharder = IndexSharding(num_nodes=4, virtual_nodes=150)
    sharder.distribute(data)

    # check if any node is overloaded
    monitor = PartitionMonitor(sharder, skew_threshold=0.15)
    needs_rebalancing = monitor.check_skew()

    # if overloaded, rebalance automatically
    if needs_rebalancing:
        rebalancer = Rebalancer(sharder, skew_threshold=0.15)
        rebalancer.rebalance()

    # get partition sizes
    sizes = monitor.get_partition_sizes()
    # returns {0: 1944, 1: 1693, 2: 1743, 3: 1748}

---

## How it works internally

### Consistent Hashing
Each image is assigned to a worker node using a consistent hash ring. Each node gets multiple positions called virtual nodes. More virtual nodes means more even distribution.

With 10 virtual nodes (uneven):

    Node 0 = 22.2%,  Node 1 = 25.9%,  Node 2 = 18.1%,  Node 3 = 33.8%

With 150 virtual nodes (even):

    Node 0 = 27.3%,  Node 1 = 23.8%,  Node 2 = 24.5%,  Node 3 = 24.5%

### Partition Monitoring
The monitor tracks how many images each node holds and compares it to the average. If any node differs from the average by more than 15%, rebalancing is triggered. The 15% threshold can be tuned based on system requirements.

### Reactive Rebalancing
When skew is detected, the rebalancer moves images from overloaded nodes to underloaded nodes in multiple rounds until balanced or 10 rounds maximum.

Example with 5 virtual nodes (intentionally unbalanced):

    BEFORE: Node 0 = 51.3%,  Node 1 = 17.5%,  Node 2 = 17.0%,  Node 3 = 14.3%
    AFTER:  Node 0 = 27.2%,  Node 1 = 23.6%,  Node 2 = 25.7%,  Node 3 = 23.6%
    Fixed in just 2 rounds.

---

## Evaluation results
Tested on 7128 images with 1184 features each:

| Nodes | VNodes | Dist Time (s) | Query Time (ms) | Max Skew |
|-------|--------|---------------|-----------------|----------|
| 2     | 150    | 0.113         | 0.0143          | 5.4%     |
| 4     | 150    | 0.154         | 0.0168          | 9.1%     |
| 8     | 150    | 0.204         | 0.0468          | 21.3%    |
| 4     | 10     | 0.051         | 0.0057          | 35.2%    |
| 4     | 50     | 0.075         | 0.0070          | 27.6%    |
| 4     | 150    | 0.147         | 0.0161          | 9.1%     |

### Key findings

1. More virtual nodes = better balance but slightly slower query time
   - 10 vnodes = 35.2% skew (too uneven)
   - 50 vnodes = 27.6% skew (okay but causes frequent rebalancing)
   - 150 vnodes = 9.1% skew (best balance)

2. More nodes = more coordination overhead
   - 2 nodes = 0.014ms query time
   - 4 nodes = 0.017ms query time
   - 8 nodes = 0.047ms query time

3. Why not 4 nodes with 50 virtual nodes even though it is faster?
   50 vnodes is faster per query (0.007ms vs 0.016ms) but has 27.6% skew. That skew triggers rebalancing frequently which costs much more than the time saved per query.

4. Best configuration: 4 nodes, 150 virtual nodes
   Only 9.1% skew, 0.016ms query time, rebalancer rarely triggers.

---

## Input and output

Input: fused_features.npy from Milestone 1, shape (7128, 1184)

Output: 4 partitions, each a dictionary of image_id to feature_vector. Access with sharder.get_partition(node_id)

---

## Common errors and fixes

    FileNotFoundError: fused_features.npy
    → update the path inside np.load() to where your file is saved
