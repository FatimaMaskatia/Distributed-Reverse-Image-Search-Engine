\# Milestone 2 - Index Sharding \& Load Balancing



\---



\## What this does

This module takes all the image feature vectors from Milestone 1 and splits

them evenly across multiple worker nodes. It also watches if any node gets

too much work and automatically fixes it.



The 7128 images from Milestone 1 each have 1184 features. This module

distributes those across 4 worker nodes so that instead of one machine

searching all 7128 images, 4 workers search in parallel which is much faster.



\---



\## Files included

\- consistent\_hashing.py  — decides which image goes to which node

\- sharding.py            — splits the images across nodes

\- monitor.py             — detects if any node is overloaded

\- rebalancer.py          — fixes overloaded nodes automatically

\- evaluation.py          — tests different configurations and documents results



\---



\## Requirements

Make sure you have numpy installed:



&#x20;   pip install numpy



\---



\## Before running anything - update the file path

Every file that loads the feature vectors has this line somewhere:



&#x20;   np.load('fused\_features.npy', allow\_pickle=True)



Change the path inside np.load() to wherever your fused\_features.npy

file is actually saved on your computer.



\---



\## How to run the test files



&#x20;   python test\_hashing.py      — tests that images are assigned to nodes correctly

&#x20;   python test\_sharding.py     — tests that images are split across nodes

&#x20;   python test\_monitor.py      — tests that overloaded nodes are detected

&#x20;   python test\_rebalancer.py   — tests that overloaded nodes are fixed automatically

&#x20;   python evaluation.py        — runs all configurations and shows a results table



\---



\## For LSH Index Construction 

This section explains what you need to plug this module into

your distributed LSH index.



The sharder gives you one partition per node. Each partition is a dictionary

of { image\_id: feature\_vector }. You build your LSH index separately on

each partition.



Update the path to fused\_features.npy:



&#x20;   from sharding import IndexSharding

&#x20;   import numpy as np



&#x20;   # load the feature vectors from Milestone 1

&#x20;   data = np.load(r'YOUR PATH TO fused\_features.npy HERE', allow\_pickle=True)

&#x20;   # data shape is (7128, 1184) — 7128 images, 1184 features each



&#x20;   # set up the sharder

&#x20;   sharder = IndexSharding(num\_nodes=4)

&#x20;   sharder.distribute(data)



&#x20;   # get the images assigned to a specific node

&#x20;   # each partition is a dictionary: { image\_id: feature\_vector }

&#x20;   partition = sharder.get\_partition(node\_id=0)



&#x20;   # example of what a partition looks like:

&#x20;   # {

&#x20;   #   1:  array(\[0.108, 0.175, 0.154, ...]),   # image 1 and its 1184 features

&#x20;   #   2:  array(\[0.135, 0.159, 0.152, ...]),   # image 2 and its 1184 features

&#x20;   #   45: array(\[0.201, 0.143, 0.198, ...]),   # image 45 and its 1184 features

&#x20;   #   ...

&#x20;   # }



&#x20;   # loop through all 4 nodes and build your LSH index per node

&#x20;   for node\_id in range(4):

&#x20;       partition = sharder.get\_partition(node\_id)

&#x20;       image\_ids = list(partition.keys())         # list of image IDs on this node

&#x20;       feature\_vectors = list(partition.values()) # their corresponding feature vectors

&#x20;       print(f'Node {node\_id} has {len(partition)} images')

&#x20;       # feed image\_ids and feature\_vectors into your LSH index 



&#x20;   # To check which node a specific image belongs to

&#x20;   node = sharder.get\_node\_for\_image(image\_id=100)

&#x20;   print(f'Image 100 is on Node {node}')



&#x20;   # To get list of all node IDs

&#x20;   nodes = sharder.ring.get\_all\_nodes()

&#x20;   # returns \[0, 1, 2, 3]



What each variable contains:

&#x20;   data               — full numpy array, shape (7128, 1184)

&#x20;   sharder            — the main object, holds all 4 partitions

&#x20;   partition          — dictionary of { image\_id: feature\_vector } for one node

&#x20;   image\_ids          — list of image IDs assigned to that node e.g. \[1, 2, 45, ...]

&#x20;   feature\_vectors    — list of 1184-dimensional numpy arrays for those images

&#x20;   node\_id            — integer 0, 1, 2, or 3



Recommended settings:

&#x20;   num\_nodes = 4        number of worker nodes

&#x20;   virtual\_nodes = 150  controls how evenly images are distributed

&#x20;                        do not go below 100, distribution becomes uneven



\---





\## For Caching \& Node Infrastructure - How to use this in your code



The sharder object holds all partition data and node assignments.

Use it to check node status, monitor load, and trigger rebalancing

when needed.



&#x20;   from sharding import IndexSharding

&#x20;   from monitor import PartitionMonitor

&#x20;   from rebalancer import Rebalancer

&#x20;   import numpy as np



&#x20;   data = np.load(r'YOUR PATH TO fused\_features.npy HERE', allow\_pickle=True)



&#x20;   sharder = IndexSharding(num\_nodes=4, virtual\_nodes=150)

&#x20;   sharder.distribute(data)



&#x20;   # check how many images each node has

&#x20;   for node\_id in range(4):

&#x20;       partition = sharder.get\_partition(node\_id)

&#x20;       print(f'Node {node\_id} has {len(partition)} images')



&#x20;   # check if any node is overloaded

&#x20;   monitor = PartitionMonitor(sharder, skew\_threshold=0.15)

&#x20;   needs\_rebalancing = monitor.check\_skew()

&#x20;   # returns True if any node is overloaded, False if everything is fine



&#x20;   # if overloaded, rebalance automatically

&#x20;   if needs\_rebalancing:

&#x20;       rebalancer = Rebalancer(sharder, skew\_threshold=0.15)

&#x20;       rebalancer.rebalance()



&#x20;   # get partition sizes as a dictionary { node\_id: image\_count }

&#x20;   sizes = monitor.get\_partition\_sizes()

&#x20;   print(sizes)

&#x20;   # returns {0: 1944, 1: 1693, 2: 1743, 3: 1748}



\## How it works internally



\### Consistent Hashing

Each image is assigned to a worker node using a consistent hash ring.

Instead of placing each node once on the ring, each node gets multiple

positions called virtual nodes. More virtual nodes means more even

distribution of images across workers.



With 10 virtual nodes:

&#x20;   Node 0 = 22.2%,  Node 1 = 25.9%,  Node 2 = 18.1%,  Node 3 = 33.8%   (uneven)



With 150 virtual nodes:

&#x20;   Node 0 = 27.3%,  Node 1 = 23.8%,  Node 2 = 24.5%,  Node 3 = 24.5%   (even)



\### Partition Monitoring

The monitor tracks how many images each node holds and compares it to

the average. If any node differs from the average by more than 15%,

it is flagged as overloaded or underloaded and rebalancing is triggered.



The 15% threshold was chosen as a balance between rebalancing overhead

and load fairness, consistent with standard practices in distributed systems.

It can be tuned based on system requirements:

\- Stricter systems  : use 10%

\- Looser systems    : use 20-25%



\### Reactive Rebalancing

When skew is detected, the rebalancer moves images from overloaded nodes

to underloaded nodes. It runs in multiple rounds until everything is

balanced or a maximum of 10 rounds is reached.



Example (intentionally unbalanced with 5 virtual nodes):

&#x20;   BEFORE: Node 0 = 51.3%,  Node 1 = 17.5%,  Node 2 = 17.0%,  Node 3 = 14.3%

&#x20;   AFTER:  Node 0 = 27.2%,  Node 1 = 23.6%,  Node 2 = 25.7%,  Node 3 = 23.6%

&#x20;   Fixed in just 2 rounds.



\## Ring Consistency After Rebalancing



When the rebalancer moves images between nodes, it also updates the hash ring

via an overrides dictionary in consistent\_hashing.py. This ensures that

get\_node\_for\_image() always returns the correct current location of an image

even after it has been moved.



Without this, the ring would still point to the original node after

rebalancing, and queries would silently return wrong results, no crash,

just missing images.



Verified by test\_rebalancer.py which confirms that moved images are tracked

correctly and the ring stays in sync after rebalancing.



\---



\## Evaluation results

Tested on 7128 images with 1184 features each:



&#x20;   Nodes   VNodes   Dist Time(s)   Query Time(ms)   Max Skew

&#x20;   ----------------------------------------------------------

&#x20;   2       150      0.099          0.0161           5.4%

&#x20;   4       150      0.241          0.0280           9.1%

&#x20;   8       150      0.393          0.0522           21.3%

&#x20;   4       10       0.055          0.0061           35.2%

&#x20;   4       50       0.114          0.0154           27.6%

&#x20;   4       150      0.183          0.0267           9.1%



Key findings:



1\. More virtual nodes = better balance but slightly slower query time

&#x20;      10 vnodes  = 35.2% skew  (too uneven)

&#x20;      50 vnodes  = 27.6% skew  (okay but causes frequent rebalancing)

&#x20;      150 vnodes = 9.1%  skew  (best balance)



2\. More nodes = more coordination overhead

&#x20;      2 nodes = 0.016ms query time

&#x20;      4 nodes = 0.028ms query time

&#x20;      8 nodes = 0.052ms query time



3\. Why not 4 nodes with 50 virtual nodes even though it is faster?

&#x20;      50 vnodes is faster per query (0.015ms vs 0.027ms) but has 27.6% skew.

&#x20;      That skew triggers rebalancing frequently which moves hundreds of images

&#x20;      between nodes. That rebalancing cost is much more than the 0.012ms saved.

&#x20;      Total time with rebalancing = more than 150 vnodes which rarely rebalances.



4\. Best spot for this dataset: 4 nodes, 150 virtual nodes

&#x20;      Only 9.1% skew, 0.028ms query time, rebalancer rarely triggers.



\---



\## Input and output



Input:

&#x20;   fused\_features.npy from Milestone 1

&#x20;   Shape: (7128, 1184) — 7128 images, 1184 features each



Output:

&#x20;   4 partitions, each a dictionary of { image\_id: feature\_vector }

&#x20;   Access with: sharder.get\_partition(node\_id)



\---



\## Common error and fix



&#x20;   FileNotFoundError: fused\_features.npy

&#x20;   → the path inside np.load() is wrong, update it to where your file is saved

