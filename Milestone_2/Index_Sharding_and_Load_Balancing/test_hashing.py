import hashlib
import numpy as np
from consistent_hashing import ConsistentHashRing

data = np.load(r'C:\Users\MKT\Downloads\milestone1_Zain\fused_features.npy', allow_pickle=True)
print(f'Loaded {data.shape[0]} images')

ring = ConsistentHashRing(num_nodes=4, virtual_nodes=150)

print('\nSample assignments (first 10 images):')
print('Image ID       Assigned Node')
print('-' * 25)
for i in range(10):
    node = ring.get_node(i)
    print(f'Image {i}  -->  Node {node}')

print('\nDistribution across all 7128 images:')
counts = {i: 0 for i in range(4)}
for i in range(data.shape[0]):
    node = ring.get_node(i)
    counts[node] += 1

print('Node      Images Assigned      Percentage')
print('-' * 40)
for node, count in counts.items():
    percent = (count / data.shape[0]) * 100
    print(f'Node {node}    {count}               {percent:.1f}%')