import numpy as np
from sharding import IndexSharding

data = np.load(r'C:\Users\MKT\Downloads\milestone1_Zain\fused_features.npy', allow_pickle=True)
print(f'Loaded {data.shape[0]} images with {data.shape[1]} features each')

sharder = IndexSharding(num_nodes=4, virtual_nodes=150)
sharder.distribute(data)

print('\nLooking inside Node 0 (first 3 images):')
partition = sharder.get_partition(0)
for image_id, vector in list(partition.items())[:3]:
    print(f'Image {image_id} --> vector preview: {vector[:5]}')

print('\nWhich node does image 100 belong to?')
node = sharder.get_node_for_image(100)
print(f'Image 100 --> Node {node}')