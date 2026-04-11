import numpy as np
from consistent_hashing import ConsistentHashRing

class IndexSharding:
    """
    Takes all feature vectors and splits them across worker nodes
    based on the consistent hash ring.
    """

    def __init__(self, num_nodes=4, virtual_nodes=150):
        self.ring = ConsistentHashRing(num_nodes=num_nodes, virtual_nodes=virtual_nodes)
        self.num_nodes = num_nodes
        # each node gets a dictionary to store its images
        # { image_id: feature_vector }
        self.partitions = {i: {} for i in range(num_nodes)}

    def distribute(self, feature_vectors):
        """
        Takes all feature vectors and distributes them across nodes.
        feature_vectors: numpy array of shape (num_images, num_features)
        """
        print(f"\nDistributing {len(feature_vectors)} images across {self.num_nodes} nodes...")

        for image_id in range(len(feature_vectors)):
            node = self.ring.get_node(image_id)
            self.partitions[node][image_id] = feature_vectors[image_id]

        print("Distribution complete!")
        self._print_summary()

    def _print_summary(self):
        print("\nPartition Summary:")
        print(f"{'Node':<8} {'Images':<12} {'Percentage'}")
        print("-" * 35)
        total = sum(len(p) for p in self.partitions.values())
        for node_id, partition in self.partitions.items():
            count = len(partition)
            percent = (count / total) * 100
            print(f"Node {node_id:<4} {count:<12} {percent:.1f}%")

    def get_partition(self, node_id):
        """Return all images assigned to a specific node."""
        return self.partitions[node_id]

    def get_node_for_image(self, image_id):
        """Which node does this image belong to?"""
        return self.ring.get_node(image_id)