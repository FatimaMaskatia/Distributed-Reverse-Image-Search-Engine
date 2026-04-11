import hashlib
import numpy as np

class ConsistentHashRing:
    def __init__(self, num_nodes=4, virtual_nodes=150):
        self.num_nodes = num_nodes
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        self.overrides = {}  # sticky notes: { image_id: node_id } for manually moved images
        self._build_ring()

    def _build_ring(self):
        for node_id in range(self.num_nodes):
            for v in range(self.virtual_nodes):
                key = f"node_{node_id}_vnode_{v}"
                position = self._hash(key)
                self.ring[position] = node_id
                self.sorted_keys.append(position)
        self.sorted_keys.sort()
        print(f"Ring built with {self.num_nodes} nodes and {self.virtual_nodes} virtual nodes each")
        print(f"Total positions on ring: {len(self.sorted_keys)}")

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def set_node_override(self, image_id, node_id):
        """Leave a sticky note: this image was manually moved to a different node."""
        self.overrides[image_id] = node_id

    def get_node(self, image_id):
        # check sticky notes first — if image was moved, return its new location
        if image_id in self.overrides:
            return self.overrides[image_id]
        if not self.ring:
            raise Exception("Ring is empty!")
        position = self._hash(str(image_id))
        for key in self.sorted_keys:
            if position <= key:
                return self.ring[key]
        return self.ring[self.sorted_keys[0]]

    def get_all_nodes(self):
        return list(range(self.num_nodes))