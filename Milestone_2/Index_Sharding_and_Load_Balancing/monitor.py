from sharding import IndexSharding
import numpy as np

class PartitionMonitor:
    """
    Watches partition sizes and detects load skew.
    Load skew = one node has way more images than others.
    """

    def __init__(self, sharder, skew_threshold=0.15):
        """
        sharder        : the IndexSharding object we want to monitor
        skew_threshold : if any node differs from average by more than
                         15%, we consider it skewed and trigger rebalancing
        """
        self.sharder = sharder
        self.skew_threshold = skew_threshold

    def get_partition_sizes(self):
        """Return how many images each node currently has."""
        sizes = {}
        for node_id in range(self.sharder.num_nodes):
            sizes[node_id] = len(self.sharder.get_partition(node_id))
        return sizes

    def check_skew(self):
        """
        Check if any node is overloaded or underloaded.
        Returns True if rebalancing is needed, False if everything is fine.
        """
        sizes = self.get_partition_sizes()
        total = sum(sizes.values())
        average = total / len(sizes)

        print("\nMonitoring partition sizes...")
        print(f"{'Node':<8} {'Images':<12} {'Vs Average':<15} {'Status'}")
        print("-" * 50)

        rebalance_needed = False

        for node_id, size in sizes.items():
            diff = (size - average) / average
            if diff > self.skew_threshold:
                status = "OVERLOADED!"
                rebalance_needed = True
            elif diff < -self.skew_threshold:
                status = "UNDERLOADED!"
                rebalance_needed = True
            else:
                status = "OK"
            print(f"Node {node_id:<4} {size:<12} {diff:+.1%}          {status}")

        print(f"\nAverage images per node: {average:.0f}")

        if rebalance_needed:
            print("WARNING: Load skew detected! Rebalancing needed.")
        else:
            print("All nodes are balanced. No rebalancing needed.")

        return rebalance_needed