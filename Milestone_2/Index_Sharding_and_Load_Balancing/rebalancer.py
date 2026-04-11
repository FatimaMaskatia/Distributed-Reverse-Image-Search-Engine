import numpy as np
from monitor import PartitionMonitor

class Rebalancer:
    def __init__(self, sharder, skew_threshold=0.15, max_rounds=10):
        self.sharder = sharder
        self.monitor = PartitionMonitor(sharder, skew_threshold)
        self.max_rounds = max_rounds
        self.moved_images = []  # track every image that gets moved

    def rebalance(self):
        self.moved_images = []  # reset at the start of each rebalance

        for round_num in range(1, self.max_rounds + 1):
            sizes = self.monitor.get_partition_sizes()
            total = sum(sizes.values())
            average = total / len(sizes)

            overloaded = []
            underloaded = []

            for node_id, size in sizes.items():
                diff = (size - average) / average
                if diff > self.monitor.skew_threshold:
                    overloaded.append(node_id)
                elif diff < -self.monitor.skew_threshold:
                    underloaded.append(node_id)

            if not overloaded:
                print(f"Balanced after {round_num - 1} rounds!")
                break

            print(f"\nRound {round_num}: Overloaded={overloaded} Underloaded={underloaded}")

            for over_node in overloaded:
                for under_node in underloaded:
                    over_size = len(self.sharder.partitions[over_node])
                    under_size = len(self.sharder.partitions[under_node])
                    current_avg = (over_size + under_size) / 2
                    images_to_move = int(over_size - current_avg)

                    if images_to_move <= 0:
                        continue

                    print(f"Moving {images_to_move} images from Node {over_node} to Node {under_node}")

                    keys_to_move = list(self.sharder.partitions[over_node].keys())[:images_to_move]
                    for image_id in keys_to_move:
                        vector = self.sharder.partitions[over_node].pop(image_id)
                        self.sharder.partitions[under_node][image_id] = vector
                        self.sharder.ring.set_node_override(image_id, under_node)  # update master list
                        self.moved_images.append(image_id)  # track it

        print("\nRebalancing complete!")
        self.monitor.check_skew()