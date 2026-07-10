"""Ported verbatim from degcn-skating-actions/feeders/balanced_sampler.py.

Yields batches with an equal number of samples per class, reshuffling +
wrapping within a class when it runs out mid-epoch. Requires
`batch_size % num_classes == 0`.
"""
import random
import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, drop_last=False):
        """
        labels: list/array/tensor of class indices (len = dataset size)
        batch_size: total batch size
        drop_last: drop final incomplete batch
        """
        self.labels = np.array(labels)
        self.drop_last = drop_last

        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)

        assert batch_size % self.num_classes == 0, (
            "Batch size must be divisible by number of classes"
        )

        self.samples_per_class = batch_size // self.num_classes

        # Build index list per class
        self.class_indices = {
            c: np.where(self.labels == c)[0].tolist()
            for c in self.classes
        }

        self.batch_size = batch_size
        self.dataset_size = len(labels)

    def __iter__(self):
        # Shuffle indices for each class at start of epoch
        for c in self.classes:
            random.shuffle(self.class_indices[c])

        # Track position in each class list
        class_pointers = {c: 0 for c in self.classes}

        batches = []
        while True:
            batch = []

            for c in self.classes:
                indices = self.class_indices[c]
                ptr = class_pointers[c]

                # If not enough samples left, reshuffle and wrap around
                if ptr + self.samples_per_class > len(indices):
                    random.shuffle(indices)
                    self.class_indices[c] = indices
                    ptr = 0

                batch.extend(indices[ptr : ptr + self.samples_per_class])
                class_pointers[c] = ptr + self.samples_per_class

            if len(batch) != self.batch_size:
                break

            random.shuffle(batch)
            batches.append(batch)

            # Stop if we've covered roughly one epoch
            if len(batches) * self.batch_size >= self.dataset_size:
                break

        for batch in batches:
            yield batch

        # Yield final incomplete batch if present and not drop_last
        if not self.drop_last and len(batch) > 0 and len(batch) < self.batch_size:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size
