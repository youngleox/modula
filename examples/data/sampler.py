import torch
import numpy as np

class RandomSampler(torch.utils.data.Sampler):

    def __init__(self, data, batch_size):
        self.indices = range(len(data))
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield np.random.choice(self.indices, size=self.batch_size, replace=False)