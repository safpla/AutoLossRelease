# __Author__ == 'Haowen Xu"
# __Date__ == "06-15-2018"

import numpy as np
import random
import scipy
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size, keys=None):
        self.buffer_size = buffer_size
        self.population = 0
        self.transition_buffer = deque(maxlen=buffer_size)
        if keys:
            self.keys = keys
        else:
            self.keys = ['state', 'action', 'reward', 'next_state']

    def add(self, transition):
        self.transition_buffer.append(transition)
        if self.population < self.buffer_size:
            self.population += 1

    def clear(self):
        self.population = 0
        self.transition_buffer.clear()

    def get_batch(self, batch_size):
        if self.population < batch_size:
            raise Exception('buffer has less data point than'
                            'batchsize {}'.format(batch_size))
        batch = random.sample(self.transition_buffer, batch_size)
        out_batch = {}
        for key in self.keys:
            out_batch[key] = []
        for t in batch:
            for key in self.keys:
                out_batch[key].append(t[key])

        return out_batch
