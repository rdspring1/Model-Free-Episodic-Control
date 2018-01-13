import numpy as np
import random

from lsh import LSH
from simhash import SimHash

# key - D dimensional vector -> Not Hashable
# [(External) Variable -> CUDA -> Torch] => [Internal (numpy vector)]

# LSH hash tables -> Create LSH Fingerprint from Keys, Store only memory indices
# Key2Memory -> Dictionary - key -> memory_index
# memory index - N x L integers -> Numpy Array
# key - N x D - D dimensional vectors -> Numpy Array
# values - N x M - N-Step Q-Values -> Numpy Array

class DND:
    MAX_SIZE = 25 
    TM = 0.1
    def __init__(self, N, D, K, L):
      self.lsh = LSH(SimHash(D, K, L), K, L)
      self.keys = np.zeros((N, D), dtype=np.float32)
      self.values = np.zeros((N, 1), dtype=np.float32)
      self.lru = np.zeros(N, dtype=np.float32)
      self.key2idx = dict()

      self.size = 0
      self.max_memory = N
      self.K = K
      self.L = L 

    def __contains__(self, key):
        return tuple(key) in self.key2idx

    def __getitem__(self, key):
        try:
            index = self.key2idx[tuple(key)]
            self.lru[index] += DND.TM
            return self.values[index]
        except:
            return None

    def __setitem__(self, key, value):
        item = tuple(key)
        try:
            # 1) Find memory index for key vector
            index = self.key2idx[item]
        except:
            # 2) Add key vector if not present 
            if self.size >= self.max_memory:
                # 3) If memory is full, select LRU memory index and remove from LSH hash tables
                index = np.argmin(self.lru)
                self.lsh.erase(self.keys[index], index)
            else:
                index = self.size
                self.size += 1

            # Rehash key into LSH hash tables
            self.lsh.insert(key, index)
            self.key2idx[item] = index

            # Add new key to memory
            self.keys[index] = key
        finally:
            # Update memory value
            self.values[index] = value
            self.lru[index] += DND.TM

    def retrieve(self, query):
        # Collect memory indices from LSH hash tables
        indices, cL = self.lsh.query(query.data, DND.MAX_SIZE)

        # Gather keys and values from memory
        keys = self.keys[indices]
        values = self.values[indices]
        self.lru[indices] += DND.TM

        assert(keys.shape[0] == values.shape[0])
        return keys, values, indices, cL
