import collections
import os
import sys
import math
import random
import numpy as np
import numpy.random
import scipy as sp
import scipy.stats

from clsh import pyLSH
import torch

class LSH:
    def __init__(self, func_, K_, L_, threads_=6):
        self.func = func_
        self.K = K_
        self.L = L_
        self.lsh_ = pyLSH(self.K, self.L, threads_)

    def insert(self, item, item_id):
        fp = self.func.hash(item)
        self.lsh_.insert(fp, item_id)

    def query(self, item, N):
        fp = self.func.hash(item)
        return self.lsh_.query(fp, N)

    def erase(self, item, item_id):
        fp = self.func.hash(item)
        self.lsh_.erase(fp, item_id)

    def clear(self):
        self.lsh_.clear()
