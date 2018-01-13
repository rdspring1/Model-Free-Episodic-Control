from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
import cython

cdef class SimHash:
    cdef int d
    cdef int k
    cdef int L
    cdef np.ndarray rp

    def __cinit__(self, int d_, int k_, int L_, int seed_=8191):
        self.d = d_
        self.k = k_
        self.L = L_
        self.rp = self.generate(d_, k_, L_, seed_)

    def generate(self, d, k, L, seed):
        rand_gen = np.random.RandomState(seed)
        matrix = rand_gen.randn(d, k*L)
        positive = np.greater_equal(matrix, 0.0)
        negative = np.less(matrix, 0.0)
        return positive.astype(np.float32) - negative.astype(np.float32)

    def hash(self, data):
        srp = np.greater(np.matmul(data, self.rp), 0).astype(np.int32)
        return self.fingerprint(srp)

    def fingerprint(self, srp):
        cdef vector[int] fps
        cdef int offset
        cdef int fp
        for idx in range(self.L):
            fp = 0
            offset = idx * self.k
            for jdx in range(self.k):
                fp |= (srp[offset + jdx] << jdx)
            fps.push_back(fp)
        return fps
