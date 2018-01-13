from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
import cython

cdef extern from "LSH.h":
    cdef cppclass LSH:
        LSH(int, int, int) except +
        void insert(int*, int) except +
        unordered_set[int] query(int*, int, int*) except +
        void erase(int*, int) except +
        void clear()

cdef class pyLSH:
    cdef LSH* c_lsh

    def __cinit__(self, int K, int L, int THREADS):
        self.c_lsh = new LSH(K, L, THREADS)

    def __dealloc__(self):
        del self.c_lsh

    def insert(self, vector[int] fp, int item_id):
        self.c_lsh.insert(&fp[0], item_id)

    def query(self, vector[int] fp, N):
        cdef int cL
        result = self.c_lsh.query(&fp[0], N, &cL)
        return list(result), cL

    def erase(self, vector[int] fp, int item_id):
        self.c_lsh.erase(&fp[0], item_id)
    
    def clear(self):
        self.c_lsh.clear()
