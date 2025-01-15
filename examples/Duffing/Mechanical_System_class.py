import numpy as np
from scipy.sparse import csr_matrix


class mechanical_system_class(object):
    def __init__(self, M: np.ndarray | float, C, K, f_nl):

        # assertions
        assert isinstance(M, np.ndarray | float | int), "M needs to be a np.ndarray (or if 1-DoF-System float/int)"
        assert isinstance(C, np.ndarray | float | int), "C needs to be a np.ndarray (or if 1-DoF-System float/int)"
        assert isinstance(K, np.ndarray | float | int), "K needs to be a np.ndarray (or if 1-DoF-System float/int)"
        
        # write inputs into the class-object
        self.M = M
        self.C = C
        self.K = K

        if isinstance(M, float | int):
            self.n = 1
        else:
            self.n = M.shape[0]

        self.f_nl = f_nl