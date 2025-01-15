import numpy as np
from scipy.sparse import csr_matrix


class steplength_parameters_class(object):
    def __init__(self, initial_stepsize, max_stepsize, min_stepsize, N_desired, initial_direction):

        assert isinstance(initial_direction, np.ndarray), "initial_direction needs to be instance of np.ndarray"
        assert isinstance(N_desired, int), "desired number of iterations in the continuation solver (N_desired) needs to be instance of int"
        
        self.initial_stepsize = initial_stepsize
        self.max_stepsize = max_stepsize
        self.min_stepsize = min_stepsize
        self.initial_direction = initial_direction
        self.N_desired = N_desired