import numpy as np
from scipy.sparse import csr_matrix


class solver_parameters_class(object):
    def __init__(self, abs_solver_tolerance, max_iter_initial, max_iter_continuation, rel_solver_tolerance):

        assert isinstance(max_iter_initial, int), "maximum number of iterations in the initial solver (max_iter_initial) needs to be instance of int"
        assert isinstance(max_iter_initial, int), "maximum number of iterations in the continuation solver (max_iter_continuation) needs to be instance of int"

        self.abs_solver_tolerance = abs_solver_tolerance
        self.max_iter_initial = max_iter_initial
        self.max_iter_continuation = max_iter_continuation
        self.rel_solver_tolerance = rel_solver_tolerance