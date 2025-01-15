import numpy as np
from scipy.sparse import csr_matrix


class ti_parameters_class(object):
    def __init__(self, Np, Ntpp, abs_ti_tolerance, rel_ti_tolerance, max_iter_NR, order_ti, error_const):

        assert isinstance(Np, int), "number of periods (Np) needs to be instance of int"
        assert isinstance(Ntpp, int), "number of timesteps per period (Ntpp) needs to be instance of int"
        assert isinstance(max_iter_NR, int), "maximum number of iterations in Newton-Raphson of the implicite time-intgerator (max_iter_NR) needs to be instance of int"

        self.Np = Np
        self.Ntpp = Ntpp
        self.abs_ti_tolerance = abs_ti_tolerance
        self.rel_ti_tolerance = rel_ti_tolerance
        self.max_iter_NR = max_iter_NR
        self.order_ti = order_ti
        self.error_const = error_const

