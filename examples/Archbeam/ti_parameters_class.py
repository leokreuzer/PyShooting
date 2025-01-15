"""   
    This file describes the time integration parameter class. 
    It defines all necessary inputs for the time integration.    

    Copyright (C) 2025  Leo Kreuzer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""



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

