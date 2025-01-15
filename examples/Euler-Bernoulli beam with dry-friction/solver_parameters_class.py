"""   
    This file describes the solver parameter class. 
    It defines all necessary inputs for the Newton-Raphson solver later used to determine roots of the residual function.

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


class solver_parameters_class(object):
    def __init__(self, abs_solver_tolerance, max_iter_initial, max_iter_continuation, rel_solver_tolerance):

        assert isinstance(max_iter_initial, int), "maximum number of iterations in the initial solver (max_iter_initial) needs to be instance of int"
        assert isinstance(max_iter_initial, int), "maximum number of iterations in the continuation solver (max_iter_continuation) needs to be instance of int"

        self.abs_solver_tolerance = abs_solver_tolerance
        self.max_iter_initial = max_iter_initial
        self.max_iter_continuation = max_iter_continuation
        self.rel_solver_tolerance = rel_solver_tolerance
