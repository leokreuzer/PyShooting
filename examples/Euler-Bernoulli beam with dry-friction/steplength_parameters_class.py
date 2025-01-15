"""   
    This file describes the step-length parameter class. 
    It defines all necessary inputs for the step-length adaptation.    

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


class steplength_parameters_class(object):
    def __init__(self, initial_stepsize, max_stepsize, min_stepsize, N_desired, initial_direction):

        assert isinstance(initial_direction, np.ndarray), "initial_direction needs to be instance of np.ndarray"
        assert isinstance(N_desired, int), "desired number of iterations in the continuation solver (N_desired) needs to be instance of int"
        
        self.initial_stepsize = initial_stepsize
        self.max_stepsize = max_stepsize
        self.min_stepsize = min_stepsize
        self.initial_direction = initial_direction
        self.N_desired = N_desired
