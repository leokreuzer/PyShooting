"""   
    This file describes the mechanical system class. 
    It defines all necessary inputs for a second order system later used for time integration.

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