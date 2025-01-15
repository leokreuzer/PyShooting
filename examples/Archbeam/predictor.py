"""   
    This file describes the predictor step of numerical continuation. 
    It defines the predictor functions used in numerical continuation by calculating a prediction vector.

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
import scipy as sp
from state_space_functions import dynamics


##### tangent predictor for frequency response analysis #####
def tangent_fra(Jacobian, X_last, p_prev, s, n, sk, X_prev, dir):

    p = sp.linalg.null_space(Jacobian)
    p = p.reshape(2*n+1,)

    p_ref = (X_last - X_prev)/np.linalg.norm(X_last - X_prev)

    # first step should have positiv frequency-step
    if sk == 0:
        if np.sign(s) != np.sign(np.dot(p.T, dir)):
            s = -s
    # controlling the direction of the continuation
    if sk>0:
        if np.sign(s) != np.sign(np.dot(p.T, p_ref)):
            s = -s

    #prediction step
    X = X_last + s*p
    X_pred = X

    return X_pred, X, p, s


##### tangent predictor for nonlinear modal analysis #####
def tangent_nma(Jacobian, X_last, p_prev, s, n, sk, X_prev, dir):

    U, sv, VH = sp.linalg.svd(Jacobian)
    p = VH[-2:, :]

    if sk == 0:
        p_ref = dir

    elif sk>0:
        p_ref = (X_last - X_prev)/np.linalg.norm(X_last - X_prev)

    V_ref = np.zeros(2*n + 1)
    V_ref[:2*n] = dynamics(X_last[:2*n], X_last[-1], 0)

    p = p.T @ np.array([[0, -1], [1, 0]]) @ (p @ V_ref)
    p /= np.linalg.norm(p)


    if np.sign(s) != np.sign(np.dot(p.T, p_ref)):
        s = -s

    #prediction step
    X = X_last + s*p
    X_pred = X

    return X_pred, X, p, s


##### secant predictor #####
def secant(Jacobian, X_last, p_prev, s, n, sk, X_prev, dir):

    if sk == 0:
        return tangent_nma(Jacobian, X_last, p_prev, s, n, sk, X_prev, dir)
    
    p = (X_last - X_prev)/(np.linalg.norm(X_last-X_prev))

    # controlling the direction of the continuation
    if sk>0:
        if np.sign(s) != np.sign(s*np.dot(p.T, p_prev)):
            s = -s

    #prediction step
    X = X_last + s*p
    X_pred = X

    return X_pred, X, p, s