"""   
    This file describes the corrector step of numerical continuation. 
    It defines the corrector functions used in numerical continuation by giving a parametrization of the search space.

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
import solver


def orthogonal(X, X_pred, p, system, fex_values, ti_parameters, solver_parameters, s, X_last):

    def parametrization(X):
        return np.dot(p,(X-X_pred))
    
    def grad_parametrization(X):
        return p

    X_sol, X, Jacobian_sol, Y_sol, number_iterations, run_var = solver.continuation(X, system, fex_values, ti_parameters, solver_parameters, parametrization, grad_parametrization)

    return X_sol, X, Jacobian_sol, Y_sol, number_iterations, run_var

def arclength(X, X_pred, p, system, fex_values, ti_parameters, solver_parameters, s, X_last):

    def parametrization(X):
        return np.dot((X-X_last),(X-X_last)) - s**2
    
    def grad_parametrization(X):
        return 2*(X - X_last)

    X_sol, X, Jacobian_sol, Y_sol, number_iterations, run_var = solver.continuation(X, system, fex_values, ti_parameters, solver_parameters, parametrization, grad_parametrization)  

    return X_sol, X, Jacobian_sol, Y_sol, number_iterations, run_var
