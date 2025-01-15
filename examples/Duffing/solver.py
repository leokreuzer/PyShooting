"""   
    This file describes the Newton-Raphson solver. 
    It defines the two distinct types of solvers, one for finding a inital solution and one for finding a solution in the continuation.
    

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
import T_integrator

def initial(X, system, fex_values, ti_parameters, solver_parameters):
    n = system.n

    for ik in range(solver_parameters.max_iter_initial):
        # calculation of the residual and jacobian for the shooting and continuation problem
        R, Jacobian, y_end, Y, monodromy_matrix = T_integrator.Newmark(X, system, ti_parameters, fex_values)

        if ik == 0:
            max_abs_R_initial = max(abs(R))
        
        if max(abs(R)) < solver_parameters.abs_solver_tolerance or max(abs(R)) < solver_parameters.rel_solver_tolerance*max_abs_R_initial:
            break

        else:
            A_shooting = monodromy_matrix - np.eye(2*n)
            X[:2*n] -= np.linalg.solve(A_shooting, R)


    if max(abs(R)) > solver_parameters.abs_solver_tolerance and max(abs(R)) > solver_parameters.rel_solver_tolerance*max_abs_R_initial:
        print("solver.initial could not find a solution |R| < shooting_tolerance")

    return X, X, Jacobian, Y, ik


def continuation(X, system, fex_values, ti_parameters, solver_parameters, parametrization, grad_parametrization):
    n = system.n
    run_var = 0

    for ik in range(solver_parameters.max_iter_continuation):
        # calculation of the residual and jacobian for the shooting and continuation problem
        R, Jacobian, y_end, Y, monodromy_matrix = T_integrator.Newmark(X, system, ti_parameters, fex_values)

        R_enlarged = np.hstack([R, parametrization(X)])

        if ik == 0:
            max_abs_R_initial = max(abs(R_enlarged))

        if max(abs(R_enlarged)) < solver_parameters.abs_solver_tolerance or max(abs(R_enlarged)) < solver_parameters.rel_solver_tolerance*max_abs_R_initial:
            break

        else:
            Jacobian_enlarged = np.vstack([Jacobian, grad_parametrization(X)])
            X -= np.linalg.solve(Jacobian_enlarged, R_enlarged)

    if max(abs(R_enlarged)) > solver_parameters.abs_solver_tolerance and max(abs(R_enlarged)) > solver_parameters.rel_solver_tolerance*max_abs_R_initial:
        print("solver.continuation could not find a solution |R| < shooting_tolerance")
        run_var = 1

    return X, X, Jacobian, Y, ik, run_var
