"""   
    The Shooting-Toolbox is a program that uses the Shooting Method for Periodic Steady-state
    Nonlinear Structural Dynamics. It also performs numerical continuation to build the frequency response, as well as nonlinear normal modes.

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
from dual_numbers_dynamics import Dual
from Mechanical_System_class import mechanical_system_class
from ti_parameters_class import ti_parameters_class
from solver_parameters_class import solver_parameters_class
from steplength_parameters_class import steplength_parameters_class
import solve_and_continue
import post_processing



n = 1   #2*n is the number of DoFs in the state-space function


# ti_parameters = [Np, Ntpp, abs_ti_tolerance, rel_ti_tolerance, max_iter NR. T-Integrator, order of T-Integrator, const. for error of EV]
ti_parameters = ti_parameters_class(1, 2**9, 10**-4, 10**-4, 20, 2, 5)

# solver_paramters = [abs_shooting_tolerance, max_iter initial solution, max_iter continuation, rel_shooting_tolerance]
solver_parameters = solver_parameters_class(10**-5, 30, 20, 10**-10)

# steplength_parameters = [initial stepsize, maximum stepsize, minimum stepsize, desired number of iterations, initial direction (w.r.t. Om)]
steplength_parameters = steplength_parameters_class(0.01, 0.5, 0.0, 1, np.array([0,0,1]))



# initialization of the solution vectors
N_sol = 600
X_sol = np.zeros([N_sol, 2*n+1])
Y_sol = np.zeros([N_sol, ti_parameters.Np*ti_parameters.Ntpp+1, 2*n])
number_iterations = np.zeros([N_sol])
steplength = np.zeros([N_sol])
p_sol = np.zeros([N_sol-1, 2, 2*n+1])
Jacobian_sol = np.zeros([N_sol, 2*n, 2*n+1])
stability_sol = np.zeros([N_sol])
EV_monodromy_sol = np.zeros([N_sol, 2*n, 2*n], complex)



# initial guess
X0 = np.array([0.0, 0.0, 0.5])



# call solve_and_continue to solve the system
X_sol, Y_sol, Jacobian_sol, number_iterations, steplength, p_sol, stability_sol, EV_monodromy_sol = solve_and_continue.fra(X0, n, ti_parameters, solver_parameters, N_sol, steplength_parameters)


# post processing
post_processing.saving(X_sol, Y_sol, N_sol, ti_parameters, n, solver_parameters, Jacobian_sol, steplength_parameters)
post_processing.plotting(X_sol, Y_sol, N_sol, ti_parameters, n, solver_parameters, Jacobian_sol, steplength_parameters)