"""   
    This file describes the solver and continuation loop. 
    It defines the main analysis types (frequency response analysis, nonlinear modal analysis)
    and selects the desired function of the predictor, corrector and step-length adaptation.

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
import predictor
import corrector
import steplength_adaptation
import stability


def fra(X0, system, ti_parameters, solver_parameters, Fex, N_sol, steplength_parameters):
    n = system.n



    # precompute all datapoints of the external force
    ##### IMPORTANT: if rungekutta is used: 2*(ti_parameters.Np*ti_parameters.Ntpp) + 1 #####
    tau = np.linspace(0, 2*np.pi, 2*(ti_parameters.Np*ti_parameters.Ntpp) + 1, endpoint=True).reshape(-1,1)
    # tau = np.linspace(0, 2*np.pi, (ti_parameters.Np*ti_parameters.Ntpp) + 1, endpoint=True).reshape(-1,1)
    fex_values = np.multiply(Fex,np.cos(tau))



    # initialize outputs
    X_sol = np.zeros([N_sol, 2*n+1])
    Y_sol = np.zeros([N_sol, ti_parameters.Np*ti_parameters.Ntpp+1, 2*n])
    number_iterations = np.zeros([N_sol])
    steplength = np.zeros([N_sol])
    p_sol = np.zeros([N_sol-1, 2*n+1])
    Jacobian_sol = np.zeros([N_sol, 2*n, 2*n+1])
    stability_sol = np.zeros([N_sol])
    EV_monodromy_sol = np.zeros([N_sol, 2*n, 2*n], complex)



    # find initial solution
    X_sol[0,:], X, Jacobian_sol[0,:,:], Y_sol[0,:,:], number_iterations[0] = solver.initial(X0, system, fex_values, ti_parameters, solver_parameters)

    print(f"Initial solution at omega = {round(X_sol[0, -1], 8)} found")

    stability_sol[0], EV_monodromy_sol[0,:,:] = stability.stability_analysis(Jacobian_sol[0,:,:], n, ti_parameters)



    # safe initial steplength
    steplength[0] = steplength_parameters.initial_stepsize

    for sk in range(N_sol-1):
        run_var = 1
        while run_var:
            X_pred, X, p_sol[sk,:], steplength[sk] = predictor.tangent_fra(Jacobian_sol[sk, :, :], X_sol[sk,:], p_sol[sk-1,:], steplength[sk], n, sk, X_sol[sk-1], steplength_parameters.initial_direction, system, fex_values)

            X_sol[sk+1,:], X, Jacobian_sol[sk+1,:,:], Y_sol[sk+1,:,:], number_iterations[sk+1], run_var = corrector.orthogonal(X, X_pred, p_sol[sk,:], system, fex_values, ti_parameters, solver_parameters, steplength[sk], X_sol[sk, :])

            if run_var == 1:
                steplength[sk] = steplength[sk]*(0.5)
                print(f'reducing stepsize by a factor 0.5')
        
        print(f"Continuation at omega = {round(X_sol[sk+1, -1], 8)}; Solution number: {sk+2}/{N_sol} ({round(float(((sk+2)/N_sol))*100, 3)}%)")

        steplength[sk+1] = steplength_adaptation.exponential(number_iterations[sk+1], steplength[sk], steplength_parameters)
        
        stability_sol[sk+1], EV_monodromy_sol[sk+1,:,:] = stability.stability_analysis(Jacobian_sol[sk+1,:,:], n, ti_parameters)



    return X_sol, Y_sol, Jacobian_sol, number_iterations, steplength, p_sol, stability_sol, EV_monodromy_sol





def nma(X0, system, ti_parameters, solver_parameters, Fex, N_sol, steplength_parameters):
    n = system.n
    if n == 1:
        if system.C != 0:
            system.C = 0
            print("C set to 0, in order to find periodic solutions")
    elif n > 1:
        if system.C.any() != 0:
            system.C = np.zeros([2*n, 2*n])
            print("C set to 0, in order to find periodic solutions")
    


    # precompute all datapoints of the external force 
    ##### IMPORTANT: if rungekutta is used: 2*(ti_parameters.Np*ti_parameters.Ntpp) + 1 #####
    fex_values = np.zeros([2*(ti_parameters.Np*ti_parameters.Ntpp) + 1])
    # fex_values = np.zeros([(ti_parameters.Np*ti_parameters.Ntpp) + 1])



    # initialize outputs
    X_sol = np.zeros([N_sol, 2*n+1])
    Y_sol = np.zeros([N_sol, ti_parameters.Np*ti_parameters.Ntpp+1, 2*n])
    number_iterations = np.zeros([N_sol])
    steplength = np.zeros([N_sol])
    p_sol = np.zeros([N_sol-1, 2*n+1])
    Jacobian_sol = np.zeros([N_sol, 2*n, 2*n+1])
    stability_sol = np.zeros([N_sol])
    EV_monodromy_sol = np.zeros([N_sol, 2*n, 2*n], complex)



    # find initial solution
    X_sol[0,:], X, Jacobian_sol[0,:,:], Y_sol[0,:,:], number_iterations[0] = solver.initial(X0, system, fex_values, ti_parameters, solver_parameters)

    print(f"Initial solution at omega = {round(X_sol[0, -1], 8)} found")

    stability_sol[0], EV_monodromy_sol[0,:,:] = stability.stability_analysis(Jacobian_sol[0,:,:], n, ti_parameters)



    # safe initial steplength
    steplength[0] = steplength_parameters.initial_stepsize

    for sk in range(N_sol-1):
        run_var = 1
        while run_var:
            X_pred, X, p_sol[sk,:], steplength[sk] = predictor.tangent_nma(Jacobian_sol[sk, :, :], X_sol[sk,:], p_sol[sk-1,:], steplength[sk], n, sk, X_sol[sk-1], steplength_parameters.initial_direction, system, fex_values)

            X_sol[sk+1,:], X, Jacobian_sol[sk+1,:,:], Y_sol[sk+1,:,:], number_iterations[sk+1], run_var = corrector.orthogonal(X, X_pred, p_sol[sk,:], system, fex_values, ti_parameters, solver_parameters, steplength[sk], X_sol[sk, :])

            if run_var == 1:
                steplength[sk] = steplength[sk]*(0.5)
                print(f'reducing stepsize by a factor 0.5')
        
        print(f"Continuation at omega = {round(X_sol[sk+1, -1], 8)}; Solution number: {sk+2}/{N_sol} ({round(float(((sk+2)/N_sol))*100, 3)}%)")

        steplength[sk+1] = steplength_adaptation.exponential(number_iterations[sk+1], steplength[sk], steplength_parameters)

        stability_sol[sk+1], EV_monodromy_sol[sk+1,:,:] = stability.stability_analysis(Jacobian_sol[sk+1,:,:], n, ti_parameters)



    return X_sol, Y_sol, Jacobian_sol, number_iterations, steplength, p_sol, stability_sol, EV_monodromy_sol