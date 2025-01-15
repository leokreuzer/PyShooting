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