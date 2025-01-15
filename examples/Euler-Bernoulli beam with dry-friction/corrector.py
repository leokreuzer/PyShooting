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