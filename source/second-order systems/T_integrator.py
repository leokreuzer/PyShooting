"""   
    This file describes the time integration functions. 
    It defines all implemented time integration methods for natively second order systems.    

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
from scipy.linalg import inv
from Mechanical_System_class import mechanical_system_class
from dual_numbers_dynamics import Dual

def Newmark(X, system, ti_parameters, fex_values):
    assert isinstance(X, np.ndarray), "X must be a np.ndarray"
    assert isinstance(system, mechanical_system_class), "system need to be an object of the class mechanical_system defined in Mechanical_System.py"
    assert isinstance(ti_parameters.Ntpp, int), "N_tpp must be an integer"
    assert isinstance(ti_parameters.Np, int), "N_p must be an integer"
    assert (fex_values.shape[0] == (ti_parameters.Np*ti_parameters.Ntpp) + 1), 'Fex needs to be precomputed at Np*Ntpp timelevels \n go to solve_andcontinue to change it'

    
    # initialize variables/functions for either case
    n = system.n
    assert isinstance(n, int), "n must be an integer"

    if n == 1:
        Om = Dual.new(value=X[-1], dq=np.zeros(n), dqdot=np.zeros(n), dOm=1)

        # definition of necessary functions  
        def dynamics(q, qdot, acc, tau, Om):
            r = M*acc*(Om**2) + C*qdot*Om + K*q + system.f_nl(q, qdot*Om) - fex_values[tau]
            return r
        
        def predictor(q_new, q, qdot, acc):
            acc_new = (1/(beta*(dtau**2)))*((-1)*(q + dtau*qdot + (dtau**2)*(0.5-beta)*acc) + q_new)
            qdot_new = qdot + (1-gamma)*dtau*acc + gamma*dtau*acc_new
            return acc_new, qdot_new
        
        def time_integration_residue(q_new, q, qdot, acc, tau, Om):
            acc_new, qdot_new = predictor(q_new, q, qdot, acc)
            return dynamics(q_new, qdot_new, acc_new, tau, Om), acc_new, qdot_new


        # initialize variables
        y_start = X[:2*n]
        q = Dual.new(value=X[:n], dq=np.eye(n), dqdot=np.zeros((n,n)), dOm=np.zeros((n)))
        qdot = Dual.new(value=X[n:2*n], dq=np.zeros((n,n)) , dqdot=np.eye(n), dOm=np.zeros((n)))
        M = system.M
        C = system.C
        K = system.K
        dtau = (2*np.pi)/(ti_parameters.Ntpp)
        gamma = 0.5
        beta = 1/6

        # constant part of the Newmark-Iteration-Matrix
        S = K + (gamma/(beta*dtau))*C*Om.value + (1/(beta*(dtau**2)))*M*(Om.value**2)

        # initialize ouputs
        R = np.zeros(2*n)
        Jac = np.zeros([2*n, 2*n+1])
        monodromy = np.zeros([2*n, 2*n])
        Y = np.zeros([ti_parameters.Np*ti_parameters.Ntpp+1, 2*n])
        y_end = np.zeros(2*n)

        Y[0,:] = y_start
        # calculate initial acceleration
        acc = (1/M)*(((-1)*C*qdot*Om - K*q - system.f_nl(q, qdot*Om) + fex_values[0])*(1/(Om**2)))

        # start time-integration
        for j in range(ti_parameters.Np*ti_parameters.Ntpp):
            # prediction of values at next time-step
            q_next_value = q.value + dtau*qdot.value + 0.5*(dtau**2)*acc.value

            # solve for displacement at next timestep
            for ik in range(ti_parameters.max_iter_NR):
                r, acc_next_value, qdot_next_value = time_integration_residue(q_next_value, q.value, qdot.value, acc.value, (j+1), Om.value)

                if ik == 0:
                    r_initial_norm = np.linalg.norm(r)
                
                if np.linalg.norm(r) < ti_parameters.abs_ti_tolerance or np.linalg.norm(r) < ti_parameters.rel_ti_tolerance*r_initial_norm:
                    if ik == 0:
                        q_iter = Dual.new(value=q_next_value, dq=np.eye(n), dqdot=np.zeros((n,n)), dOm=np.zeros((n)))
                        qdot_iter = Dual.new(value=qdot_next_value, dq=np.zeros((n,n)), dqdot=np.eye(n), dOm=np.zeros((n)))
                        f_nl_qiter = system.f_nl(q_iter, qdot_iter*Om.value)
                        S_full = (S + float(f_nl_qiter.dq) + float(f_nl_qiter.dqdot)*(gamma/(beta*dtau))*Om.value)

                    break
                
                q_iter = Dual.new(value=q_next_value, dq=np.eye(n), dqdot=np.zeros((n,n)), dOm=np.zeros((n)))
                qdot_iter = Dual.new(value=qdot_next_value, dq=np.zeros((n,n)), dqdot=np.eye(n), dOm=np.zeros((n)))
                # calculating the update for the displacement
                f_nl_qiter = system.f_nl(q_iter, qdot_iter*Om.value)
                S_full = (S + float(f_nl_qiter.dq) + float(f_nl_qiter.dqdot)*(gamma/(beta*dtau))*Om.value)
                q_next_value -= r/S_full

            if np.linalg.norm(r) > ti_parameters.abs_ti_tolerance and np.linalg.norm(r) > ti_parameters.rel_ti_tolerance*r_initial_norm:
                print('No convergence in the time-integration')
                print(X[:])
                exit()


            # calculate q_next including the differentials with respect to q0, qdot0, Om
            ti_residual, acc_trash, qdot_trash = time_integration_residue(q_next_value, q, qdot, acc, (j+1), Om)
            q_next = (-1)*(ti_residual/S_full)
            q_next.value = q_next_value

            # calculate qdot_next and acc_next including the differentials with respect to q0, qdot0, Om
            acc_next, qdot_next = predictor(q_next, q, qdot, acc)
            
            # safe results to Y
            Y[j+1,:n] = q_next.value
            Y[j+1,n:2*n] = qdot_next.value
            # update current values
            q = q_next
            qdot = qdot_next
            acc = acc_next


        # calculate the outputs
        y_end[:n] = q.value
        y_end[n:2*n] = qdot.value
        R = y_end - y_start
        monodromy[:n,:n] = q.dq
        monodromy[:n,n:2*n] = q.dqdot
        monodromy[n:2*n,:n] = qdot.dq
        monodromy[n:2*n,n:2*n] = qdot.dqdot

        Jac[:n,:n] = q.dq - np.eye(n)
        Jac[:n,n:2*n] = q.dqdot
        Jac[:n,2*n] = q.dOm
        Jac[n:2*n,:n] = qdot.dq
        Jac[n:2*n,n:2*n] = qdot.dqdot - np.eye(n)
        Jac[n:2*n,2*n] = qdot.dOm

        return R, Jac, y_end, Y, monodromy
    

    elif n > 1:
        Om = Dual.new(value=X[-1], dq=np.zeros(n), dqdot=np.zeros(n), dOm=1)

        # definition of necessary functions  
        def dynamics(q, qdot, acc, tau, Om):
            r = Dual.dot(M,acc)*(Om**2) + Dual.dot(C,qdot)*Om + Dual.dot(K,q) + system.f_nl(q, qdot*Om) - fex_values[tau]
            return r
        
        def predictor(q_new, q, qdot, acc):
            acc_new = (1/(beta*(dtau**2)))*((-1)*(q + dtau*qdot + (dtau**2)*(0.5-beta)*acc) + q_new)
            qdot_new = qdot + (1-gamma)*dtau*acc + gamma*dtau*acc_new
            return acc_new, qdot_new
        
        def time_integration_residue(q_new, q, qdot, acc, tau, Om):
            acc_new, qdot_new = predictor(q_new, q, qdot, acc)
            return dynamics(q_new, qdot_new, acc_new, tau, Om), acc_new, qdot_new


        # initialize variables
        y_start = X[:2*n]
        q = Dual.new(value=X[:n], dq=np.eye(n), dqdot=np.zeros((n,n)), dOm=np.zeros((n)))
        qdot = Dual.new(value=X[n:2*n], dq=np.zeros((n,n)) , dqdot=np.eye(n), dOm=np.zeros((n)))
        M = system.M
        C = system.C
        K = system.K
        dtau = (2*np.pi)/(ti_parameters.Ntpp)
        gamma = 0.5
        beta = 1/6

        # constant part of the Newmark-Iteration-Matrix
        S = K + (gamma/(beta*dtau))*C*Om.value + (1/(beta*(dtau**2)))*M*(Om.value**2)

        # initialize ouputs
        R = np.zeros(2*n)
        Jac = np.zeros([2*n, 2*n+1])
        monodromy = np.zeros([2*n, 2*n])
        Y = np.zeros([ti_parameters.Np*ti_parameters.Ntpp+1, 2*n])
        y_end = np.zeros(2*n)

        Y[0,:] = y_start
        # calculate initial acceleration
        acc = Dual.dot(np.linalg.inv(M), (((-1)*Dual.dot(C, qdot)*Om - Dual.dot(K,q) - system.f_nl(q, qdot*Om) + fex_values[0]) * (1/(Om**2))))

        # start time-integration
        for j in range(ti_parameters.Np*ti_parameters.Ntpp):
            # prediction of values at next time-step
            q_next_value = q.value + dtau*qdot.value + 0.5*(dtau**2)*acc.value

            # solve for displacement at next timestep
            for ik in range(ti_parameters.max_iter_NR):
                r, acc_next_value, qdot_next_value = time_integration_residue(q_next_value, q.value, qdot.value, acc.value, (j+1), Om.value)

                if ik == 0:
                    r_initial_norm = np.linalg.norm(r)
                
                if np.linalg.norm(r) < ti_parameters.abs_ti_tolerance or np.linalg.norm(r) < ti_parameters.rel_ti_tolerance*r_initial_norm:
                    q_iter = Dual.new(value=q_next_value, dq=np.eye(n), dqdot=np.zeros((n,n)), dOm=np.zeros((n)))
                    qdot_iter = Dual.new(value=qdot_next_value, dq=np.zeros((n,n)), dqdot=np.eye(n), dOm=np.zeros((n)))
                    f_nl_qiter = system.f_nl(q_iter, qdot_iter*Om.value)
                    S_full = S + f_nl_qiter.dq + f_nl_qiter.dqdot*(gamma/(beta*dtau))*Om.value
                    break
                
                q_iter = Dual.new(value=q_next_value, dq=np.eye(n), dqdot=np.zeros((n,n)), dOm=np.zeros((n)))
                qdot_iter = Dual.new(value=qdot_next_value, dq=np.zeros((n,n)), dqdot=np.eye(n), dOm=np.zeros((n)))
                # calculating the update for the displacement
                f_nl_qiter = system.f_nl(q_iter, qdot_iter*Om.value)
                S_full = S + f_nl_qiter.dq + f_nl_qiter.dqdot*(gamma/(beta*dtau))*Om.value
                q_next_value -= np.linalg.solve(S_full, r)

            if np.linalg.norm(r) > ti_parameters.abs_ti_tolerance and np.linalg.norm(r) > ti_parameters.rel_ti_tolerance*r_initial_norm:
                print('No convergence in the time-integration')
                print(X[:])
                exit()


            # calculate q_next including the differentials with respect to q0, qdot0, Om
            ti_residual, acc_trash, qdot_trash = time_integration_residue(q_next_value, q, qdot, acc, (j+1), Om)
            q_next = (-1)*Dual.dot(np.linalg.inv(S_full), ti_residual)
            q_next.value = q_next_value

            # calculate qdot_next and acc_next including the differentials with respect to q0, qdot0, Om
            acc_next, qdot_next = predictor(q_next, q, qdot, acc)
            
            # safe results to Y
            Y[j+1,:n] = q_next.value
            Y[j+1,n:2*n] = qdot_next.value
            # update current values
            q = q_next
            qdot = qdot_next
            acc = acc_next


        # calculate the outputs
        y_end[:n] = q.value
        y_end[n:2*n] = qdot.value
        R = y_end - y_start
        monodromy[:n,:n] = q.dq
        monodromy[:n,n:2*n] = q.dqdot
        monodromy[n:2*n,:n] = qdot.dq
        monodromy[n:2*n,n:2*n] = qdot.dqdot

        Jac[:n,:n] = q.dq - np.eye(n)
        Jac[:n,n:2*n] = q.dqdot
        Jac[:n,2*n] = q.dOm
        Jac[n:2*n,:n] = qdot.dq
        Jac[n:2*n,n:2*n] = qdot.dqdot - np.eye(n)
        Jac[n:2*n,2*n] = qdot.dOm

        return R, Jac, y_end, Y, monodromy
    

def rungekutta4(X, system, ti_parameters, fex_values):
    assert (fex_values.shape[0] == 2*(ti_parameters.Np*ti_parameters.Ntpp) + 1), 'Fex needs to be precomputed at 2*Np*Ntpp timelevels \n go to solve_andcontinue to change it'
    n = system.n

    Om = Dual.new(value=X[-1], dq=np.zeros(n), dqdot=np.zeros(n), dOm=1)

    # initialize variables
    y_start = X[:2*n]
    y = np.array(np.zeros(2*n), Dual)
    for dk in range(n):
        diff = np.zeros(n)
        diff[dk] = 1
        y[dk] = Dual.new(value=X[dk], dq=diff, dqdot=np.zeros(n), dOm=0)
        y[dk+n] = Dual.new(value=X[dk+n], dq=np.zeros(n) , dqdot=diff, dOm=0)
    M = system.M
    C = system.C
    K = system.K
    dtau = (2*np.pi)/(ti_parameters.Ntpp)

    # initialize ouputs
    R = np.zeros(2*n)
    Jac = np.zeros([2*n, 2*n+1])
    monodromy = np.zeros([2*n, 2*n])
    Y = np.zeros([ti_parameters.Np*ti_parameters.Ntpp+1, 2*n])
    y_end = np.zeros(2*n)

    if n == 1:
        statespace_matrix = np.array([[0, 1], [(-1)*K/(M*(Om**2)), (-1)*C/(M*Om)]])
        def phi(tau, y):
            r = Dual.dot(statespace_matrix, y) - np.array([0, system.f_nl(y[:n], y[n:2*n]*Om) - fex_values[int(2*tau)]])*(1/(M*Om**2))
            return r
    elif n>1:
        statespace_matrix = np.vstack([np.hstack([np.zeros([n,n]), np.eye(n)]), np.hstack([(-1)*np.dot(inv(M), K)*(1/(Om**2)), (-1)*np.dot(inv(M), C)*(1/Om)])])
        def phi(tau, y):
            r = Dual.dot(statespace_matrix, y) - np.hstack([np.zeros(n), np.dot(inv(M) ,system.f_nl(y[:n], y[n:2*n]*Om) - fex_values[int(2*tau),:])*(1/(Om**2))])
            return r


    Y[0,:] = y_start

    for j in range(ti_parameters.Np*ti_parameters.Ntpp):

        k1 = phi(j, y)
        k2 = phi((j+0.5), y+0.5*k1*dtau)
        k3 = phi((j+0.5), y+0.5*k2*dtau)
        k4 = phi((j+1), y+k3*dtau)

        y_next = y + (k1 + 2*k2 + 2*k3 + k4)*(dtau/6)
            
        # safe results to Y
        for dk in range(n):
            Y[j+1, dk] = y_next[dk].value
            Y[j+1, dk+n] = y_next[dk+n].value
        # update current values
        y = y_next

    for dk in range(n):
        y_end[dk] = y[dk].value
        y_end[dk+n] = y[dk+n].value

        monodromy[dk, 0:n] = y[dk].dq
        monodromy[dk, n:2*n] = y[dk].dqdot
        monodromy[dk+n, 0:n] = y[dk+n].dq
        monodromy[dk+n, n:2*n] = y[dk+n].dqdot

        Jac[dk, 2*n] = y[dk].dOm
        Jac[dk+n, 2*n] = y[dk+n].dOm

    Jac[:2*n, :2*n] = monodromy - np.eye(2*n)
    R = y_end - y_start

    return R, Jac, y_end, Y, monodromy