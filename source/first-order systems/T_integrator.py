"""   
    This file describes the time integration functions. 
    It defines all implemented time integration methods for natively first order systems.    

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
from state_space_functions import dynamics, symbolic_df_dx, symbolic_df_dOm



##### If you have a statespace function given(including precomputed derivatives)
def rungekutta4_symbolic(X, ti_parameters, n):

    Om = X[-1]

    # initialize variables
    y_start = X[:2*n]
    y = y_start

    dz_0 = np.eye(2)
    dOm = np.zeros([2*n])

    dtau = (2*np.pi)/(ti_parameters.Ntpp)

    # initialize ouputs
    R = np.zeros(2*n)
    Jac = np.zeros([2*n,2*n+1])
    monodromy = np.zeros([2*n, 2*n])
    Y = np.zeros([ti_parameters.Np*ti_parameters.Ntpp+1, 2*n])
    y_end = np.zeros(2*n)


    Y[0,:] = y_start

    for j in range(ti_parameters.Np*ti_parameters.Ntpp):

        args1 = [y, Om, j*dtau]
        k1 = dynamics(*args1).squeeze()
        d1 = np.dot(symbolic_df_dx(*args1), dz_0)
        O1 = np.dot(symbolic_df_dx(*args1), dOm) + symbolic_df_dOm(*args1).squeeze()

        args2 = [y+0.5*k1*dtau, Om, (j+0.5)*dtau]
        k2 = dynamics(*args2).squeeze()
        d2 = np.dot(symbolic_df_dx(*args2), dz_0+0.5*d1*dtau)
        O2 = np.dot(symbolic_df_dx(*args2), dOm + 0.5*O1*dtau) + symbolic_df_dOm(*args2).squeeze()

        args3 = [y+0.5*k2*dtau, Om, (j+0.5)*dtau]
        k3 = dynamics(*args3).squeeze()
        d3 = np.dot(symbolic_df_dx(*args3), dz_0+0.5*d2*dtau)
        O3 = np.dot(symbolic_df_dx(*args3), dOm + 0.5*O2*dtau) + symbolic_df_dOm(*args3).squeeze()

        args4 = [y+k3*dtau, Om, (j+1)*dtau]
        k4 = dynamics(*args4).squeeze()
        d4 = np.dot(symbolic_df_dx(*args4), dz_0+d3*dtau)
        O4 = np.dot(symbolic_df_dx(*args4), dOm + O3*dtau) + symbolic_df_dOm(*args4).squeeze()

        y = y + (k1 + 2*k2 + 2*k3 + k4)*(dtau/6)
        dz_0 = dz_0 + (d1 + 2*d2 + 2*d3 + d4)*(dtau/6)
        dOm = dOm + (O1 + 2*O2 + 2*O3 + O4)*(dtau/6)

        # safe results to Y
        Y[j+1, :] = y


    y_end[:] = y[:]

    monodromy = dz_0
    Jac[:2*n, :2*n] = monodromy - np.eye(2*n)
    Jac[:, 2*n] = dOm
    R = y_end - y_start



    return R, Jac, y_end, Y, monodromy



#### If you have a statespace function given(without precomputed derivatives)
def rungekutta4_statespace(X, n, ti_parameters):

    Om = Dual.new(value=X[-1], dq=np.zeros(n), dqdot=np.zeros(n), dOm=1)

    # initialize variables
    y_start = X[:2*n]
    y = np.array(np.zeros(2*n), Dual)
    for dk in range(n):
        diff = np.zeros(n)
        diff[dk] = 1
        y[dk] = Dual.new(value=X[dk], dq=diff, dqdot=np.zeros(n), dOm=0)
        y[dk+n] = Dual.new(value=X[dk+n], dq=np.zeros(n) , dqdot=diff, dOm=0)

    dtau = (2*np.pi)/(ti_parameters.Ntpp)

    # initialize ouputs
    R = np.zeros(2*n)
    Jac = np.zeros([2*n, 2*n+1])
    monodromy = np.zeros([2*n, 2*n])
    Y = np.zeros([ti_parameters.Np*ti_parameters.Ntpp+1, 2*n])
    y_end = np.zeros(2*n)


    Y[0,:] = y_start

    for j in range(ti_parameters.Np*ti_parameters.Ntpp):

        k1 = dynamics(y, Om, j*dtau).squeeze()
        k2 = dynamics(y+0.5*k1*dtau, Om, (j+0.5)*dtau).squeeze()
        k3 = dynamics(y+0.5*k2*dtau, Om, (j+0.5)*dtau).squeeze()
        k4 = dynamics(y+k3*dtau, Om, (j+1)*dtau).squeeze()

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