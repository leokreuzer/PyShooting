"""   
    This file describes the implemented post processing functions.
    It allows for visualization and saving of the computed data.

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
import matplotlib.pyplot as plt


def plotting(X_sol, Y_sol, N_sol, ti_parameters, n, solver_parameters, Jacobian_sol, steplength_parameters):
    frf = np.zeros([n,N_sol,2])

    for jk in range(N_sol):
        for dofs in range(n):
            displacement = Y_sol[jk, :-1, dofs]
            fourier_transform = abs((np.fft.fft(displacement))/ti_parameters.Ntpp)[0:int(ti_parameters.Ntpp/2)+1]
            fourier_transform[1:-1] = 2*fourier_transform[1:-1]
            frf[dofs, jk, 1] = fourier_transform[1]
            frf[dofs, jk, 0] = X_sol[jk, -1]

    plt.figure(1)
    for ik in range(n):
        plt.plot(frf[ik,:,0], frf[ik,:,1])
    plt.xlabel('$\omega$')
    plt.ylabel('$2|Q_1|$')
    plt.show()

# def plotting_fep(X_sol, Y_sol, N_sol, ti_parameters, n, system, Fex, solver_parameters, Jacobian_sol, steplength_parameters):
#     energy = np.zeros(N_sol)
#     M = system.M
#     K = system.K
#     gamma = 0.1

#     for jk in range(N_sol):
#         energy[jk] = 0.5*np.dot(X_sol[jk,n:2*n].T*X_sol[jk, 2*n],np.dot(M, X_sol[jk,n:2*n]*X_sol[jk, 2*n])) + 0.5*np.dot(X_sol[jk,0:n].T,np.dot(K, X_sol[jk,0:n])) + 0.25*gamma*X_sol[jk,0]**4

#     plt.plot(energy[:], X_sol[:,2*n]/(2*np.pi), color='b')


#     plt.xlabel('$Energy~(log scale)$', fontsize='xx-large')
#     plt.ylabel('$Frequency~[Hz]$', fontsize='xx-large')
#     plt.xscale('log')
#     plt.grid()
#     plt.show()


def saving(X_sol, Y_sol, N_sol, ti_parameters, n, solver_parameters, Jacobian_sol, steplength_parameters):

    np.save('test_X_sol', X_sol)
    np.save('test_Y_sol', Y_sol)
    np.save('test_N_sol', N_sol)
    np.save('test_ti_parameters', ti_parameters)
    np.save('test_solver_parameters', solver_parameters)
    np.save('test_Jacobian_sol', Jacobian_sol)
    np.save('test_steplength_parameters', steplength_parameters)