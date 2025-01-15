import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plotting(X_sol, Y_sol, N_sol, ti_parameters, n, system, Fex, solver_parameters, Jacobian_sol, steplength_parameters):
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


def plotting_amp(X_sol, Y_sol, N_sol, ti_parameters, n, system, Fex, solver_parameters, Jacobian_sol, steplength_parameters):
    frf1 = np.zeros([N_sol,2])
    frf1[:,0] = X_sol[:,-1]
    for jk in range(N_sol):
        frf1[jk,1] = max(Y_sol[jk,:,0])

    frf2 = np.zeros([N_sol,2])
    frf2[:,0] = X_sol[:,-1]
    for jk in range(N_sol):
        frf2[jk,1] = max(Y_sol[jk,:,1])

    frf3 = np.zeros([N_sol,2])
    frf3[:,0] = X_sol[:,-1]
    for jk in range(N_sol):
        frf3[jk,1] = max(Y_sol[jk,:,2])

    frf4 = np.zeros([N_sol,2])
    frf4[:,0] = X_sol[:,-1]
    for jk in range(N_sol):
        frf4[jk,1] = max(Y_sol[jk,:,3])

    plt.figure(1)
    plt.plot(frf1[:,0], frf1[:,1], label='$q_1$')
    plt.plot(frf2[:,0], frf2[:,1], label='$q_2$')
    plt.plot(frf3[:,0], frf3[:,1], label='$q_3$')
    plt.plot(frf4[:,0], frf4[:,1], label='$q_4$')

    plt.xlabel('$\omega$')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.yscale('log')
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


def saving(X_sol, Y_sol, N_sol, ti_parameters, n, system, Fex, solver_parameters, Jacobian_sol, steplength_parameters):

    np.save('test_BeamDryFriction_X_sol', X_sol)
    np.save('test_BeamDryFriction_Y_sol', Y_sol)
    np.save('test_BeamDryFriction_N_sol', N_sol)
    np.save('test_BeamDryFriction_ti_parameters', ti_parameters)
    np.save('test_BeamDryFriction_solver_parameters', solver_parameters)
    np.save('test_BeamDryFriction_Jacobian_sol', Jacobian_sol)
    np.save('test_BeamDryFriction_steplength_parameters', steplength_parameters)