import numpy as np
import matplotlib.pyplot as plt


X_sol = np.load('Final Results 0.1Krack regularization/test_BeamDryFriction_X_sol.npy')
Y_sol = np.load('Final Results 0.1Krack regularization/test_BeamDryFriction_Y_sol.npy')
N_sol = np.load('Final Results 0.1Krack regularization/test_BeamDryFriction_N_sol.npy')



frf1 = np.zeros([N_sol,2])
frf1[:,0] = X_sol[:,-1]
for jk in range(N_sol):
    frf1[jk,1] = max(abs(Y_sol[jk,:,0]))

frf2 = np.zeros([N_sol,2])
frf2[:,0] = X_sol[:,-1]
for jk in range(N_sol):
    frf2[jk,1] = max(abs(Y_sol[jk,:,1]))

frf3 = np.zeros([N_sol,2])
frf3[:,0] = X_sol[:,-1]
for jk in range(N_sol):
    frf3[jk,1] = max(abs(Y_sol[jk,:,2]))

frf4 = np.zeros([N_sol,2])
frf4[:,0] = X_sol[:,-1]
for jk in range(N_sol):
    frf4[jk,1] = max(abs(Y_sol[jk,:,3]))

plt.figure(1)
plt.plot(frf1[:,0], frf1[:,1], label='$q_1$')
plt.plot(frf2[:,0], frf2[:,1], label='$q_2$')
plt.plot(frf3[:,0], frf3[:,1], label='$q_3$')
plt.plot(frf4[:,0], frf4[:,1], label='$q_4$')

plt.xlabel('$\omega$', fontsize='x-large')
plt.ylabel('Amplitude', fontsize='x-large')
plt.legend(fontsize='x-large')
plt.xlim([25, 640])
plt.grid()
plt.show()



# t = np.linspace(0, 2*np.pi/50, 2**13, endpoint=True)
# plt.plot(t, Y_sol[0,:-1,:4])

# plt.grid()
# plt.show()