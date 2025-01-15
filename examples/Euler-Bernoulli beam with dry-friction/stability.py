import numpy as np


def stability_analysis(Jacobian, n, ti_parameters):

    error = ti_parameters.error_const*(ti_parameters.Ntpp**(-ti_parameters.order_ti))

    monodromy = Jacobian[:2*n, :2*n] + np.eye(2*n)

    EV = np.linalg.eigvals(monodromy)
    largest_EV = np.sort(abs(EV))[-1]
    # calculation of stability
    if largest_EV > 1 + error:
        stability = False
    elif largest_EV < 1 - error:
        stability = True
    else:
        stability = np.NaN



    return stability, EV
