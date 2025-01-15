"""   
    This file describes the stability analysis. 
    It determines the eigenvalues of the Jacobian leading to a classification of solutions into stable, instable and not-classifiable.
    

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
