"""   
    This file describes the step-length adaptation. 
    It determines the step-length for the next predictor step on the basis of the number of iterations of the last corrector.
    

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

##### trivial predictor, increase/decrease if iterations is larger/smaller than N_star #####
def trivial(number_iterations, steplength, steplength_parameters):
    alpha1 = 2
    alpha2 = 1

    if number_iterations > steplength_parameters.N_desired:
        steplength = (1/(alpha1*alpha2))*steplength
    elif number_iterations < steplength_parameters.N_desired:
        steplength = alpha1*steplength

    if abs(steplength) > steplength_parameters.max_stepsize:
        steplength = np.sign(steplength)*steplength_parameters.max_stepsize
    elif abs(steplength) < steplength_parameters.min_stepsize:
        steplength = np.sign(steplength)*steplength_parameters.min_stepsize

    return steplength


##### averaging to N_star (see Kerschen part2) #####
def N_star_avg(number_iterations, steplength, steplength_parameters):

    if number_iterations != steplength_parameters.N_desired and number_iterations != 0:
        steplength = steplength*(steplength_parameters.N_desired/number_iterations)
    elif number_iterations == 0:
        steplength = 2*steplength

    if abs(steplength) > steplength_parameters.max_stepsize:
        steplength = steplength_parameters.max_stepsize
    elif abs(steplength) < steplength_parameters.min_stepsize:
        steplength = steplength_parameters.min_stepsize

    return steplength


##### exponantial adaptation, increase/decrease by the a constant ti the power of  the differenc bewteen iterations and N_star #####
def exponential(number_iterations, steplength, steplength_parameters):
    alpha = 2

    diff = steplength_parameters.N_desired - number_iterations

    steplength = (alpha**diff)*steplength

    if abs(steplength) > steplength_parameters.max_stepsize:
        steplength = steplength_parameters.max_stepsize
    elif abs(steplength) < steplength_parameters.min_stepsize:
        steplength = steplength_parameters.min_stepsize

    return steplength