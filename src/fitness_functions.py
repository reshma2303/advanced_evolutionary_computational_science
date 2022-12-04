

import numpy as np
import math
from pemfc_helper_functions import *

def felli_fitness_function(input_array):
    """
        Ref: https://www.mathworks.com/matlabcentral/fileexchange/37057-fast-adaptive-coordinate-descent-for-non-linear-optimization
            function f=felli(x)
        %  f = rand();
        %  return;
          N = size(x,1); if N < 2 error('dimension must be greater one'); end
          f=1e6.^((0:N-1)/(N-1)) * x.^2;
        end


        function f=felli(x)
        111 N = size(x,1); if N < 2 error(’dimension must be greater one’); end
        112 f=1e6.ˆ((0:N-1)/(N-1)) * x.ˆ2; % condition number 1e6
    :param input_array:
    :return:
    """
    input_len = input_array.shape[0] # Size of the x-dimension
    if input_len < 2:
        raise Exception('dimension must be greater one')
    x_square = input_array * input_array
    n_1 = input_len - 1
    n_1_array = [i*1.0/(n_1) for i in range(input_len)]
    n_1_array_1e6 = np.array([math.pow(1e6, each)for each in n_1_array])
    fitness_value = np.matmul(n_1_array_1e6 , x_square) 
    return fitness_value[0]

v_actual = compute_end_to_end_Vstack(ip_x1=-0.944957, ip_x2=0.00301801, ip_x3=7.401*math.pow(10,5), 
	ip_x4=-1.88 * math.pow(10,-4), ip_lambda=23, ip_Rc=0.0001, ip_B=0.02914489)

def pemfc_fitness_function(input_parameters_array): 	
	Vstack_model = compute_end_to_end_Vstack(input_parameters_array[0][0],
 		input_parameters_array[1][0], input_parameters_array[2][0], input_parameters_array[3][0],
 		input_parameters_array[4][0], input_parameters_array[5][0], input_parameters_array[6][0])
	diff = Vstack_model - v_actual
	diff_square = diff * diff
	return diff_square
