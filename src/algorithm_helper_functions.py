
import math
import numpy as np
from numpy import linalg as LA


def sort_fitness_and_get_xmean_zmean(input_fitness,input_mu, input_arx_list, input_arz_list, input_weights):
	sorted_arfitness = np.sort(input_fitness)
	sorted_arfitness_indexes = np.argsort(input_fitness)
	top_mu_indexes = sorted_arfitness_indexes[:input_mu]
	arx_list_subset = []
	arz_list_subset = []
	for ix in top_mu_indexes:
		arx_list_subset.append(input_arx_list[ix])
		arz_list_subset.append(input_arz_list[ix])
	arx_list_subset_reshaped = np.array(arx_list_subset).reshape(input_mu,7,-1).reshape(input_mu,7).transpose(1,0)
	xmean = np.matmul(arx_list_subset_reshaped,input_weights)

	arz_list_subset_reshaped = np.array(arz_list_subset).reshape(input_mu,7,-1).reshape(input_mu,7).transpose(1,0)
	zmean = np.matmul(arz_list_subset_reshaped,input_weights)
	return sorted_arfitness, sorted_arfitness_indexes, arx_list_subset_reshaped,arz_list_subset_reshaped, xmean, zmean


def adapt_covariance_matrix(inputc1, inputcmu, inputpc, inputhsig, inputcc, inputweights, inputarz_list_subset_reshaped, inputC,
							inputB, inputD):

    term1 = (1-inputc1-inputcmu) * inputC
    term2 = inputc1 * (np.matmul(inputpc, np.transpose(inputpc)) + (1-inputhsig) * inputcc * (2-inputcc) * inputC)
    term3_1 = np.matmul(np.matmul(np.matmul(inputB, inputD), inputarz_list_subset_reshaped),np.diagflat(inputweights))
    term3_2 = np.transpose(np.matmul(np.matmul(inputB, inputD), inputarz_list_subset_reshaped))
    term3 = np.matmul(term3_1, term3_2)
    C = term1 + term2 + term3
    return C

def compute_hsig(ip_cs, ip_counteval, ip_lambda_val, ip_ps, ip_chiN, ip_N):
    value_inside_sqrt = 1 - math.pow(1 - ip_cs, (2 * ip_counteval)/ip_lambda_val)
    numerator_1 = LA.norm(ip_ps)/value_inside_sqrt
    lhs = numerator_1/ip_chiN
    rhs = 1.4 + 2/(ip_N+1)
    if lhs < rhs:
        hsig = 1
    else:
        hsig = 0
    return hsig