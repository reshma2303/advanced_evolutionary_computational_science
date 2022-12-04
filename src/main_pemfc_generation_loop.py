"""
% CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
3 % nonlinear function minimization.
"""

from params.pemfc_initializations import *
import math
import numpy as np
from numpy import linalg as LA
from fitness_functions import *
from algorithm_helper_functions import *

import warnings
warnings.filterwarnings('ignore')

if __name__=="__main__":

    counteval = 0

    iteration_i = 1
    while counteval <= stopeval:
        """
            % Generate and evaluate lambda offspring
            51 for k=1:lambda,
            52 arz(:,k) = randn(N,1); % standard normally distributed vector
            53 arx(:,k) = xmean + sigma * (B*D * arz(:,k)); % add mutation % Eq. 40
            54 arfitness(k) = feval(quadratic_fitness_function, arx(:,k)); % objective function call
            55 counteval = counteval+1;
        """
        arz_list = []
        arx_list = []
        arfitness = []
        for k in range(1, int(lambda_val)):
            # converted to -1,1 shape for vertical representation
            arz = np.random.normal(size=N).reshape(-1,1) # check if we have to change mean
            arz_list.append(arz)
            BD_matmul = np.matmul(B, D)
            BD_arz_mathmul = np.matmul(BD_matmul, arz)
            arx = xmean + sigma * BD_arz_mathmul
            arx_list.append(arx)
            current_fitness = pemfc_fitness_function(arx)
            arfitness.append(current_fitness)
            counteval += 1
        iteration_i += 1
        
        """
        % Sort by fitness and compute weighted mean into xmean
        59 [arfitness, arindex] = sort(arfitness); % minimization
        60 xmean = arx(:,arindex(1:mu))*weights; % recombination % Eq. 42
        61 zmean = arz(:,arindex(1:mu))*weights; % == Dˆ-1*B’*(xmean-xold)/sigma
        """
        sorted_arfitness, sorted_arfitness_indexes, arx_list_subset_reshaped,arz_list_subset_reshaped, xmean, zmean = sort_fitness_and_get_xmean_zmean(
            arfitness, mu, arx_list, arz_list, weights)

        """
        % Cumulation: Update evolution paths
        64 ps = (1-cs)*ps + (sqrt(cs*(2-cs)*mueff)) * (B * zmean); % Eq. 43
        65 hsig = norm(ps)/sqrt(1-(1-cs)ˆ(2*counteval/lambda))/chiN < 1.4+2/(N+1);
        66 pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean); % Eq. 45
        """
        B_zmean = np.matmul(B, zmean)
        value_in_sqrt = cs * (2-cs) * mueff
        ps = (1 - cs) * ps + math.sqrt(value_in_sqrt) * B_zmean
        hsig = compute_hsig(cs, counteval, lambda_val, ps, chiN, N)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2-cc) * mueff) * np.matmul(np.matmul(B, D),zmean)
        """
            Adapt covariance matrix C
            69 C = (1-c1-cmu) * C ... % regard old matrix % Eq. 47
            70 + c1 * (pc*pc’ ... % plus rank one update
            71 + (1-hsig) * cc*(2-cc) * C) ... % minor correction
            72 + cmu ... % plus rank mu update
            73 * (B*D*arz(:,arindex(1:mu))) ...
            74 * diag(weights) * (B*D*arz(:,arindex(1:mu)))’;
        """
        C = adapt_covariance_matrix(c1, cmu, pc, hsig, cc, weights, arz_list_subset_reshaped, C,
                                    B, D)
        """
            % Adapt step-size sigma
            77 sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1)); % Eq. 44
            78
            79 % Update B and D from C
            80 if counteval - eigeneval > lambda/(cone+cmu)/N/10 % to achieve O(Nˆ2)
            81 eigeneval = counteval;
            82 C=triu(C)+triu(C,1)’; % enforce symmetry
            83 [B,D] = eig(C); % eigen decomposition, B==normalized eigenvectors
            84 D = diag(sqrt(diag(D))); % D contains standard deviations now
        """
        sigma = sigma * math.exp(cs/damps) *(LA.norm(ps)/(chiN-1))
        """
            % Break, if fitness is good enough
            88 if arfitness(1) <= stopfitness
                    break;
        """
        if sorted_arfitness[0] <= stopfitness:
            print("sorted_arfitness[0] is less than stopfitness")
            break

        """
        % Escape flat fitness, or better terminate?
            93 if arfitness(1) == arfitness(ceil(0.7*lambda))
            94 sigma = sigma * exp(0.2+cs/damps);
            95 disp(’warning: flat fitness, consider reformulating the objective’);
            96 end
            97
            98 disp([num2str(counteval) ’: ’ num2str(arfitness(1))]);
        """
        if sorted_arfitness[0] <= sorted_arfitness[math.ceil(0.7*lambda_val)]:
            sigma = sigma * math.exp((0.2 + (cs/damps)))
            #print("Warning: Flat fitness, consider reformulating the objective")
        #print(f"Best fitness value for the epoch {counteval} is: {sorted_arfitness[0]}")
        # end while loop
    # -------------------- Final Message ---------------------------------
    """
    disp([num2str(counteval) ’: ’ num2str(arfitness(1))]);
    105 xmin = arx(:, arindex(1)); % Return best point of last generation.
    106 % Notice that xmean is expected to be even
    107 % better.
    """
    #print(f"Best fitness value for the epoch {counteval} is: {sorted_arfitness[0]}")
    xmin = arx_list_subset_reshaped[0]

