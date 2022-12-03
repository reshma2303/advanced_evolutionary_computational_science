from initialization import *
import math
import numpy as np
from numpy import linalg as LA

counteval = 0

def strfitnessfct(input_array):
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


iteration_i = 1
while counteval <= eigenval:
    """
        % Generate and evaluate lambda offspring
        51 for k=1:lambda,
        52 arz(:,k) = randn(N,1); % standard normally distributed vector
        53 arx(:,k) = xmean + sigma * (B*D * arz(:,k)); % add mutation % Eq. 40
        54 arfitness(k) = feval(strfitnessfct, arx(:,k)); % objective function call
        55 counteval = counteval+1;
        56 end
        

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
        current_fitness = strfitnessfct(arx)
        arfitness.append(current_fitness)
        counteval += 1
    print(f"All Fitnesses for iteration {iteration_i}")
    iteration_i += 1
    print(arfitness)
    


    """
    % Sort by fitness and compute weighted mean into xmean
    59 [arfitness, arindex] = sort(arfitness); % minimization
    60 xmean = arx(:,arindex(1:mu))*weights; % recombination % Eq. 42
    61 zmean = arz(:,arindex(1:mu))*weights; % == Dˆ-1*B’*(xmean-xold)/sigma
    """
    sorted_arfitness = np.sort(arfitness)
    sorted_arfitness_indexes = np.argsort(arfitness)
    top_mu_indexes = sorted_arfitness_indexes[:mu]
    arx_list_subset = []
    arz_list_subset = []
    for ix in top_mu_indexes:
        arx_list_subset.append(arx_list[ix])
        arz_list_subset.append(arz_list[ix])

    arx_list_subset_reshaped = np.array(arx_list_subset).reshape(5,10,-1).reshape(5,10).transpose(1,0)
    xmean = np.matmul(arx_list_subset_reshaped,weights)

    arz_list_subset_reshaped = np.array(arz_list_subset).reshape(5,10,-1).reshape(5,10).transpose(1,0)
    zmean = np.matmul(arz_list_subset_reshaped,weights)
    


    

    """
    % Cumulation: Update evolution paths
    64 ps = (1-cs)*ps + (sqrt(cs*(2-cs)*mueff)) * (B * zmean); % Eq. 43
    65 hsig = norm(ps)/sqrt(1-(1-cs)ˆ(2*counteval/lambda))/chiN < 1.4+2/(N+1);
    66 pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (B*D*zmean); % Eq. 45
    """
    B_zmean = np.matmul(B, zmean)
    value_in_sqrt = cs * (2-cs) * mueff
    ps = (1 - cs) * ps + math.sqrt(value_in_sqrt) * B_zmean
    #print(LA.norm(ps))
    #hsig =    1 - (1-cs)
    value_inside_sqrt = 1 - math.pow(1 - cs, (2 * counteval)/lambda_val)
    numerator_1 = LA.norm(ps)/value_inside_sqrt
    lhs = numerator_1/chiN
    rhs = 1.4 + 2/(N+1)
    if lhs < rhs:
        hsig = 1
    else:
        hsig = 0
    pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2-cc) * mueff) * np.matmul(np.matmul(B, D),zmean)

    """
        Adapt covariance matrix C
        69 C = (1-c1-cmu) * C ... % regard old matrix % Eq. 47
        70 + c1 * (pc*pc’ ... % plus rank one update
        71 + (1-hsig) * cc*(2-cc) * C) ... % minor correction
        72 + cmu ... % plus rank mu update
        73 * (B*D*arz(:,arindex(1:mu))) ...
        74 * diag(weights) * (B*D*arz(:,arindex(1:mu)))’;

        cmu * artmp * diag(weights) * artmp'
    """
    term1 = (1-c1-cmu) * C 
    term2 = c1 * (np.matmul(pc, np.transpose(pc)) + (1-hsig) * cc * (2-cc) * C)
    term3_1 = np.matmul(np.matmul(np.matmul(B, D), arz_list_subset_reshaped),np.diagflat(weights))
    term3_2 = np.transpose(np.matmul(np.matmul(B, D), arz_list_subset_reshaped))
    term3 = np.matmul(term3_1, term3_2)

    C = term1 + term2 + term3


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
        85 end
    """
    print(cs)
    print(damps)
    #sigma = sigma * *(LA.norm(ps)/(chiN-1))


    """
        % Break, if fitness is good enough
        88 if arfitness(1) <= stopfitness
        89 break;
        90 end
    """


    """
    % Escape flat fitness, or better terminate?
        93 if arfitness(1) == arfitness(ceil(0.7*lambda))
        94 sigma = sigma * exp(0.2+cs/damps);
        95 disp(’warning: flat fitness, consider reformulating the objective’);
        96 end
        97
        98 disp([num2str(counteval) ’: ’ num2str(arfitness(1))]);
    """
    # end while loop

# -------------------- Final Message ---------------------------------


"""
disp([num2str(counteval) ’: ’ num2str(arfitness(1))]);
105 xmin = arx(:, arindex(1)); % Return best point of last generation.
106 % Notice that xmean is expected to be even
107 % better.
"""


