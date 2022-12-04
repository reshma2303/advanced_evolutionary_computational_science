

import random
import math
import numpy as np

"""
PEMFC Parameters
"""
strfitnessfct = "felli" # name of objective/fitness function
N = 7 # number of objective variables/problem dimension
Ns = 24 # number of cells in the stack
T = 283 # stack temperature, K
RHa = 1 # relative humidity of vapor in anode
RHc = 1 # relative humidity of vapor in cathode
pa_value = 3
pc_value = 5
i_value = 0.25
A_value  = 27
Rm =  10 # equivalent resistance of membrane, U
Rc =  10 # resistance, U
B = 0.08 # concentration loss constant, V
"""
iden current density, A/cm2
ilimit,den limiting current density, A
"""
iden = 80
ilimit_den =  86


"""
CMA-ES algorithm parameters
"""

#xmean = rand(N,1) # objective variables initial point
xmean = np.random.uniform(size=N).reshape(-1,1) # reshaping to make it vertical
sigma = 0.5 # coordinate wise standard deviation (step-size)
stopfitness = 1e-10 # stop if fitness < stopfitness (minimization)
stopeval = 1e3*(N ^ 2) #stop after stopeval number of function evaluations


"""
% Strategy parameter setting: Selection
21 lambda = 4+floor(3*log(N)); % population size, offspring number
22 mu = lambda/2; % lambda=12; mu=3; weights = ones(mu,1); would be (3_I,12)-ES
23 weights = log(mu+1/2)-log(1:mu)’; % muXone recombination weights
24 mu = floor(mu); % number of parents/points for recombination
25 weights = weights/sum(weights); % normalize recombination weights array
26 mueff=sum(weights)ˆ2/sum(weights.ˆ2); % variance-effective size of mu
"""
lambda_val = 4+math.floor(3*math.log(N)) # population size, offspring number
mu = lambda_val/2 # lambda=12; mu=3; weights = ones(mu,1); would be (3_I,12)-ES

exp1 = math.log(mu+1.0/2)
matrix = [math.log(each) for each in range(1,int(mu + 1))]
np_matrix = np.array(matrix)
weights = np_matrix.T # % muXone recombination weights 1D array transpose no effect in python unlike matlab
#print(np.transpose(weights))
#print(weights)
weights = [exp1-each for each in weights]

weights = np.array(weights).reshape(-1,1)

mu = math.floor(mu) # number of parents/points for recombination
weights = weights/sum(weights)  #normalize recombination weights array
weights_squares = [each * each for each in weights]
numerator = math.pow(sum(weights), 2)
mueff=numerator/sum(weights_squares)  #variance-effective size of mu
mueff = mueff[0]


"""
% Strategy parameter setting: Adaptation
2cc = (4+mueff/N) / (N+4 + 2*mueff/N); % time constant for cumulation for C
30 cs = (mueff+2)/(N+mueff+5); % t-const for cumulation for sigma control
31 c1 = 2 / ((N+1.3)62+mueff); % learning rate for rank-one update of C
32 cmu = 2 * (mueff-2+1/mueff) / ((N+2)^2+2*mueff/2); % and for rank-mu update
33 damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma
"""

cc = (4 + mueff/N) / (N + 4 + 2 * mueff/N)
cs = (mueff + 2)/(N + mueff + 5)
c1 = 2/ (math.pow((N + 1.3), 2) + mueff)
cmu = 2 * (mueff - 2 + 1 /mueff)
damps = 1 + 2 * max(0, math.sqrt ((mueff - 1)/ (N + 1)) - 1) + cs



"""
% Initialize dynamic (internal) strategy parameters and constants
37 pc = zeros(N,1); ps = zeros(N,1); % evolution paths for C and sigma
38 B = eye(N); % B defines the coordinate system
39 D = eye(N); % diagonal matrix D defines the scaling
40 C = B*D*(B*D)'; % covariance matrix
41 eigeneval = 0; % B and D updated at counteval == 0
42 chiN=N^0.5*(1-1/(4*N)+1/(21*N^2)); % expectation of
43 % ||N(0,I)|| == norm(randn(N,1))
"""

pc = np.zeros(N).reshape(-1,1)
ps = np.zeros(N).reshape(-1,1)
B = np.eye(N)
D = np.eye(N)
BD_T = np.array(B * D).T
C = np.matmul(np.matmul(B , D) , BD_T) # TODO: Check if transpose works

eigenval = 0

chiN = math.pow(N, 0.5) * (1 - 1 / (4 * N) + 1 / (21 * math.pow(N,2)))



def print_user_defined_parameters():
    print("strfitnessfct: ", strfitnessfct)
    print("N: ", N)

    print("xmean:\n")
    print(xmean)
    print("\n")

    print("sigma: ", sigma)
    print("stopfitness: ", stopfitness)
    print("stopeval: ", stopeval)


def print_strategy_parameter_setting():
    print("lambda_val: ", lambda_val)
    print("exp1: ", exp1)

    print("weights: ", weights)
    print("mu: ", mu)
    print("mueff: ", mueff)

def print_adapatation_parameters():
    print("cc: ", cc)
    print("cs: ", cs)
    print("c1: ", c1)
    print("cmu: ", cmu)
    print("damps: ", damps)    

def print_dynamic_strategy_constants():
    print("pc: ", pc)
    print("B: ", B)
    print("D: ", D)
    print("BD_T: ", BD_T)
    print("C: ", C)
    print("eigenval: ", eigenval)
    print("chiN: ", chiN)

if __name__=="__main__":
    #print_user_defined_parameters()
    #print_strategy_parameter_setting()
    #print_adapatation_parameters()
    #print_dynamic_strategy_constants()
    print("done")
