

import math
from params.pemfc_initializations import Ns, T, pa_value, i_value, A_value, RHa, RHc, pc_value, Rm, Rc, B, iden, ilimit_den

def compute_ph20_sat(ip_T):
	term1 = 2.95 * (math.pow(10, -2) * (ip_T - 273.15))
	term2 = 9.19 * math.pow(10, -5) * (ip_T - 273.15) * (ip_T - 273.15)
	term3 = 1.44 * math.pow(10, -7) * (ip_T - 273.15) * (ip_T - 273.15)  * (ip_T - 273.15)
	return term1 - term2 + term3 - 2.18

def compute_ph2(ip_RHa, ip_ph20_sat, ip_pa, ip_i, ip_A, ip_T):
	term1 = 0.5 * ip_RHa * ip_ph20_sat
	denominator = ((ip_RHa * ip_ph20_sat)/ip_pa) * math.exp((1.635 * (ip_i/ip_A)) * math.pow(ip_T, 1.334))
	return term1  * (1 /denominator - 1)

def compute_po2(ip_RHc, ip_ph20_sat, ip_pc, ip_i, ip_A, ip_T):
	term1 = ip_RHc * ip_ph20_sat
	denominator = ((ip_RHc * ip_ph20_sat)/ip_pc) * math.exp((4.192 * ip_i/ip_A)/math.pow(ip_T,1.334))
	return term1 * (1 / denominator - 1)

def compute_co2(ip_po2, ip_T):
	return ip_po2/(5.08 * math.pow(10, 6) * math.exp((-498/T)))

def compute_enernst(ip_T, ip_Ph2, ip_Po2):
	return 1.229 - (0.85 * (0.001 * (ip_T - 298.15))) + (4.3085 * (math.pow(10, -5) * ip_T * math.log(ip_Ph2 * math.sqrt(ip_Po2))))

def compute_thetha_act(ip_x1, ip_x2, ip_x3, ip_x4, ip_T, ip_co2, ip_i):
	return -1 * (ip_x1 + ip_x2 * ip_T + ip_x3 * ip_T * math.log(ip_co2) + ip_x4 * ip_T * math.log(ip_i))

def compute_thetha_ohm(ip_i, ip_Rm, ip_Rc):
	return ip_i * (ip_Rm + ip_Rm)

def compute_thetha_conc(ip_B, ip_iden, ip_ilimit_den):
	"""
		B: a parametric coefficient (V),
		iden is the current density(iden = i/A)
		ilimit;den the limiting current density.
	"""
	return -1 * math.log(1- (ip_iden/ip_ilimit_den))

def compute_Vcell(ip_enernst, ip_theta_act, ip_theta_ohm, ip_thetha_conc):
	return ip_enernst - ip_theta_act - ip_theta_ohm - ip_thetha_conc

def compute_Vstack(ip_nS, ip_vCell):
	"""
		Vstack = NsVcell:
	"""
	return ip_nS * ip_vCell

def compute_end_to_end_Vstack(ip_x1, ip_x2, ip_x3, ip_x4, ip_lambda, ip_Rc, ip_B):


	ph20_sat = compute_ph20_sat(T)
	ph2 = compute_ph2(RHa, ph20_sat, pa_value, i_value, A_value, T)
	po2 = compute_po2(RHc, ph20_sat, pc_value, i_value, A_value, T)
	Enernst = compute_enernst(T, ph2, po2)
	co2 = compute_co2(po2, T)
	theta_act = compute_thetha_act(ip_x1, ip_x2, ip_x3, ip_x4, T, co2, i_value)
	thetha_ohm = compute_thetha_ohm(i_value, Rm, Rc)
	thetha_conc = compute_thetha_conc(B, iden, ilimit_den)
	return compute_Vcell(Enernst, theta_act, thetha_ohm, thetha_conc)	
