import pickle 
from HJB_solver import *
from HJB_tools import * 
from random_walker import *

"""
define parameters
"""


ks = list(map(int, np.exp(np.linspace(np.log(100), np.log(501), 10))))

phi_s_list = list(map(int, np.exp(np.linspace(np.log(5+.5), np.log(25+.5), 5))))

for k in ks:
	for p in phi_s_list:

		#the population at or below which we consider the species extinct 
		extinction_cutoff = 0

		#The minimum resistant population which qualifies for fixation
		fixation_cutoff = 15

		#the cost incurred if extinction is reached
		extinction_cost = np.sin((np.pi/2)-0.02)

		#the reward (negative cost) for fixating 
		fixation_cost = 0#-1#np.sin(1.4)

		#the per-time-step cost of not having fixated
		time_cost = np.cos((np.pi/2)-0.02)

		#the cost associated with velocity 1/T
		speed_cost = 0

		parameter_keys = ['extinction cutoff', 'fixation cutoff', 'extinction cost', 'fixation cost', 'time cost', 'speed cost']
		parameter_values = [extinction_cutoff, fixation_cutoff, extinction_cost, fixation_cost, time_cost, speed_cost]

		control_parameters_dict = dict(zip(parameter_keys, parameter_values))

		channels = ['probability of fixation', 
					'probability of extinction', 
					'expected time of fixation']

		K = k

		padding = 1.5

		sim_size = int(round(padding*K))

		simulation_size_parameters_dict = dict([['K', K], ['padding', padding], ['sim_size', sim_size]])

		gammas, gammar, nu, mu, phis, phir = 10, 10, 1/7, 0.1, p, 5

		evolution_parameters = [('gammas', gammas), ('gammar', gammar), ('nu', nu), ('mu', mu), ('phis', phis), ('phir', phir)]

		evolution_parameters_dict = dict(evolution_parameters)

		opt = OptimalEvolve(control_parameters_dict, channels, evolution_parameters_dict, simulation_size_parameters_dict)
		opt.backwards_solve()

		string = 'k_'+str(k)+'_phi_s_'+str(p)

		pickle_out = open("saved_opt_"+string+".pkl","wb")
		pickle.dump(opt, pickle_out)
		pickle_out.close()



