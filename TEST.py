"""
A file to test all functions
"""
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from scipy.stats import skellam
import gc
import cProfile
from random_walker import RandomWalker
from HJB_tools import OptimalEvolve
from HJB_solver import HJB_solver
import glob
import pickle
import cupy as cp

class TEST_HJB_tools(object):
	"""
	Here we define a class to run some basic tests on the functions of 

	OptimalEvolve
	HJB_solver
	RandomWalker
	"""
	def __init__(self):
		"""
		set up the objects we'll use to test
		"""
		#the optimal evolver
		#the population at or below which we consider the species extinct 
		extinction_cutoff = 0

		#The minimum resistant population which qualifies for fixation
		fixation_cutoff = 15

		#the cost incurred if extinction is reached
		extinction_cost = 100

		#the reward (negative cost) for fixating 
		fixation_cost = -np.sin(1.4)

		#the per-time-step cost of not having fixated
		time_cost = np.cos(1.4)

		#the cost associated with velocity 1/T
		speed_cost = 0

		parameter_keys = ['extinction cutoff', 'fixation cutoff', 'extinction cost', 'fixation cost', 'time cost', 'speed cost']
		parameter_values = [extinction_cutoff, fixation_cutoff, extinction_cost, fixation_cost, time_cost, speed_cost]

		self.control_parameters_dict = dict(zip(parameter_keys, parameter_values))

		self.channels = ['probability of fixation', 
					'probability of extinction', 
					'expected time of fixation', 
					'expected velocity to fixation']

		self.K = 100

		self.padding = 1.5

		self.sim_size = int(round(self.padding*self.K))

		self.simulation_size_parameters_dict = dict([['K', self.K], ['padding', self.padding], ['sim_size', self.sim_size]])

		self.gammas, self.gammar, self.nu, self.mu, self.phis, self.phir = 10, 10, 50, 0.1, 11, 5

		self.evolution_parameters = [('gammas', self.gammas), ('gammar', self.gammar), ('nu', self.nu), ('mu', self.mu), ('phis', self.phis), ('phir', self.phir)]

		self.evolution_parameters_dict = dict(self.evolution_parameters)

		self.opt = OptimalEvolve(self.control_parameters_dict, self.channels, self.evolution_parameters_dict, self.simulation_size_parameters_dict)

		saved_files = glob.glob('saved_opt.pkl')
		#print(saved_files)
		if saved_files:
			pickle_in = open("saved_opt.pkl","rb")
			self.opt = pickle.load(pickle_in)
		else:
			self.opt.backwards_solve()
			pickle_out = open("saved_opt.pkl","wb")
			pickle.dump(self.opt, pickle_out)
			pickle_out.close()

	def test_isfixed(self):
		threshold = self.opt.fixation_cutoff

		assert(self.opt.isfixed(0, threshold))
		assert(self.opt.isfixed(0, threshold+100))
		assert(not self.opt.isfixed(0, threshold-1))
		assert(self.opt.isfixed(0, threshold))

		for _ in range(20):
			""" None of these should be fixed """
			n0, n1 = 1+int(np.random.rand()*self.opt.K), 1+int(np.random.rand()*self.opt.K)
			assert(not self.opt.isfixed(n0, n1))

		return True

	def test_birthrate(self):
		for _ in range(10):
			n0, n1 = np.array([1+int(np.random.rand()*self.opt.K)]), np.array([1+int(np.random.rand()*self.opt.K)])
			observed_rates = self.opt.BirthRate(n0, n1)

			desired_rate_0 = (1+self.opt.Gs[0])*n0 /(1+(self.opt.Gs[0]*n0+self.opt.Gs[1]*n1)/self.K) + self.opt.mu0 * (n1-n0)
			equiv_desired_rate_0 = (1+self.opt.gs)*n0 /(1+(self.opt.Gs[0]*n0+self.opt.Gs[1]*n1)/self.K) + self.opt.mu0 * (n1-n0)
			desired_rate_1 = (1+self.opt.Gs[1])*n1 /(1+(self.opt.Gs[0]*n0+self.opt.Gs[1]*n1)/self.K) + self.opt.mu0 * (n0-n1)
			equiv_desired_rate_1 = (1+self.opt.gr)*n1 /(1+(self.opt.gs*n0+self.opt.gr*n1)/self.K) + self.opt.mu0 * (n0-n1)

			assert(desired_rate_0 == equiv_desired_rate_0)
			assert(desired_rate_1 == equiv_desired_rate_1)
			assert(abs(np.sum(desired_rate_0) - np.sum(observed_rates[0])) <10**-4)
			assert(abs(np.sum(desired_rate_1) - np.sum(observed_rates[1])) <10**-4)
		return True

	def test_deathrate(self):
		for _ in range(10):
			n0, n1 = np.array([1+int(np.random.rand()*self.opt.K)]), np.array([1+int(np.random.rand()*self.opt.K)])
			observed_rates = self.opt.DeathRate(n0, n1, 1)

			desired_rate_0 = n0*(1+self.opt.Fs[0])
			equiv_desired_rate_0 = n0*(1+self.opt.f0)
			desired_rate_1 = n1*(1+self.opt.f1)
			equiv_desired_rate_1 = n1*(1+self.opt.Fs[1])

			assert(abs(desired_rate_0 - equiv_desired_rate_0)<10**-4)
			assert(abs(desired_rate_1 - equiv_desired_rate_1)<10**-4)
			assert(abs(np.sum(desired_rate_0) - np.sum(observed_rates[0])) <10**-4)
			assert(abs(np.sum(desired_rate_1) - np.sum(observed_rates[1])) <10**-4)
			
		return True

	def test_get_dist(self):
		print("testing get dist")
		n = np.arange(10, 20)[:, None]
		ns = np.array([range(100) for _ in range(n.shape[0])])
		assert(n.shape == (10, 1))
		assert(ns.shape == (10, 100))

		mus1 = np.arange(1, 11)[:, None]
		mus2 = np.arange(0, 10, dtype = np.float32)[:, None]
		#mus2 has a zero in it
		#change this so that it works with the scipy skellam function
		mus2[0,0]+=10**-8
		assert(mus1.shape == mus2.shape == (10, 1))

		#create the W(n'|n) array explicitly
		diff_array = ns-n
		test_output = np.zeros_like(ns, dtype = np.float64)
		for i, v in np.ndenumerate(diff_array):
			if ns[i]>0:
				test_output[i] = skellam.pmf(diff_array[i], mus1[i[0],0], mus2[i[0],0])
					
			else:
				#then we use cdf and the initial n
				init_ns = diff_array-ns
				test_output[i] = skellam.cdf(init_ns[i], mus1[i[0],0], mus2[i[0],0])

			if np.isnan(test_output[i]):
				test_output[i] = 0

		"""
		This should match the ouput of get_dist
		"""
		output1 = self.opt.get_dist(n, ns, mus1, mus2, normalize=False)

		assert(output1.shape == test_output.shape)

		#what is the maximum percent error
		percent_error = np.max(np.abs(output1-test_output)/(np.abs(output1)+10**-9))

		assert(percent_error<= 10**-5)

		"""
		Now test the normalization
		"""
		test_norm = []
		for i in range(10):
			v = test_output[i, :]
			test_norm.append(v/np.sum(v))
		test_norm = np.array(test_norm)

		output2 = self.opt.get_dist(n, ns, mus1, mus2, normalize=True)

		assert(output2.shape == test_norm.shape)

		#what is the maximum percent error
		percent_error = np.max(np.abs(output2-test_norm)/(np.abs(output2)+10**-9))

		assert(percent_error<= 10**-5)


		return True

	def test_Ws(self):
		"""
		This function should map from point to 
		the whole sim_size
		"""
		positions = np.array([[10,10], [0,0], [0, self.opt.fixation_cutoff]])
		u = 1
		probs0, probs1 = self.opt.Ws(positions, u)
		#print(probs0.shape, probs1.shape, self.opt.fixation_cutoff, type(self.opt.fixation_cutoff))
		assert(probs0.shape == (3, self.opt.sim_size) and probs1.shape == (3, self.opt.sim_size))

		"""
		Check that extinction and fixation are sticky
		"""
		#extinction
		assert(abs(probs0[1][0] - 1)<10**-5 and abs(probs1[1][0] - 1)<10**-5)# all prob stays on extinct
		assert(abs(probs0[2][0] - 1)<10**-5)
		assert(abs(probs1[2][int(self.opt.fixation_cutoff)] - 1)<10**-5)# all prob stays on fixation

		return True

	def test_W(self):
		"""
		This function relies on getting the right position indices
		We'll test this here
		"""

		n0s = [0,1,0,1, 25]
		n1s = [0,0,10,2, 20]

		position_indices = np.ravel_multi_index([n0s, n1s], (self.opt.sim_size, self.opt.sim_size))

		#should match this
		test_index_0 = n0s[0]*self.opt.sim_size + n1s[0]
		test_index_1 = n0s[1]*self.opt.sim_size + n1s[1]
		test_index_2 = n0s[2]*self.opt.sim_size + n1s[2]
		test_index_3 = n0s[3]*self.opt.sim_size + n1s[3]
		test_index_4 = n0s[4]*self.opt.sim_size + n1s[4]

		assert(test_index_0 == position_indices[0])
		assert(test_index_1 == position_indices[1])
		assert(test_index_2 == position_indices[2])
		assert(test_index_3 == position_indices[3])
		assert(test_index_4 == position_indices[4])

		#print(test_index_0 , position_indices[0])
		#print(test_index_1 , position_indices[1])
		#print(test_index_2 , position_indices[2])
		#print(test_index_3 , position_indices[3])
		#print(test_index_4 , position_indices[4])

		for _ in range(10):
			n0, n1 = int(np.random.rand()*self.opt.sim_size), int(np.random.rand()*self.opt.sim_size)
			index = np.ravel_multi_index([[n0], [n1]], (self.opt.sim_size, self.opt.sim_size))
			
			corresponding_position = np.reshape(self.opt.space_mesh, (-1, 2))[index]
			#print(corresponding_position[0, 0], n0)
			#print(corresponding_position[0, 1], n1)
			assert(np.sum(corresponding_position[0, 0])==n0 and np.sum(corresponding_position[0, 1])==n1)

		return True

	def test_fixation_func(self):
		assert(self.opt.fixation_func((0,0)) == 0)
		assert(self.opt.fixation_func((0,self.opt.fixation_cutoff)) == 1)
		assert(self.opt.fixation_func((0,self.opt.fixation_cutoff-1)) == 0)
		assert(self.opt.fixation_func((10,10)) == 0)

		return True

	def test_extinction_func(self):
		assert(self.opt.extinction_func((0,0)) == 1)
		assert(self.opt.extinction_func((0,self.opt.fixation_cutoff)) == int(self.opt.fixation_cutoff<=self.opt.extinction_cutoff))
		assert(self.opt.extinction_func((0,self.opt.extinction_cutoff)) == 1)

		return True

	def test_expected_time_func(self):
		"""
		this should match fixation_func
		"""
		for _ in range(15):
			n0, n1 = int(np.random.rand()*self.opt.sim_size), int(np.random.rand()*self.opt.sim_size)
			assert(1-self.opt.fixation_func((n0, n1)) == self.opt.expected_time_func((n0,n1), 1, .1))
			
		assert(1-self.opt.fixation_func((0,self.opt.fixation_cutoff)) == self.opt.expected_time_func((0,self.opt.fixation_cutoff), 1, .1) == 0)
		return True

	def test_expected_velocity_func(self):
		"""
		For now I don't plan on using this, so I won't test it
		"""
		pass

	def test_propagation_to_time(self):
		"""
		For now I don't plan on using this, so I won't test it
		"""
		pass

	def test_phi(self):
		"""
		this should penalize extinction and reward fixation 
		these should apply to the first entry
		J should have shape (sim_size**2, len(channels)+1)
		So phi has shape len(channels)+1
		"""
		n0, n1 = 0, self.opt.extinction_cutoff
		assert(self.opt.phi((n0,n1))[0] == self.opt.extinction_cost)

		n0, n1 = 0, self.opt.fixation_cutoff
		assert(self.opt.phi((n0,n1))[0] == self.opt.fixation_cost)

		for _ in range(25):
			n0, n1 = int(np.random.rand()*self.opt.sim_size), int(np.random.rand()*self.opt.sim_size)

			n0+=self.opt.extinction_cutoff+1
			n1+=self.opt.extinction_cutoff+1
			#there's no way now that the population can be fixated or extinct

			#check the shape
			assert(self.opt.phi((n0,n1)).shape == (len(self.channels)+1,))

			#check the value
			assert(self.opt.phi((n0,n1))[0] == 0)
		return True

	def test_R(self):
		"""
		This should have size 
		(sim_size**2, len(channels)+1)

		"""
		u=1
		Rtest = self.opt.R(np.reshape(self.opt.space_mesh, (-1, 2)), 1, u)
		assert(Rtest.shape == (self.opt.sim_size**2, len(self.opt.channels)+1))

		"""
		test that the channel for time accumulation is working as expected
		"""
		index = self.opt.channels.index('expected time of fixation')

		time_to_fix_array = np.reshape(Rtest[..., index+1], (self.opt.sim_size, self.opt.sim_size))

		space_array = np.reshape(np.reshape(self.opt.space_mesh, (-1, 2)), (self.opt.sim_size, self.opt.sim_size, 2))
		for i, v in np.ndenumerate(time_to_fix_array):
			assert(v == 1-int(self.opt.isfixed(*space_array[i])))

		return True

	def test_format_array(self):
		test_array = np.random.rand(self.opt.sim_size, self.opt.sim_size)
		test_flat = np.reshape(np.copy(test_array), -1)
		assert(np.sum(np.abs(self.opt.format_array(test_flat)-test_array))<10**-5)
		return True

	def test_format_for_viewing(self):
		test_array = np.random.rand(self.opt.sim_size, self.opt.sim_size)
		transposed_flipped_array = np.transpose(test_array)[::-1]
		assert(np.sum(np.abs(self.opt.format_for_viewing(test_array)-transposed_flipped_array))<10**-5)
		return True

	def test_give_j(self):
		should_match_this = np.reshape(self.opt.optimal_ctg[..., 0], (self.opt.sim_size, self.opt.sim_size))
		
		assert(np.sum(np.abs(self.opt.give_j()-should_match_this))<10**-5)
		return True

	def test_give_u(self):
		should_match_this = np.reshape(self.opt.optimal_control, (self.opt.sim_size, self.opt.sim_size))
		assert(np.sum(np.abs(self.opt.give_u()-should_match_this))<10**-5)
		return True

	def test_present_u(self):
		""" 
		This is a visual function, so i won't test it
		"""
		pass

	def test_present_j(self):
		""" 
		This is a visual function, so i won't test it
		"""
		pass

	def test_prob_extinction(self):
		"""
		This is a very short function and I won't test it here
		"""
		pass

	def test_prob_fixation(self):
		"""
		This is a very short function and I won't test it here
		"""
		pass

	def test_expected_time(self):
		"""
		This is a very short function and I won't test it here
		"""
		pass

	def test_expected_velocity(self):
		"""
		This is a very short function and I won't test it here
		"""
		pass

	def test_propagation(self):
		"""
		This is a very short function and I won't test it here
		"""
		pass

	def test_symmetric_control(self):
		"""
		if there is no difference between the control's effect 
		on one allele or the other, the control should default to 0
		"""
		extinction_cutoff = 0

		#The minimum resistant population which qualifies for fixation
		fixation_cutoff = 15

		#the cost incurred if extinction is reached
		extinction_cost = 100

		#the reward (negative cost) for fixating 
		fixation_cost = -np.sin(1.4)

		#the per-time-step cost of not having fixated
		time_cost = np.cos(1.4)

		#the cost associated with velocity 1/T
		speed_cost = 0

		parameter_keys = ['extinction cutoff', 'fixation cutoff', 'extinction cost', 'fixation cost', 'time cost', 'speed cost']
		parameter_values = [extinction_cutoff, fixation_cutoff, extinction_cost, fixation_cost, time_cost, speed_cost]

		control_parameters_dict = dict(zip(parameter_keys, parameter_values))

		channels = []

		#make it smaller
		K = 50

		padding = 1.5

		sim_size = int(round(padding*K))

		simulation_size_parameters_dict = dict([['K', K], ['padding', padding], ['sim_size', sim_size]])

		gammas, gammar, nu, mu, phis, phir = 10, 10, 10, 0.1, 0, 0

		evolution_parameters = [('gammas', gammas), ('gammar', gammar), ('nu', nu), ('mu', mu), ('phis', phis), ('phir', phir)]

		evolution_parameters_dict = dict(evolution_parameters)

		opt = OptimalEvolve(control_parameters_dict, channels, evolution_parameters_dict, simulation_size_parameters_dict)
		opt.backwards_solve()

		"""
		since the effect of the control is the same on both alleles
		we expect the optimal control to be completely zero
		"""
		#plt.imshow(np.reshape(opt.optimal_control , (opt.sim_size, opt.sim_size)))
		#plt.show()
		#print(opt.optimal_control)
		assert(phis==phir)
		assert(np.max(opt.optimal_control) == 0)

		return True

	def TEST_HJB_tools(self):
		"""
		Run all tests
		"""
		self.test_isfixed()
		self.test_birthrate()
		self.test_deathrate()
		self.test_get_dist()
		self.test_Ws()
		self.test_W()
		self.test_fixation_func()
		self.test_extinction_func()
		self.test_expected_time_func()
		self.test_expected_velocity_func()
		self.test_propagation_to_time()
		self.test_phi()
		self.test_R()
		self.test_format_array()
		self.test_format_for_viewing()
		self.test_give_j()
		self.test_give_u()
		self.test_present_u()
		self.test_present_j()
		self.test_prob_extinction()
		self.test_prob_fixation()
		self.test_expected_time()
		self.test_expected_velocity()
		self.test_propagation()
		self.test_symmetric_control()


class TEST_HJB_solver(object):
	"""
	Here we define a class to run some basic tests on the functions of 

	OptimalEvolve
	HJB_solver
	RandomWalker
	"""
	def __init__(self):
		"""
		We'll make a simple solver and check all functions it runs

		trivial target:
		y=0 => 100 cost
		x=0 => -100 cost

		"""
		self.space_mesh = np.array([[(i, j) for j in range(50)] for i in range(50)])
		self.space_dims = list(self.space_mesh.shape[:-1])
		self.sim_size = 50
		self.channels = []
		self.record_time = False
		self.xmesh = np.reshape(self.space_mesh, -1)
		self.time_mesh = np.linspace(0, 1, 75)


		self.solver = HJB_solver(phi = self.phi, 
								R = self.R, 
								W = self.W, 
								xmesh = np.reshape(self.space_mesh, (-1, len(self.space_dims))), 
								sim_size = self.sim_size, 
								tmesh = self.time_mesh, 
								channels = self.channels, 
								record_time = self.record_time)


		
		self.initial_index = self.solver.index
		self.solver.percent_change_cutoff = 0.0
		self.solver.backwards_solve()
		self.final_index = self.solver.index

		#now we collect results
		self.optimal_control = self.solver.umesh[-1, :]
		self.optimal_ctg = self.solver.jmesh[-1, :]


	def phi(self, n):
		return np.array([-100*int(n[1]==0)+100*int(n[0]==0)])

	def R(self, xs, t, u):
		return 0

	#now define the trivial propagator
	def W(self, positions, u):
		props0, props1 = [], []

		for pos in positions:
			prop0, prop1 = np.zeros(50), np.zeros(50)
			"""
			if we've hit either axis we stick
			"""
			x, y = pos
			if x==0 or y == 0:
				prop0[x]=1
				prop1[y]=1
			else:
				#the control can affect things
				#u=1 shifts down
				#u=0 shifts left
				if u == 0:
					prop0[x-1]=1
					prop1[y]=1
				else:
					assert(u==1)
					prop0[x]=1
					prop1[y-1]=1

			props0.append(prop0)
			props1.append(prop1)

		return np.array(props0), np.array(props1)

	"""
	Now we define the tests
	"""
	def test_custom_dot(self):
		"""
		test the custom dot here
		first do it out explicitly, then confirm this matches what we 
		calculated

		Note this function calculates the rhs (flattened) of

		J(n, t-1) = Sum_{n'} J(n', t)W_u(n'|n)

		"""
		#these are the x and y propagators. 
		#note each point has its own propagator
		props0 = np.random.rand(50*50, 50)
		props1 = np.random.rand(50*50, 50)
		#normalizing
		props0 /= np.sum(props0, -1)[:, None]
		props1 /= np.sum(props1, -1)[:, None]
		#test normalization

		assert(abs(max(np.sum(props0, -1))-1)<10**5)
		assert(abs(min(np.sum(props0, -1))-1)<10**5)
		assert(abs(max(np.sum(props1, -1))-1)<10**5)
		assert(abs(min(np.sum(props1, -1))-1)<10**5)

		test_array = np.random.rand(self.sim_size, self.sim_size)
		test_flat = np.reshape(test_array, (-1, 1))

		#now compute the inner product at each point
		test_outputs = np.zeros_like(test_array)
		for i, v in np.ndenumerate(test_array):
			x, y = i 
			long_index = x*50 +y

			shaped_prop = np.outer(props0[long_index, :], props1[long_index, :])
			inner_prod = np.sum(shaped_prop*test_array)
			test_outputs[i] = inner_prod

		#now custom dot should match this
		#print(test_flat.shape, props0.shape, props1.shape)
		outputs = cp.asnumpy(self.solver.custom_dot(test_flat, props0, props1))
		outputs = np.reshape(outputs, (self.sim_size, self.sim_size))
		#outputs and test_outputs should match
		difference = np.sum(np.abs(outputs - test_outputs))
		#print("difference: ", difference)

		assert(difference<10**-8)

		return True 

	def test_expJ(self):
		"""
		This is really just an implementation of custom dot, 
		so we won't test it here
		"""
		pass

	def test_backstep(self):
		"""
		For now, I'll let the mechanics here be tested by the final 
		test_results function
		"""
		pass

	def test_backwards_solve(self):
		"""
		This is just a packaging of backstep repeatedly applied
		We will test that it does enough steps to match self.N
		"""
		numsteps = abs(self.initial_index - self.final_index)
		assert(numsteps == self.time_mesh.size-1)
		return True

	def test_results_1(self):
		"""
		"""
		J = np.reshape(self.optimal_ctg, (50,50))
		u = np.reshape(self.optimal_control, (50,50))

		#the control on each of the axes should be zero
		#since bc's are sticky, and the control defaults to 
		#u=0 if there is no difference
		y_axis_u = u[0, :]
		x_axis_u = u[:, 0]

		assert(max(y_axis_u)==0)
		assert(max(x_axis_u)==0)

		"""
		The optimal control here is actually very subtle
		we know hitting y=0 gives you a reward of 100
		hitting x=0 gives a cost of 100

		AND if two choices are equal, it chooses 
		the u=0 choice

		therefore, this will choose the path u=0 until x=1
		then u=1 down to y=0


		"""
		#x=1, y=1, 2, 3... should give u=1
		assert(np.min(u[1, 1:])==1)
		#the rest should be zero
		assert(np.max(u[2:, :])==0)

		return True

	def test_results_2(self):
		"""
		now we include R which accumulates as a function of time
		This should make the paths strictly down
		"""
		def modified_R(xs, t, u):
			assert(len(xs.shape)==2)
			Rshape = (xs.shape[0], 1)
			#the last element of the shape will be 2

			output = []
			for i in range(xs.shape[0]):
				pos = xs[i] 
				x, y = pos 

				if y == 0:
					output.append([0])
				else:
					output.append([1])

			return np.array(output)

		solver = HJB_solver(phi = self.phi, 
								R = modified_R, 
								W = self.W, 
								xmesh = np.reshape(self.space_mesh, (-1, len(self.space_dims))), 
								sim_size = self.sim_size, 
								tmesh = self.time_mesh, 
								channels = self.channels, 
								record_time = self.record_time)


		
		
		solver.percent_change_cutoff = 0.0
		solver.backwards_solve()
		

		#now we collect results
		optimal_control = solver.umesh[-1, :]
		optimal_ctg = solver.jmesh[-1, :]

		J = np.reshape(optimal_ctg, (50,50))
		u = np.reshape(optimal_control, (50,50))

		#now all but the axes should have u=1

		assert(np.max(u[0, :])==np.max(u[:, 0])==0)
		#print(u[1:, 1:])

		assert(np.min(u[1:, 1:])==1)

		return True




	def TEST_HJB_solver(self):
		"""
		"""
		self.test_custom_dot()
		self.test_expJ()
		self.test_backstep()
		self.test_backwards_solve()
		self.test_results_1()
		self.test_results_2()


class TEST_random_walker(object):
	"""
	Here we define a class to run some basic tests on the functions of 

	OptimalEvolve
	HJB_solver
	RandomWalker
	"""
	def __init__(self):
		"""
		set up the objects we'll use to test
		"""
		#the optimal evolver
		extinction_cutoff = 0

		#The minimum resistant population which qualifies for fixation
		fixation_cutoff = 15

		#the cost incurred if extinction is reached
		extinction_cost = 100

		#the reward (negative cost) for fixating 
		fixation_cost = -np.sin(1.4)

		#the per-time-step cost of not having fixated
		time_cost = np.cos(1.4)

		#the cost associated with velocity 1/T
		speed_cost = 0

		parameter_keys = ['extinction cutoff', 'fixation cutoff', 'extinction cost', 'fixation cost', 'time cost', 'speed cost']
		parameter_values = [extinction_cutoff, fixation_cutoff, extinction_cost, fixation_cost, time_cost, speed_cost]

		self.control_parameters_dict = dict(zip(parameter_keys, parameter_values))

		self.channels = ['probability of fixation', 
					'probability of extinction', 
					'expected time of fixation', 
					'expected velocity to fixation']

		self.K = 100

		self.padding = 1.5

		self.sim_size = int(round(self.padding*self.K))

		self.simulation_size_parameters_dict = dict([['K', self.K], ['padding', self.padding], ['sim_size', self.sim_size]])

		self.gammas, self.gammar, self.nu, self.mu, self.phis, self.phir = 10, 10, 50, 0.1, 11, 5

		self.evolution_parameters = [('gammas', self.gammas), ('gammar', self.gammar), ('nu', self.nu), ('mu', self.mu), ('phis', self.phis), ('phir', self.phir)]

		self.evolution_parameters_dict = dict(self.evolution_parameters)

		

		self.newK = 1000
		self.new_sim_size = int(round(self.padding*self.newK))
		#the RandomWalker
		saved_files = glob.glob('saved_walk.pkl')
		#print(saved_files)
		if saved_files:
			pickle_in = open("saved_walk.pkl","rb")
			self.opt, self.rw = pickle.load(pickle_in)
		else:
			self.opt = OptimalEvolve(self.control_parameters_dict, self.channels, self.evolution_parameters_dict, self.simulation_size_parameters_dict)
			self.opt.backwards_solve()
			self.rw = RandomWalker(self.opt, self.newK)
			to_save = [self.opt, self.rw]
			pickle_out = open("saved_walk.pkl","wb")
			pickle.dump(to_save, pickle_out)
			pickle_out.close()
		

	def test_rw_isfixed(self):
		assert(self.rw.rw_isfixed(0, self.rw.rw_fixation_cutoff))
		assert(self.rw.rw_isfixed(0, self.rw.rw_fixation_cutoff+100))
		assert(not self.rw.rw_isfixed(0, self.rw.rw_fixation_cutoff-1))
		assert(not self.rw.rw_isfixed(10,10))

		assert(abs(self.rw.rw_fixation_cutoff - self.rw.scale*self.opt.fixation_cutoff)<1.5)

		return True

	def test_rw_isextinct(self):
		assert(self.rw.rw_isextinct(0, self.rw.rw_extinction_cutoff))
		assert(not self.rw.rw_isextinct(0, self.rw.rw_extinction_cutoff+100))
		assert(not self.rw.rw_isextinct(1,self.rw.rw_extinction_cutoff))

		assert(abs(self.rw.rw_extinction_cutoff - self.rw.scale*self.opt.extinction_cutoff)<1.5)

		return True

	def test_rw_BirthRate(self):
		"""
		This is in-essence already tested 
		"""
		pass

	def test_rw_DeathRate(self):
		"""
		This is in-essence already tested 
		"""
		pass

	def test_approx_skellam_pmf(self):
		"""
		Make sure this matches skellam
		"""
		mu1 = 9
		mu2 = 8

		skellam_output = skellam.pmf(np.arange(-20,20), mu1, mu2)
		approx_output = self.rw.approx_skellam_pmf(np.arange(-20,20), mu1, mu2)

		assert(np.sum(np.abs(skellam_output-approx_output))<0.03)

	def test_get_dist(self):
		"""
		This is in-essence already tested 
		"""
		pass

	def test_step(self):
		"""
		This is a really straightforward selection from two skellam distributions
		Lets make sure that the distribution looks like a skellam
		"""
		num_tests = 4
		ns = np.arange(self.new_sim_size)[None, :]

		for _ in range(num_tests):
			n0, n1 = int(np.random.rand(1)*self.newK), int(np.random.rand(1)*self.newK)
			n0 = np.reshape(n0, (1,))
			n1 = np.reshape(n1, (1,))
			u=1
			"""
			now generate a distribution of steps
			make sure it resembles the expected distribution
			"""
			mc_runs = 2500
			x_dist = np.zeros(self.new_sim_size)
			y_dist = np.zeros(self.new_sim_size)

			for run in range(mc_runs):
				new_n0, new_n1 = self.rw.step(n0, n1, u)
				#print(new_n0, new_n1)
				x_dist[int(new_n0)]+=1
				y_dist[int(new_n1)]+=1

			x_dist /= np.sum(x_dist)
			y_dist /= np.sum(y_dist)

			"""
			now generate the distribution we expect
			"""
			#print('ns shape', n0.shape, n1.shape)
			b0, b1 = self.rw.rw_BirthRate(n0, n1)
			d0, d1 = self.rw.rw_DeathRate(n0, n1, u)

			#print(b0.shape, d0.shape)

			skellam_x = self.rw.get_dist(n0, ns, b0, d0)
			skellam_y = self.rw.get_dist(n1, ns, b1, d1)
			#print(skellam_x.shape)
			#print('   ')

			expected_mean_x = np.sum(ns*skellam_x)
			expected_mean_y = np.sum(ns*skellam_y)
			observed_mean_x = np.sum(ns*x_dist)
			observed_mean_y = np.sum(ns*y_dist)
			
			#the means should be pretty close
			assert(abs(expected_mean_x-observed_mean_x)/expected_mean_x < .05)
			assert(abs(expected_mean_y-observed_mean_y)/expected_mean_y < .05)

			#the variances should be fairly close as well
			expected_var_x = np.sum((ns**2  - expected_mean_x**2)*skellam_x)
			expected_var_y = np.sum((ns**2  - expected_mean_y**2)*skellam_y)
			observed_var_x = np.sum((ns**2  - observed_mean_x**2)*x_dist)
			observed_var_y = np.sum((ns**2  - observed_mean_y**2)*y_dist)

			print('x var relative deviation', abs(expected_var_x-observed_var_x)/expected_var_x)
			print('y var relative deviation', abs(expected_var_y-observed_var_y)/expected_var_y)

			assert(abs(expected_var_x-observed_var_x)/expected_var_x < .08)
			assert(abs(expected_var_y-observed_var_y)/expected_var_y < .08)

		return True

	def test_convert_control(self):
		"""
		We have visualized this already
		"""
		pass

	def test_walk(self):
		"""
		We'll simply make sure that the number of steps matches 
		what we want it to, and that the end condition feature works
		"""

		#we'll start here for the tests
		n0, n1 = 60,60

		def end_condition(x, y):
			if x>n0+20 or x<n0-20:
				return True 
			if y>n1+20 or y<n1-20:
				return True 
			else:
				return False 

		path_outputs = []
		test_reps = 20
		for _ in range(test_reps):
			path = self.rw.walk(n0, n1, steps = 40, end_condition = end_condition)
			path_outputs.append(path)

		#now confirm that this works as intended
		for path in path_outputs:
			if end_condition(*path[-1]):
				#this should be the first point satisfying that condition
				for step in path[:-1]:
					assert(not end_condition(*step))
			else:
				#this should have run out of steps
				assert(len(path) == 40)

		"""
		before we go, test this for if track_control == True
		"""
		path_outputs = []
		test_reps = 20
		for _ in range(test_reps):
			path_and_control = self.rw.walk(n0, n1, steps = 40, end_condition = end_condition, track_control = True)
			path_outputs.append(path_and_control)

		#now confirm that this works as intended
		for path, control_hist in path_outputs:
			#print("path len", len(path))
			#print("control len", len(control_hist))
			assert(len(control_hist) == len(path)-1)

			#control hist should be 1's and 0's 
			assert(set(control_hist).issubset({0, 1}))

			if end_condition(*path[-1]):
				#this should be the first point satisfying that condition
				for step in path[:-1]:
					assert(not end_condition(*step))
			else:
				#this should have run out of steps
				assert(len(path) == 40)

		return True 

	def test_ensemble_paths(self):
		num=100
		output = self.rw.ensemble_paths(70, 70, num, 500)
		fixed_paths, extinct_paths, other_paths, control_histories_fixed, control_histories_extinct, control_histories_other = output
		assert(len(fixed_paths) == len(control_histories_fixed))
		assert(len(extinct_paths) == len(control_histories_extinct))
		assert(len(other_paths) == len(control_histories_other))

		if fixed_paths:
			assert(len(fixed_paths[0]) == len(control_histories_fixed[0])+1)
		if extinct_paths:
			assert(len(extinct_paths[0]) == len(control_histories_extinct[0])+1)
		if other_paths:
			assert(len(other_paths[0]) == len(control_histories_other[0])+1)

		assert(len(fixed_paths)+len(extinct_paths)+len(other_paths) == num)

		return True

	def TEST_random_walker(self):
		self.test_rw_isfixed()
		self.test_rw_isextinct()
		self.test_rw_BirthRate()
		self.test_rw_DeathRate()
		self.test_approx_skellam_pmf()
		self.test_get_dist()
		self.test_step()
		self.test_convert_control()
		self.test_walk()
		self.test_ensemble_paths()


hjb_tool_tester = TEST_HJB_tools()
hjb_tool_tester.TEST_HJB_tools()

hjb_solver_tester = TEST_HJB_solver()
hjb_solver_tester.TEST_HJB_solver()

hjb_random_walker_tester = TEST_random_walker()
hjb_random_walker_tester.TEST_random_walker()

print('All tests successful')