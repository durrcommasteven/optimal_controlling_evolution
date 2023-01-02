import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import glob
from time import time
from scipy.stats import skellam
from HJB_solver import HJB_solver
import gc
import h5py

"""
Define a class to hold the evolutionary optimizer
"""


class OptimalEvolve(object):
	"""
	A class to contain all variables and functions for the optimal fixation 
	procedure
	"""

	def __init__(self, control_parameters_dict, channels, evolution_parameters_dict, simulation_size_parameters_dict):
		"""
		load the parameters using dictionaries
		if the parameters needed are not present, print them and return an error
		"""
		
		"""
		Load control_parameter_dict
		"""
		#check for missing parameters
		control_parameters = ['extinction cutoff', 'fixation cutoff', 'extinction cost', 'fixation cost', 'time cost', 'speed cost']
		missing = []
		for param in control_parameters:
			if param not in control_parameters_dict:
				missing.append(param)

		if missing:
			print('control_parameter_dict missing the following parameters: ')
			for param in missing:
				print(param)
			assert(False)

		#now save these values
		self.extinction_cutoff = control_parameters_dict['extinction cutoff']
		self.fixation_cutoff = control_parameters_dict['fixation cutoff']
		self.extinction_cost = control_parameters_dict['extinction cost']
		self.fixation_cost = control_parameters_dict['fixation cost']
		self.time_cost = control_parameters_dict['time cost']
		self.speed_cost = control_parameters_dict['speed cost']

		"""
		Load channels for tracking the expectation of quantities
		"""
		#check that the channels provided are a subset of the allowed channels
		self.channels = channels 
		self.all_possible_channels = set(['probability of fixation',
							 'probability of extinction',
							 'expected time of fixation',
							 'expected velocity to fixation',
							 'propagation to fixation at time'])

		assert(set(self.channels).issubset(self.all_possible_channels))

		#set default propagation_time
		self.propagation_time = 0
		self.record_time = bool(self.propagation_time)


		"""
		Load evolution parameters
		"""
		#check for missing parameters
		evolution_parameters = ['gammas', 'gammar', 'nu', 'mu', 'phis', 'phir']
		missing = []
		for param in evolution_parameters:
			if param not in evolution_parameters_dict:
				missing.append(param)

		if missing:
			print('evolution_parameter_dict missing the following parameters: ')
			for param in missing:
				print(param)
			assert(False)

		#now save these values
		self.gammas = float(evolution_parameters_dict['gammas'])
		self.gammar = float(evolution_parameters_dict['gammar'])
		self.nu = float(evolution_parameters_dict['nu'])
		self.mu = float(evolution_parameters_dict['mu'])
		self.phis = float(evolution_parameters_dict['phis'])
		self.phir = float(evolution_parameters_dict['phir'])


		"""
		Load the parameters dictating the size of the simulation
		"""
		#check for missing parameters
		simulation_size_parameters = ['K', 'padding', 'sim_size']
		missing = []
		for param in simulation_size_parameters:
			if param not in simulation_size_parameters_dict:
				missing.append(param)

		if missing:
			print('control_parameter_dict missing the following parameters: ')
			for param in missing:
				print(param)
			assert(False)

		#now save the values
		self.K = simulation_size_parameters_dict['K']
		assert(0<self.K<1000)

		self.padding = simulation_size_parameters_dict['padding']
		self.sim_size = simulation_size_parameters_dict['sim_size']

		self.space_mesh = []
		for i in range(self.sim_size):
			self.space_mesh.append([(i, j) for j in range(self.sim_size)])

		self.space_mesh=np.array(self.space_mesh)
		self.space_dims = list(self.space_mesh.shape[:-1])

		"""
		Now find the specific values for our K
		"""

		self.gs = self.gammas/self.K
		self.gr = self.gammar/self.K
		self.mu0 = self.mu/self.K
		self.f0 = self.phis/self.K
		self.f1 = self.phir/self.K
		self.N = int(round(self.K/self.nu))

		self.Gs = [self.gs, self.gr]
		self.Fs = [self.f0, self.f1]

		self.time_mesh = np.arange(self.N) # np.linspace(0, 1, self.N)

		self.dt = self.time_mesh[1]-self.time_mesh[0]

		"""
		Now load the propagators 
		"""
		self.prop_identifier = "gammas_{}_gammar_{}_nu_{}_mu_{}_phis_{}_phir_{}_sim_size_{}".format(self.gammas, 
																							self.gammar, 
																							self.nu, 
																							self.mu, 
																							self.phis, 
																							self.phir, 
																							self.sim_size)
		

		self.propagator_file = glob.glob('propagators'+self.prop_identifier+'.hdf5')

		if not self.propagator_file:
			#this was a list, now we replace its value with the desired propagator file identifier
			self.propagator_file = 'propagators'+self.prop_identifier+'.hdf5'

			with h5py.File(self.propagator_file, "w") as f:

				print('Creating and saving the propagators')

				time0 = time()

				dset = f.create_dataset(
    								"ws_n0_n1_u0", 
    								data = self.Ws(np.reshape(self.space_mesh, (-1, len(self.space_dims))), 0),
    								dtype = 'float32')

				time1 = time()

				print('Expected time remaining: ', (time1-time0)//3600, ' hours, ', ((time1-time0)//60)%60, ' minutes, ', (time1-time0)%60, ' seconds')


				dset = f.create_dataset(
    								"ws_n0_n1_u1", 
    								data = self.Ws(np.reshape(self.space_mesh, (-1, len(self.space_dims))), 1),
    								dtype='float32')

		else:
			#then the desired file is present 
			print('Propagators identified in memory')
			#this was a list, now we replace its value with the desired propagator file identifier
			self.propagator_file = 'propagators'+self.prop_identifier+'.hdf5'

		"""
		List the functions used for tracking channels
		We use these to construct the dictionaries:

		R_functions_dict
		phi_fucntions_dict
		"""
		phi_channels = ['probability of fixation','probability of extinction']
		R_channels = ['expected time of fixation', 'expected velocity to fixation', 'propagation to fixation at time']
		phi_functions = [self.fixation_func, self.extinction_func]
		R_functions = [self.expected_time_func, self.expected_velocity_func, self.propagation_to_time]

		self.R_channels_dict =  dict(zip(R_channels, R_functions))
		self.phi_channels_dict = dict(zip(phi_channels, phi_functions))

		"""
		Finally, create an instance of the HJB_solver class which actually does the solving
		"""
		self.solver = HJB_solver(phi = self.phi, 
								R = self.R, 
								W = self.W, 
								xmesh = np.reshape(self.space_mesh, (-1, len(self.space_dims))), 
								sim_size = self.sim_size, 
								tmesh = self.time_mesh, 
								channels = self.channels, 
								record_time = self.record_time)

		#until the following values are set, its value is None
		self.optimal_ctg = None
		self.optimal_control = None

		#we need some things to be true about solver
		#the assertion(s) following are to ensure this
		assert(self.solver.dt == self.dt)

		#now collect any unassigned scraps
		gc.collect()


	"""
	First we define functions which define the actual evolution process:
	birthrates and deathrates, the resulting propagators
	"""

	"""
	define a helper function for determining whether fixation has been reached
	"""
	def isfixed(self, n0, n1):
		"""
		A function to describe whether the resistant population is considered fixed
		"""
		if n0==0 and n1>=self.fixation_cutoff:
			return True
		return False

	
	def BirthRate(self, n0, n1):
		"""
		Given n0, n1, return the Birthrate
		"""
		birth0 = (1+self.Gs[0])*n0/(1 + self.Gs[0]*n0/self.K + self.Gs[1]*n1/self.K)
		mutation0 = self.mu0*(n1-n0)
		
		birth1 = (1+self.Gs[1])*n1/(1 + self.Gs[0]*n0/self.K + self.Gs[1]*n1/self.K)
		mutation1 = self.mu0*(n0-n1)
		
		growth_rate0 = (birth0 + mutation0).astype(np.float32, copy=False)
		growth_rate1 = (birth1 + mutation1).astype(np.float32, copy=False)
		
		return (growth_rate0, growth_rate1)

	
	def DeathRate(self, n0, n1, u):
		"""
		
		"""
		death0 = (n0*(1+u*self.Fs[0])).astype(np.float32, copy=False)
		death1 = (n1*(1+u*self.Fs[1])).astype(np.float32, copy=False)
		
		return (death0, death1)  


	def get_dist(self, n, ns, mu1, mu2, normalize=True):
		"""
		here n is an array of shape (num,1)
		ns is an array of shape (num, sim_size)
		
		mu1 has shape (num,1)
		mu2 has shape (num,1)
		
		A skellam function to deal with mu1, mu2 which may be zero
		"""
		
		"""
		we need to control for the mu values
		
		this is slightly artificial, but for 
		mu==0, I'll set it to a small cutoff of 10**-8
		
		in the limit of mu-> 0 this yields a poisson distribution
		"""
		mu1 = (mu1 + (10**-15)*(mu1==0)).astype(np.float32, copy=False)
		mu2 = (mu2 + (10**-15)*(mu2==0)).astype(np.float32, copy=False)
		
		"""
		We also have to control for n==0
		
		specifically, if n==0, we must use cdf
		if n>0, we use pmf
		"""
		
		#I need to sum over ns, since these are the expected next positions
		#n is the current position
		
		outputs = skellam.pmf(ns-n, mu1, mu2)
		
		if 0 in ns:
			outputs = (ns!=0)*outputs
			cdf_outputs = (ns==0)*skellam.cdf(-n, mu1, mu2)
			outputs += cdf_outputs
		
		#we could have underflow errors
		outputs = np.nan_to_num(outputs)

		if normalize:
			norm = np.sum(outputs, -1)
			outputs /= norm[:, None]

		#now return values
		return outputs


	def Ws(self, positions, u, absorb_at_fixation=True):
		"""
		positions is an array of shape:
		(num, len(space_dims))
		
		gives an array for the
		probability of 
		(n0, n1) -> (x0, x1)
		
		if n0 or n1 = 0, then skellam.cdf
		else skellam.pmf
		
		this returns the propagators for 0 direction and 1 direction
		and is applied in order to produce all propagators, which 
		are then saved so their values don't need to be recalculated
		"""

		#positions should have shape (num, 2)
		positions = positions.astype(np.float32)
		num = positions.shape[0]
		
		#n0s and n1s each should have shape (num,)
		n0s, n1s = np.array_split(positions, 2, -1)
		
		#b0s and b1s should have shape (num,)
		b0s, b1s = self.BirthRate(n0s, n1s)
		
		#d0s and d1s should have shape (num,)
		d0s, d1s = self.DeathRate(n0s, n1s, u)
		
		#i need ns to have shape (num, sim_size)
		ns = np.arange(self.sim_size, dtype = np.float32)*np.ones((num, self.sim_size), dtype=np.float32)
		
		#probs0 and probs1 should have shape (num, sim_size) each
		#since each (n0, n1) pair produces a full (sim_size,)
		#length propagator
		probs0 = self.get_dist(n0s, ns, b0s, d0s)
		probs1 = self.get_dist(n1s, ns, b1s, d1s)
		
		if np.isnan(probs0).any():
			print("has isnans")
			print(d0s)
		if np.isnan(probs1).any():
			print("has isnans")
			print(d1s)
		
		#now enforce absorption
		positions_list = positions.tolist()
		if absorb_at_fixation:
			for i, pos in enumerate(positions_list):
				if self.isfixed(*pos):
					"""
					Then we need to ensure absorption
					that is, the only propagation is
					(n0, n1) -> (n0, n1) with probability 1
					"""
					n0, n1 = pos
					probs0[i]*=0
					probs1[i]*=0
					#set the propagation
					probs0[i, round(n0)] = 1
					probs1[i, round(n1)] = 1
				"""
				otherwise, we dont have to do anything
				"""
					
		
		#now combine these probs arrays -- each with shape (num, sim_size)
		#into a (sim_size, sim_size, num) array
		#noting that theyre independent 
		return probs0, probs1


	def W(self, positions, u):
		"""
		based on positions and u, this function calculates the array of propagators
		
		positions has shape (-1, 2)
		
		u is a single number
		"""
		
		#first map positions to flattened indices
		n0s, n1s = np.array_split(positions, 2, -1)
		
		n0s = np.squeeze(n0s, -1)
		n1s = np.squeeze(n1s, -1)
		
		position_indices = np.ravel_multi_index([n0s, n1s], (self.sim_size, self.sim_size))
		
		#identify relevant arrays
		with h5py.File(self.propagator_file, "r") as f:
			if u==0:
				#print(np.expand_dims(position_indices, -1))
				data = f["ws_n0_n1_u0"]

				n0_propagators, n1_propagators = data[:, position_indices, :]

			elif u==1:
				data = f["ws_n0_n1_u1"]

				n0_propagators, n1_propagators = data[:, position_indices, :]

			else:
				#this case shouldnt occur
				assert(u==1 or u==0)
		
		return n0_propagators, n1_propagators

	"""
	Channel functions

	Next we define the functions used to track quantities through the backwards
	evolution. 
	These may be applied at the final point in evolution to provide us with a 
	cost function to be backpropagated. 
	These may also be applied at intermediate times to, for instance, sum up 
	the time to fixation. 
	"""
	"""
	The first two functions apply at the final time
	These are referred to as phi functions
	"""

	def fixation_func(self, position):
		if self.isfixed(*position):
			return 1
		else:
			return 0

	def extinction_func(self, position):
		n0, n1 = position
		if n0+n1 <= self.extinction_cutoff:
			return 1
		else:
			return 0

	"""
	The following functions apply for R
	that is, they are accumulated during intermediate times
	"""
		
	def expected_time_func(self, position, t, dt):
		if self.isfixed(*position):
			return 0
		else:
			return 1

	def expected_velocity_func(self, position, t, dt):
		"""
		Note that to obtain <1/T> we effectively integrate 
		
		(1/dt)-Integrate[1/t^2, {t, dt, T}] = 1/T
		
		This means we take R to be -1/t^2 in the bulk, zero at fixation
		"""
		if self.isfixed(*position):
			
			#we have hit fixation
			return 0
		else:
			#we're still in the bulk
			
			#remember we need to apple a dt cutoff
			if t==0:
				#this is below our cutoff of dt
				return 0
			else:
				return -1/(t**2)

	def propagation_to_time(self, position, t, dt):
		"""
		this is a dirac delta function in essence
		
		dirac_delta[t-propagation_time]*is_fixated
		
		
		we have two trackers:
		one to remove all the propagated individuals up to t-dt
		one to add the newly propagated bits between t-dt and t
		
		note this is useful because everything that has hit at t-dt has also hit at t
		"""
		
		#check that t is within deltaT of propagation_time
		time_bool = abs(t-self.propagation_time)<=dt
		
		#second time bool
		second_time_bool = abs(t-(self.propagation_time-2*dt))<=dt
		
		if self.isfixed(*position) and time_bool:
			
			#we have hit fixation, and we're at the right time
			return 1
		
		elif self.isfixed(*position) and second_time_bool:
			
			#we have hit fixation, and we're at the previous time
			return -1
		
		else:
			return 0

	def phi(self, position):
		"""
		phis takes a position as its input, as it is swept over J_final before any 
		backpropagating has been done.

		Here we need to supply the final cost to the optimal cost-to-go 
		as well as to each channel
		"""
		n0, n1 = position
		
		#note the first element here is for the total cost
		final_cost = np.zeros(1+len(self.channels))
		
		"""
		Compute the total cost
		"""
		
		if n0+n1 <= self.extinction_cutoff:
			total_cost = self.extinction_cost
		
		elif self.isfixed(n0, n1):
			#note the 'fixation cost' is negative
			#i.e. we want fixation, we want to avoid extinction
			total_cost = self.fixation_cost
		
		else:
			total_cost = 0
		
		final_cost[0] = total_cost
		
		"""
		Now go through each channel applicable to phi
		Not all will necessarily be tracked of course, so we go through those
		listed in channels, and only track those
		"""
		
		for i, channel in enumerate(self.channels):
			if channel in self.phi_channels_dict:
				#Note we shift i by 1 to account for the optimal ctg
				final_cost[i+1] = self.phi_channels_dict[channel](position)

		
		return final_cost
		
	def R(self, xs, t, u):
		"""
		xs is an array of points
		xs ~ [[1, 2], [3, 4], [5, 6]]
		
		R needs to have the same shape as gens: (sim_size**2, len(channels)+1)
		"""

		#xs should have shape = (slice_size, 2)
		#if the system size is small enough, slice_size = self.sim_size**2

		R_cost = np.zeros((xs.shape[0], 1+len(self.channels)))
		
		"""
		First we find the contribution to the total cost
		"""
		if self.time_cost != 0:
			outputs = []
			position_list = xs.tolist()
			for pos in position_list:
				if self.isfixed(*pos):
					outputs.append(0)
				else:
					outputs.append(self.time_cost)
			R_cost[:, 0] += outputs 
		
		if self.speed_cost != 0:
			outputs = []
			position_list = xs.tolist()
			for i, pos in enumerate(position_list):
				if self.isfixed(*pos):
					outputs.append(0)
				else:
					outputs.append(self.speed_cost*(-t**(-2)))
			R_cost[:, 0] += outputs 
		
		"""
		Now we add to the channels
		We select those of these that we want to track
		"""
		for i, channel in enumerate(self.channels):
			if channel in self.R_channels_dict:
				#enumerate through space and apply the function
				outputs = []
				position_list = xs.tolist()
				for pos in position_list:
					outputs.append(self.R_channels_dict[channel](pos, t, self.dt))

				#add to the R cost (shift i by 1 to account for the optimal ctg)
				R_cost[:, i+1] = outputs
		
		return R_cost

	def backwards_solve(self, gens=1):
		"""
		Run the backwards solver 
		"""

		self.solver.backwards_solve(gens=gens)
		self.optimal_ctg = cp.asnumpy(self.solver.jmesh)
		self.optimal_control = cp.asnumpy(self.solver.umesh)

	"""
	Define functions to format and view the data 
	"""
	def format_array(self, array):
		"""
		return a formatted version of the arrays
		The output has the correct spatial shape
		(0,0) corresponds to the upper left corner of this array. 

		in a plot of this array, the n_s axis will be along 
		the 'y' axis
		"""
		#we assume here that array is the shape of the control, or the ctg
		
		assert(array.size == self.space_dims[0]*self.space_dims[1]
			or array.size == self.space_dims[0]*self.space_dims[1]*self.N)

		if self.record_time:
			fullshape = [self.N,]+self.space_dims
		else:
			fullshape = self.space_dims
		
		#so i don't end up modifying the original array
		array_copy = cp.copy(array)

		reshaped_array = cp.reshape(array_copy, fullshape)
		
		return cp.asnumpy(reshaped_array)

	def format_for_viewing(self, array):
		"""
		return a reshaped, reoriented version of the array which is 
		formatted for viewing. This means

		(0,0) is on the lower left hand corner. 
		'y' axis => n_r
		'x' axis => n_s
		"""
		shaped_array = self.format_array(array)
		fullshape = shaped_array.shape
		axes_list = list(range(len(fullshape)))
		axes_list[-1], axes_list[-2] = axes_list[-2], axes_list[-1]
		return np.transpose(shaped_array, axes = axes_list)[::-1]


	def give_j(self):
		"""
		reshape and return J
		"""
		return self.format_array(self.optimal_ctg[..., 0])


	def give_u(self):
		"""
		reshape and return u
		"""
		return self.format_array(self.optimal_control)

	def present_u(self, t = 0):
		"""
		return u in a nice form 
		matching the style of the article
		"""
		
		if self.record_time:
			t_index = np.argmin(np.abs(self.time_mesh - t))
			u_array = self.give_u()[t_index, ...]
		else:
			u_array = self.give_u()
		
		im = plt.imshow(u_array, cmap = 'plasma')
		locs, labels = plt.xticks()
		
		length = u_array.shape[0]
		num = len(np.arange(0, length, step=20))
		max_val=padding*(20*(length//20)/length)
		
		plt.xticks(np.arange(0, length, step=20), [round(i*max_val/(num-1), 2) for i in range(num)])
		plt.yticks(np.arange(length - 20*(length//20), length, step=20), [round(i*max_val/(num-1), 2) for i in range(num)][::-1])
		
		#make a legend
		values = np.unique(u_array.ravel())
		colors = [ im.cmap(im.norm(value)) for value in values]
		patches = [ mpatches.Patch(color=colors[i], label="u = {l}".format(l=values[i]) ) for i in range(len(values)) ]
		plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
		plt.show()

	def present_j(self, t=0):
		"""
		return u in a nice form 
		matching the style of the article
		"""
		
		if self.record_time:
			t_index = np.argmin(np.abs(self.time_mesh - t))
			j_array = self.give_j()[t_index, :]
		else:
			j_array = self.give_j()
		
		scaled_j = j_array #my_scale(j_array)
		
		jplot = plt.imshow(scaled_j, cmap='cubehelix')
		locs, labels = plt.xticks()
		
		length = j_array.shape[0]
		num = len(np.arange(0, length, step=20))
		max_val=padding*(20*(length//20)/length)
		
		plt.xticks(np.arange(0, length, step=20), [round(i*max_val/(num-1), 2) for i in range(num)])
		plt.yticks(np.arange(length - 20*(length//20), length, step=20), [round(i*max_val/(num-1), 2) for i in range(num)][::-1])
		
		cbar = plt.colorbar(jplot, ticks = [0])
		plt.show()

	"""
	Functions to return expected quantities we've tracked in channels
	"""

	def prob_extinction(self):
		"""
		reshape and return extinction probability
		"""
		if 'probability of extinction' in self.channels:
			channel_index = self.channels.index('probability of extinction')+1
		else:
			return None
		
		return self.format_array(self.optimal_ctg[..., channel_index])

	def prob_fixation(self):
		"""
		reshape and return J
		"""
		if 'probability of fixation' in self.channels:
			channel_index = self.channels.index('probability of fixation')+1
		else:
			return None
		
		return self.format_array(self.optimal_ctg[..., channel_index])

	def expected_time(self):
		"""
		reshape and return extinction probability
		"""
		if 'expected time of fixation' in self.channels:
			channel_index = self.channels.index('expected time of fixation')+1
		else:
			return None
		
		return self.format_array(self.optimal_ctg[..., channel_index])

	def expected_velocity(self):
		"""
		reshape and return the expected velocity
		"""
		if 'expected velocity to fixation' in self.channels:
			channel_index = self.channels.index('expected velocity to fixation')+1
		else:
			return None
		
		
		return (1/dt)+self.format_array(self.optimal_ctg[..., channel_index])

	def propagation(self, t=0):
		"""
		reshape and return the propagation to the time 
		specified at the top of the notebook
		"""
		if record_time:
			#then t actually means something
			#otherwise, we just get to see a single snapshot of the propagation 
			#at t=0
			
			#identify what index of tmesh t corresponds to 
			t_index = np.argmin(np.abs(self.time_mesh-t))
		else:
			t_index = 0
		
		if 'propagation to fixation at time' in channels:
			channel_index = self.channels.index('propagation to fixation at time')+1
		else:
			return None
		
		return self.format_array(self.optimal_ctg[..., channel_index])
	
