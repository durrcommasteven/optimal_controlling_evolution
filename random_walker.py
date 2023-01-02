import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import skellam

class RandomWalker(object):
	"""
	A class to hold the methods involved in running a random walk simulation
	"""
	def __init__(self, OptimalEvolver, newK):
		"""

		"""
		self.OptimalEvolver = OptimalEvolver
		self.oldControl = self.OptimalEvolver.give_u()

		self.gammas = OptimalEvolver.gammas 
		self.gammar = OptimalEvolver.gammar
		self.nu = OptimalEvolver.nu 
		self.mu = OptimalEvolver.mu
		self.phis = OptimalEvolver.phis
		self.phir = OptimalEvolver.phir

		#Ks
		self.newK = newK
		self.oldK = OptimalEvolver.K
		self.rw_sim_size = int(self.newK*self.OptimalEvolver.padding)
		self.scale = self.newK/self.oldK

		self.rw_fixation_cutoff = int(round(self.scale*self.OptimalEvolver.fixation_cutoff))
		self.rw_extinction_cutoff = int(round(self.scale*self.OptimalEvolver.extinction_cutoff))

		
		self.gs = self.gammas/self.newK
		self.gr = self.gammar/self.newK
		self.mu0 = self.mu/self.newK
		self.f0 = self.phis/self.newK
		self.f1 = self.phir/self.newK
		self.N = int(round(self.newK/self.nu))

		self.Gs = [self.gs, self.gr]
		self.Fs = [self.f0, self.f1]

		self.time_mesh = np.linspace(0, 1, self.N)
		self.dt = self.time_mesh[1]-self.time_mesh[0]

		#finally, upscale the old control
		self.newControl = self.convert_control()

	def rw_isfixed(self, n0, n1):
		"""
		If the population is fixed, return true, else return false
		we must scale up the original cutoff
		"""
		if n0 == 0 and n1>=self.rw_fixation_cutoff:
			return True
		return False

	def rw_isextinct(self, n0, n1):
		"""
		If the population is extinct, return true, else return false
		we must scale up the original cutoff
		"""
		if n0+n1<=self.rw_extinction_cutoff:
			return True
		return False


	def rw_BirthRate(self, n0, n1):
		"""

		"""
		birth0 = (1+self.Gs[0])*n0/(1+self.Gs[0]*n0/self.newK+self.Gs[1]*n1/self.newK)
		mutation0 = self.mu0*(n1-n0)

		birth1 = (1+self.Gs[1])*n1/(1+self.Gs[0]*n0/self.newK+self.Gs[1]*n1/self.newK)
		mutation1 = self.mu0*(n0-n1)

		growth_rate0 = (birth0+mutation0)
		growth_rate1 = (birth1+mutation1)

		return (growth_rate0, growth_rate1)

	def rw_DeathRate(self, n0, n1, u):
		"""

		"""
		death0 = (n0*(1+u*self.Fs[0]))
		death1 = (n1*(1+u*self.Fs[1]))

		return (death0, death1)    

	def approx_skellam_pmf(self, ns, mu1, mu2):
		"""
		
		"""
		
		#check that the mus are either a float, or an array witht the same shape as ns
		#assert(type(mu1)==float)
		
		a = -(mu1+mu2)-(1/2)*np.log(2*np.pi)
		b = (1/2)*np.log(mu1/mu2)
		z = 2*np.sqrt(mu1*mu2)
		zn_norm = np.sqrt(ns**2 + z**2)
		
		log_skellam = a + b*ns + zn_norm - (1/2)*np.log(zn_norm) + np.abs(ns)*np.log(z/(np.abs(ns)+zn_norm))
		
		return np.exp(log_skellam)

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
		mu1 = (mu1 + (10**-8)*(mu1==0))
		mu2 = (mu2 + (10**-8)*(mu2==0))
		
		"""
		We also have to control for n==0
		
		specifically, if n==0, we must use cdf
		if n>0, we use pmf
		"""
		
		#I need to sum over ns, since these are the expected next positions
		#n is the current position
		outputs = self.approx_skellam_pmf(ns-n, mu1, mu2)

		#outputs = skellam.pmf(ns-n, mu1, mu2)
		
		if 0 in ns:
			outputs = (ns!=0)*outputs
			cdf_outputs = (ns==0)*skellam.cdf(-n, mu1, mu2)
			outputs += cdf_outputs
		
		if normalize:
			norm = np.sum(outputs, -1)
			outputs/=norm[:, None]
			
		#now return values
		return outputs

	def step(self, n0, n1, u):
		"""
		Take a step given the previously defined 
		BirthRate(n0, n1)
		and
		DeathRate(n0, n1, u)
		"""
		
		b0, b1 = self.rw_BirthRate(n0, n1)
		if type(b0) == float and type(b1) == float:
			b0 = np.array([b0])
			b1 = np.array([b1])
		
		
		d0, d1 = self.rw_DeathRate(n0, n1, u)
		if type(d0) == float and type(d1) == float:
			d0 = np.array([d0])
			d1 = np.array([d1])
		
		#i need ns to have shape (num, sim_size)
		ns = np.arange(self.rw_sim_size, dtype = np.float32)[None, :]
		
		"""
		In order to do this rapidly for very large system sizes, 
		we will restrict the jumps to be near to the initial point
		"""
		#shift them up if theyre equal to zero
		b0=b0+(10**-8)*(b0==0)
		b1=b1+(10**-8)*(b1==0)
		b0=b0+(10**-8)*(d0==0)
		d1=d1+(10**-8)*(d1==0)
		
		probs0 = self.get_dist(n0, ns, b0, d0)
		probs1 = self.get_dist(n1, ns, b1, d1)
		
		probs0 = np.squeeze(probs0)
		probs1 = np.squeeze(probs1)
		ns = np.squeeze(ns)
		
		new_n0 = np.random.choice(ns, p=probs0)
		new_n1 = np.random.choice(ns, p=probs1)
		
		return new_n0, new_n1

	def convert_control(self):
		"""
		upscale the smaller u_array to an array corresponding to 
		K-> rw_K
		
		with rw_sim_size
		
		
		I'll do this using knn
		"""
		
		"""
		first upscale and embed the points in the larger-sized array
		"""
		
		scale = self.rw_sim_size/self.oldControl.shape[0]
		assert(scale>1)
		
		upscaled_array = np.zeros((self.rw_sim_size, self.rw_sim_size))
		
		#upscaled_array = np.zeros_like(u_array)
		
		#X are the indices
		X = []
		#y are the corresponding u's
		y = []
		
		for i, v in np.ndenumerate(self.oldControl):
			i0, i1 = i
			
			upscaled_i0 = int(round(i0*scale))
			upscaled_i1 = int(round(i1*scale))
			
			upscaled_i = (upscaled_i0, upscaled_i1)
			
			X.append(upscaled_i)
			y.append(v)
		
		neigh = KNeighborsClassifier(n_neighbors=5)
		neigh.fit(X, y)
		
		indices = np.array([index for index, value in np.ndenumerate(upscaled_array)])
		upscaled_values = neigh.predict(indices)
		upscaled_array = np.reshape(upscaled_values, (self.rw_sim_size, self.rw_sim_size))
		
		return upscaled_array

	def access_control(self, i0, i1, rw_u_array):
		"""
		access the element corresponding to (i0, i1)
		
		This is its own function just so i make sure this is done correctly 
		(is the ix, iy element [ix, iy] or [iy, ix])
		"""
		#print('hjhjh', i1, i0)
		assert(int(i0)==i0 and int(i1)==i1)
		#print(rw_u_array.shape)
		return rw_u_array[int(i0)][int(i1)]

	def TEST_access_control(self):
		for _ in range(10):
			i0 = int(np.random.rand()*self.OptimalEvolver.sim_size)
			i1 = int(np.random.rand()*self.OptimalEvolver.sim_size)
			assert((i0, i1) == tuple(self.access_control(i0, i1, self.OptimalEvolver.space_mesh)))
		return True

	def walk(self, n0, n1, steps, end_condition = None, track_control = False):
		"""
		here we enact a random walk based on a scaled up K

		end condition allows us to combine this with an end condition
		which would tell us when to terminate.
		If None, just keep going.
		"""

		if end_condition == None:
			end_condition = lambda n0, n1: False

		control_history = []
		positions = [(n0, n1)]
		for i in range(steps):
			if end_condition(*positions[-1]):
				break

			current_u=self.access_control(positions[-1][0], 
								  positions[-1][1], 
								  self.newControl)

			#print(rw_u_array.shape)
			next_step=self.step(positions[-1][0], 
						   positions[-1][1], 
						   current_u)
			positions.append(next_step)
			if track_control:
				control_history.append(current_u)
		
		if track_control:
			return positions, control_history

		return positions

	def ensemble_paths(self, n0, n1, num, max_steps):
		"""
		given a starting point (n0, n1), run num simulations 
		"""
		def end_state(n0, n1):
			if self.rw_isfixed(n0, n1):
				return 'fixed'
			elif self.rw_isextinct(n0, n1):
				return 'extinct'
			else:
				return False

		end_condition = lambda n0, n1: bool(end_state(n0, n1))

		fixed_paths = []
		extinct_paths = []
		other_paths = []
		control_histories_fixed = []
		control_histories_extinct = []
		control_histories_other = []


		for n in range(num):
			path, control_history = self.walk(n0, n1, max_steps, end_condition = end_condition, track_control = True)
			final_state = end_state(*path[-1])

			if final_state == 'fixed':
				fixed_paths.append(path)
				control_histories_fixed.append(control_history)

			elif final_state == 'extinct':
				extinct_paths.append(path)
				control_histories_extinct.append(control_history)

			else:
				other_paths.append(path)
				control_histories_other.append(control_history)

			

		return fixed_paths, extinct_paths, other_paths, control_histories_fixed, control_histories_extinct, control_histories_other