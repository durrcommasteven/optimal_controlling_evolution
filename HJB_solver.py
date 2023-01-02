import numpy as np
import time 
from math import ceil
import cupy as cp 
"""
Here we define a class to backwards-solve the HJB equation

This assumes that the propagator is a product of the x propagator and 
the y propagator

"""

class HJB_solver(object):
	"""
	This class enacts backwards propagation to solve the HJB equation

	assumes that if the relevant spatial dimensions are >1, the 
	propagator is a product of the propagators in each dimension:
	W(x', y'|x, y) = W_0(x'|x) * W_1(y'|y)

	Also assumes that channels may be used to track other quantities. 
	This means that J will be of the shape (total lattice size, num_channels+1)
	only the first channel is used for deciding the optimal control
	"""
	def __init__(self, phi, R, W, xmesh, sim_size, tmesh, channels, record_time = False):
		"""
		Arguments:

		phi(position) -> final cost
		Gives the cost which is incurred at the final time at a position.
		We expect the output to be an array of shape (1+len(channels))

		R(xs, t, u) -> R_cost
		Returns an array of costs which are incurred at intermediate times. 
		We expect the output to be an array of shape (total simulation size, 1+len(channels))

		xmesh:
		An array which lists out all the points
		Expected to be of shape (-1, len(space_dims))

		sim_size: 
		the spatial lattice is expected to be in some kind of line/square/cube
		sim_size gives us the length along one dimension of the lattice.
		e.g. (250, 250)=> sim_size = 250

		tmesh:
		Array listing all the times 
		Should be one dimensional
		"""
		self.W = W #this is a function of the form:
		#self.W(positions, u, absorb_at_fixation=bool)
		#it returns a tuple of propagators -- one for each dimension

		self.phi = phi
		self.R = R
		self.tmesh = tmesh
		self.xmesh = xmesh
		self.sim_size = sim_size
		assert(self.sim_size == np.sqrt(xmesh.shape[0]))

		#now using these inputs, define other useful quantities
		self.xshape = xmesh.shape[-1] #the dimensions of the spatial shape
		self.xsize = xmesh.shape[0] #the number of spatial points
		self.tsize = tmesh.size #the number of time points
		#the number of steps is self.tsize-1
		self.dt = tmesh[1]-tmesh[0] #the dt increment

		assert self.dt == int(self.dt), "dt must be an integer"
		self.dt = int(self.dt)

		self.channels = channels
		self.record_time = record_time
		"""
		Now we create a mesh for the entire space to save the 
		optimal cost to go (along with the channels), and the optimal control
		"""
		"""
		if record_time == True, we'll be backtracking through time, recording and saving 
		all the values as we go
		"""

		if self.record_time:
			self.jmesh = cp.zeros((self.tsize, self.xsize, 1+len(self.channels)), dtype=cp.float32)
			self.umesh = cp.zeros((self.tsize, self.xsize))
		else:
			self.jmesh = cp.zeros((1, self.xsize, 1+len(self.channels)), dtype=cp.float32)
			self.umesh = cp.zeros((1, self.xsize))

		#set the umin and umax values
		self.umin, self.umax = 0, 1

		"""
		Enforce boundary conditions
		"""
		self.jmesh[-1, :] = cp.array([self.phi(x) for x in self.xmesh])
		self.index = -1

		"""
		We're also going to be tracking the changes to the control after each update
		This will allow us to enforce a cutoff and potentially speed up processing

		e.g. If the total difference from u(T-n) to u(T-n-1) (remember solve backwards in time)
		is (0<b<1)

		~ a * b**n

		Then we could end the processing after n > log(cutoff/a)/log(b)  
		"""
		self.control_difference_history = [1]
		#when is a cutoff for stopping the backpropagation
		self.percent_change_cutoff = 0#0.00001



	"""
	Now define functions for enacting and running the backwards solution
	"""

	def custom_dot_old(self, flattened_array, props0, props1):
		"""
		props0 is an array of propagators for the n0 coordinate: (num, sim_size)
		
		props1 is an array of propagators for the n0 coordinate: (num, sim_size)
		
		flatted array is a flattened array
		(sim_size, sim_size, len(channels)) -> (sim_size**2, len(channels)+1)
		
		in particular, this will be the Js array
		"""
		flattened_array, props0, props1 = cp.array(flattened_array), cp.array(props0), cp.array(props1)


		
		#dot product should have its first length equal to the first length of the props
		assert(props0.shape==props1.shape)
		dot_prod = cp.zeros((props0.shape[0], flattened_array.shape[-1]))
		
		#speed of the process of multiplying v by props    
		for i, v in enumerate(flattened_array.tolist()):
			cp.add(dot_prod, cp.array(v)*(props0[:, i//self.sim_size]*props1[:, i%self.sim_size])[:, None], out=dot_prod, casting="unsafe")
			
		return dot_prod

	def custom_dot(self, flattened_array, props0, props1):
		"""
		props0 is an array of propagators for the n0 coordinate: (num, sim_size)
		
		props1 is an array of propagators for the n1 coordinate: (num, sim_size)
		
		flatted array is a flattened array
		(sim_size, sim_size, len(channels)) -> (sim_size**2, len(channels)+1)
		
		in particular, this will be the Js array
		"""
		flattened_array, props0, props1 = cp.array(flattened_array), cp.array(props0), cp.array(props1)

		#dot product should have its first length equal to the first length of the props
		assert(props0.shape==props1.shape)
		dot_prod = cp.zeros((props0.shape[0], flattened_array.shape[-1]))
		
		#speed of the process of multiplying v by props    
		for i in range(self.sim_size):
			v = flattened_array[i*self.sim_size: (i+1)*self.sim_size, :]

			cp.add(dot_prod, 
				cp.tensordot(props0[:, [i]]*props1, v, axes = ((1), (0))),
				out=dot_prod, casting="unsafe")
			
		return dot_prod

	def expJ(self, xs, u, gens = 1):
		"""
		xs is an array of shape
		[length, 2]
		
		Given a position, and a control, u, this outputs the expected t-1 J value
		x is position
		"""
		
		input_length = xs.shape[0]
		
		if self.record_time:
			Js = self.jmesh[self.index, :]
		else:
			Js = self.jmesh[-1, :]
		
		
		current_time = self.tmesh[self.index]
		
		#W is expected to output two (input_length, sim_size) shaped arrays
		
		props0, props1 = self.W(xs, u)
		
		#print(Js.shape , "js shape")
		#print(props0.shape, 'ps shape')

		# TODO: to add more generations between steps, it could be as easy as running
		# 	self.custom_dot(Js, props0, props1) multiple times to itself, and modifying dt

		"""
		Below, we've modified this to change the number of generations between updates
		"""

		print("self dt", self.dt)
		print("Js shape:", Js.shape)

		total_js_contribution = self.custom_dot(Js, props0, props1)
		print("xs", xs)
		print("xs shape", xs.shape)
		print(total_js_contribution.shape)
		print("props shape", props0.shape, props1.shape)
		for rep in range(1, gens):
			total_js_contribution = self.custom_dot(total_js_contribution, props0, props1)
			print(total_js_contribution.shape)

		print("rs shape", self.R(xs, current_time, u).shape)

		outputs = cp.array(self.R(xs, current_time, u))*self.dt*gens + total_js_contribution

		"""
		outputs should have shape (length,)
		"""
		#lets round to get rid of machine precision errors
		round_order = 5
		return cp.around(outputs, decimals = round_order)

	def backstep(self, slice_size = None, gens=1):
		"""
		take the current time derivative, and step backward in 
		time by dt

		self.index is the current time point. This is where all
		calculations have taken place. 

		We fist shift back by 1 unit in time, and apply the results 
		of these calculations
		"""
		#check if its on the default or not
		if slice_size == None:
			"""
			we should limit the size of the imported propagators
			importing all would have size 
			(self.sim_size**2)*(2*self.sim_size)*32 bits 
			for self.sim_size=750 this is 3.4 GB

			We would like to limit this to have a maximum size of 500 MB
			"""

			MB_to_bits = 8*(10**6) #number of bits in a megabyte
			max_memory = 500*MB_to_bits

			#N*(2*self.sim_size)*32 bits  = max_memory

			N = int(max_memory/(64*self.sim_size))

			slice_size = min(self.xsize, N)
			#print('SLICE SIZE', slice_size, self.xsize)

		#uarray starts with 0, since then if there is no preference, the control
		#is zero
		#if zero is already a control option, remove it from the list to save time
		uarray = np.array(sorted(list(set([0, self.umin, self.umax]))))
		
		"""
		slice_size is the size of the slices we take from xmesh
		we break xmesh into n slices, and apply ufunc operations to 
		each sub-array. This can increase speed.
		maintaining the slice_size at a not huge value minimizes memory requirements
		"""
		
		"""
		here i assume uarray has only 2 elements
		i.e. either min or umax = 0
		"""
		assert(len(uarray)==2)
		
		j_umin_list = []
		j_umax_list = []
		
		for slice_count in range(ceil(self.xsize/slice_size)):

			#if ceil(self.xsize/slice_size)!=1:
			#	print('iterating through slices')

			start = slice_count*slice_size
			stop = min(self.xsize, (slice_count+1)*slice_size)
			
			xs = self.xmesh[start:stop]
			
			j_umin_list.append(self.expJ(xs, self.umin, gens=gens))
			j_umax_list.append(self.expJ(xs, self.umax, gens=gens))
		
		j_umin_array = cp.concatenate(j_umin_list)
		j_umax_array = cp.concatenate(j_umax_list)
		
		
		
		#for now we'll hardcode this
		#i can come up with something slicker later
		jmin_list = j_umin_array.tolist() #(sim_size**2, len(channels)+1)
		jmax_list = j_umax_array.tolist() #(sim_size**2, len(channels)+1)

		try: 
			self.diff.append([])
		except:
			self.diff = [[]]
		
		Js = []
		us = []
		for i, j1 in enumerate(jmin_list):
			j2 = jmax_list[i]
			
			if j1[0]<=j2[0]:
				#by default we assign u=0
				#therefore if j1==j2, take umin
				us.append(self.umin)
				Js.append(j1)
			else:
				us.append(self.umax)
				Js.append(j2)

			self.diff[-1].append(abs(j1[0]-j2[0]))
		
		us = cp.array(us)
		Js = cp.array(Js)
		
		#make sure we're not going too far back in time
		assert(self.index >= -self.tsize)
		self.index-=1#gens#1

		#if we want to record time
		if self.record_time:
			#set next time row of js
			self.jmesh[self.index, :] = Js 
			#update umesh
			self.umesh[self.index, :] = us

		else:
			#set next time row of js
			self.jmesh[-1, :] = Js 
			#the percent change
			percent_change = cp.mean(cp.abs(self.umesh[-1, :]-us))
			#update umesh
			self.umesh[-1, :] = us
			
			self.control_difference_history.append(percent_change)

	def backwards_solve(self, gens = 1):
		"""
		Run backwards to solve HJB
		this should save the jmesh and the umesh
		solving our problem completely
		
		For convenience, read out time as well
		"""
		
		init_time = time.time()
		starting_index = self.index
		final_index = -self.tsize
		#keep going backwards until we run out of steps
		
		while self.index > -self.tsize and self.control_difference_history[-1] >= self.percent_change_cutoff:
			#note that we need >, because during the backstep, we reduce the index by 1 
			
			print('fraction change: ', self.control_difference_history[-1])

			time_per_step = (time.time()-init_time)/(abs(self.index-starting_index)+1)
			time_remaining = (self.index + self.tsize)*time_per_step
			print("expected minutes remaining: ", time_remaining/60)
			self.backstep(gens=gens)

	def forward_step(self, dist, us):
		"""

		"""

		MB_to_bits = 8*(10**6) #number of bits in a megabyte
		max_memory = 500*MB_to_bits
		#N*(2*self.sim_size)*32 bits  = max_memory

		N = int(max_memory/(64*self.sim_size))
		slice_size = min(self.xsize, N)

		next_dist = cp.zeros(self.xsize)

		for slice_count in range(ceil(self.xsize/slice_size)):

			start = slice_count*slice_size
			stop = min(self.xsize, (slice_count+1)*slice_size)
			
			xs = self.xmesh[start:stop]
			controls = us[start:stop]
			ps = dist[start:stop]
			print("ps shape", ps.shape)

			props0u0, props1u0 = self.W(xs, 0)

			props0u0, props1u0 = cp.array(props0u0), cp.array(props1u0)

			print("props shape", props0u0.shape)

			#set those to zero which are not u=0

			mask = cp.asarray((controls==0)*ps)

			props0u0 = mask*props0u0

			hjk

		"""

		next_dist = cp.zeros(self.xsize)

		for index in range(self.xsize):

			p = dist[index]

			#print("pshape", p.shape, p)

			if p>10**-8:

				#print(type(self.xmesh))

				xs = np.array([self.xmesh[index]])

				u = us[index]

				#print("xshape", xs.shape)

				#print("ushape", u.shape)

				#find props here
				props0, props1 = self.W(xs, u)

				props0, props1 = cp.array(props0), cp.array(props1)

				#props0, props1 = cp.array(props0), cp.array(props1)

				delta = p*cp.reshape(cp.outer(props0, props1), -1)

				#print("delta", delta.shape)
				
				next_dist+=delta

		return next_dist
		"""

	def forward_propagate(self, us, initial_dist, time_steps, record = True):
		"""
		iteratively apply forward_step. time_steps times
		"""
		#first test to make sure that the shape is correct

		if initial_dist.shape != (self.xsize, 1):
			initial_dist = np.reshape(initial_dist, (self.xsize))

		history = [initial_dist]

		for rep in range(time_steps):
			print(rep)

			current_dist = history[-1]

			next_dist = self.forward_step(current_dist, us)

			if record:
				history.append(next_dist)

			else:
				history[0] = next_dist

		return history 



