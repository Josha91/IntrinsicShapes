"""
This script can be used to generate the expected distribution 
of axis ratios, given some intrinsic distribution of ellipticity
and triaxiliaty parameters

USAGE:
params = [mu_T,mu_E,sig_E,sig_T,mu_o,sig_o,fob]
obj = observeQ(params)

obj.f_interp then contains the function f(q' | params) where
q' is the observed axis ratio. 

mu_T,sig_T determine the triaxiality of the triaxial component,
muE, sig_E determine its elliplticity.
mu_o,sig_o == mu_E,sig_E of an oblate component. 
fob determines the relative population of triaxial and oblate objects, such that
fob = 1 is fully oblate and fob=0 is fully triaxial
"""

import numpy as np
from random import uniform
import IPython
#from scipy.optimize import least_squares
import matplotlib.pyplot as plt
#import seaborn as sns
import random
import time
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.integrate import simps
from scipy.stats import binned_statistic


class observeQ():
	def __init__(self,params):
		"""
		This object and the functions within it 
		are used to generate random angles and draw
		from the user-provided intrinsic shape distributions.
		It computes the distribution of axis ratios that will result.
		------
		mu_T, mu_E - the average ellipticity and triaxiality
		sig_T,sig_E - standard deviation for E and T
		mu_o,sig_o - intrinsic axis ratio & std for oblate component
		fob - fraction of oblate galaxies

		For definitions, see Chang+2013 
		"""
		self.mu_T,self.mu_E,self.sig_T,self.sig_E,self.mu_o,self.sig_o,self.fob = params
		start = time.time()
		self.sample_viewing() #Step 1: sample viewing angles randomly. 
		self.get_components() #Step 2: Compute Triax & Obl pdfs
		self.get_observed_qp() #Step 3: for every viewing angle, compute q' on a (coarse) grid in p,q
		self.make_fq() #Final step: create an output function 
		self.runtime = time.time()-start

	def sample_viewing(self,N=1e4):
		"""Draw N random 2-dimensional angles"""
		self.N = int(N)
		rand,rand2 = [],[]
		for _ in np.arange(N):
			rand.append(uniform(0,1))
			rand2.append(uniform(0,1))
		rand = np.asarray(rand)
		rand2 = np.asarray(rand2)
		#theta = np.arcsin(2*rand-1) + np.pi/2.
		theta = np.arcsin(2*rand-1)+np.pi/2.#np.arcsin(rand)+np.pi/2.
		phi = rand2*np.pi*2
		self.theta,self.phi = theta,phi#np.degrees(theta),np.degrees(phi)

		#Some geometric functions
		self.cth = np.cos(self.theta)
		self.cth2, self.sth2 = self.cth**2., 1 - self.cth**2.
		self.cph = np.cos(self.phi)
		self.cph2,self.sph2 = self.cph**2.,1-self.cph**2.

	def get_components(self):
		self.ptrix = self.get_ptrix()
		self.pobl = self.get_pobl()

	def get_ptrix(self):
		"""
		Triaxial pdf, given input parameters. 
		"""
		if not hasattr(self,'theta'): self.sample_viewing()
		nx = 100
		x1,x2 = np.linspace(0,1,nx),np.linspace(0,1,nx)
		psT,qsT = self._sample_triaxiality(10000000)
		H,x_edges,y_edges = np.histogram2d(psT,qsT,bins=(np.linspace(0,1,101),np.linspace(0,1,101)),normed=True)
		xbin = np.array([(x_edges[i]+x_edges[i+1])/2 for i in np.arange(x_edges.size-1)])
		xbin[0],xbin[-1] = 0,1
		ybin = np.array([(y_edges[i]+y_edges[i+1])/2 for i in np.arange(y_edges.size-1)])
		ybin[0],ybin[-1] = 0,1
		return xbin,ybin,H

	def _sample_triaxiality(self,N):
		Es,Ts = self._sample_TE(N)
		qs = 1-Es #'gamma' in Chang+13
		useind = np.logical_and(qs>=0,qs<=1)
		qs = qs[useind]
		Es, Ts = Es[useind],Ts[useind]
		ps = np.sqrt(1-Ts*(1-qs**2.)) #'beta' in Chang+13
		use = ~np.isnan(ps)
		ps,qs = ps[use],qs[use]
		return ps, qs 

	def _sample_TE(self,N):
		rand_E = np.random.normal(self.mu_E,self.sig_E,N)
		rand_T = np.random.normal(self.mu_T,self.sig_T,N)
		return rand_E,rand_T

	def get_pobl(self):
		"""
		Compute oblate pdf
		"""
		nx,ny = 1000,1
		x1,x2 = np.linspace(0,1,nx),np.linspace(0,1,nx)
		x2 = np.ones(ny)
		qs = norm(self.mu_o,self.sig_o).pdf(x1)
		ps = np.ones_like(qs)
		test = np.zeros((nx,ny))
		test[:,-1] = qs
		return x1,x2,test.T.flatten()

	def get_observed_qp(self):
		"""
		Given the random angles, compute the observed axis ratio on a grid of 
		intrinsic ratios p and q
		"""
		self.ps1,self.qs1 = np.linspace(0,1,40), np.linspace(0,1,40)

		#ps2, qs2 = np.meshgrid(ps1,qs1)
		#ps2 = ps2.flatten()
		#qs2 = qs2.flatten()
		#zipp = [(ps2,qs2) for _ in np.arange(self.N)]

		self.A2a4_func = lambda p,q,cth2,sph2,cph2,sth2: p**2.*cth2 + (sph2+p**2*cph2)*q**2.*sth2
		self.Ba2_func = lambda p,q,cth2,sph2,cph2,sth2:(cph2*cth2+sph2)+p**2.*(sph2*cth2+cph2) + q**2.*sth2

		#this stores the projected axis ratio on a grid of p,q, for every viewing angles. 
		#Shape is NxM with N the number of angles and M the gridpoitns of p,q
		results = np.zeros((self.sth2.size,self.ps1.size*self.qs1.size))
		for i in np.arange(self.sth2.size):
			A2a4 = self.A2a4_func(self.ps1[:,None],self.qs1[None,:],self.cth2[i],self.sph2[i],self.cph2[i],self.sth2[i])
			Ba2 = self.Ba2_func(self.ps1[:,None],self.qs1[None,:],self.cth2[i],self.sph2[i],self.cph2[i],self.sth2[i])
			nom = Ba2 - np.sqrt(Ba2**2.-4*A2a4)
			denom = Ba2 + np.sqrt(Ba2**2.-4*A2a4)
			qp = np.sqrt(nom/denom)
			results[i,:] = qp.flatten()
		#Sped up a factor 50 wrt the previous computation...!!!

		#self.qp_out,self.p_used,self.q_used= results.flatten(),np.repeat(ps2.flatten(),self.sth2.size),np.repeat(qs2.flatten(),self.sth2.size)
		self.qp_out = results.flatten().reshape((self.sth2.size,self.ps1.size,self.qs1.size))

	def make_fq(self):
		"""
		The output function, i.e. f(q' | params). Given the parameters, what is the probability of observing q?
		"""
		qbins = np.linspace(0,1,80)
		#Interpolate for (p,q) and use f(p,q) to get the final, posterior distribution f(q' | params)
		prob_tot = np.zeros_like(qbins[1:])
		for i in np.arange(self.N,dtype=int):
			f = RBS(self.ps1,self.qs1,self.qp_out[i,:,:])

			#Triaxial component
			tmp = f(self.ptrix[1],self.ptrix[0])
			prob_tmp = binned_statistic(tmp.flatten(),self.ptrix[2].flatten(),statistic='sum',bins=qbins)[0]
			#binned = np.digitize(tmp.flatten(),bins=qbins)
			#prob_tmp = np.array([np.sum(self.ptrix[2].flatten()[binned==j]) for j in np.arange(qbins.size)])

			#Oblate component 
			tmp2 = f(self.pobl[1],self.pobl[0])
			#binned = np.digitize(tmp2.flatten(),bins=qbins)
			#prob_tmp2 = np.array([np.sum(self.pobl[2].flatten()[binned==j]) for j in np.arange(qbins.size)])
			prob_tmp2 = binned_statistic(tmp2.flatten(),self.pobl[2].flatten(),statistic='sum',bins=qbins)[0]
			#Combination, using fob
			prob_tot += prob_tmp*(1-self.fob) + prob_tmp2*self.fob

		bins_q = (qbins[1:]+qbins[0:-1])/2
		prob_tot /= simps(prob_tot,bins_q)

		self.f_interp = interp1d(bins_q,prob_tot,bounds_error=False,fill_value=0)


def test():
	#mu_T,mu_E,sig_T,sig_E,mu_o,sig_o,fob
	params = np.array([0.6,0.45,0.16,0.23,0.08,0.41,0.0])
	#params = np.array([0.2,0.2,0.07,0.6,0.3,0.2,1.0])
	comp = observeQ(params)
	IPython.embed()

if __name__ == "__main__":
	start=  time.time()
	test()
	#obl1 = pq_given_TQ(11.0,z=0,triaxial_only=True,sf=True) 