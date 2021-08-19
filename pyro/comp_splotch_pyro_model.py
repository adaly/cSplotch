import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser

import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

from splotch.utils import read_rdump


# Based on the sparse_car_lpdf implementation in Stan by Max Joseph:
#   https://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
class SparseCARDist(dist.TorchDistribution):
	support = constraints.real
	has_rsample = False

	def __init__(self, tau, alpha, W_sparse, D_sparse, eig_values, W_n):
		'''
		Parameters:
		----------
		tau: float
			spatial precision parameter
		alpha: float
			spatial dependence parameter, bounded between 0 (spatial independence) and 1 (intrinsic conditional autoregressive)
		W_sparse: (W_n, 2) binary array
			sparse representation of the adjacency matrix between spots
		D_sparse: (n_spots,) int array
			number of neighbors for each spot
		eig_values: (W_n,) float array
			eigenvalues of D^{-1/2}*W*D^{-1/2}
		W_n: int
			number of adjacent pairs
		'''
		self.tau = tau
		self.alpha = alpha

		self.W_sparse = W_sparse
		self.D_sparse = D_sparse
		self.eig_values = eig_values

		self.W_n = W_n
		self.n_spots = D_sparse.shape[0]

		self._event_shape = torch.Size([self.n_spots])

	# DUMMY FUNCTION: only required for model intitialization.
	def sample(self, sample_shape=torch.Size()):
		return torch.zeros(self.n_spots).double()

	# Log probability density of CAR prior up to additive constant:
	#   log(phi | tau, alpha) ~= n/2(log(tau)) + 1/2(sum(1 - alpha * lambda)) - 1/2(phi^T * Sigma^{-1} * phi)
	def log_prob(self, phi):
		phit_D = (phi * self.D_sparse).double()     # phi^T * D
		phit_W = torch.zeros(self.n_spots).double() # phi^T * W

		for i in range(self.W_n):
			phit_W[self.W_sparse[i,0]] += phi[self.W_sparse[i,1]]
			phit_W[self.W_sparse[i,1]] += phi[self.W_sparse[i,0]]

		ldet_terms = 1 - (self.alpha * self.eig_values)

		return 0.5 * (self.n_spots * torch.log(self.tau) 
			          + torch.sum(ldet_terms) 
			          - self.tau * (torch.dot(phit_D, phi) - self.alpha * torch.dot(phit_W, phi)))


def splotch_model(counts, size_factors, annotations, spots_per_tissue, tissue_mapping, N_covariates,
	W_sparse, D_sparse, eig_values, W_n,
	N_level_1=1, N_level_2=0, N_level_3=0, level_2_mapping=None, level_3_mapping=None):
	'''
	Parameters:
	----------
	counts: (n_spots,) tensor of int
		integer counts for gene in question at each spot
	size_factors: (n_spots,) tensor of float
		size factors for each spot (expressed as fractions of median sequencing depth across all spots)
	annotations: (n_spots,) array of int
		integers indicating annotation class of each spot
	spots_per_tissue: (n_tissues,) array of int
		number of spots in each tissue.
	tissue_mapping: (n_tissues) array of int
		maps each tissue to an index in the range [0, N_level_X-1], where X is the lowest model level (1>2>3).
		Indicates tissue identity.
	N_covariates: int
		number of unique annotation categories
	N_level_1: int
		number of level 1 conditions (must be 1 or greater)
	N_level_2: int
		number of level 2 conditions (must be 1 or greater if N_level_3 > 0)
	N_level_3: int
		number of level 3 conditions
	level_2_mapping: (N_level_2,) array of int
		maps each level 2 condition to index of a level 1 condition (e.g., genotype-sex to genotype)
	level_3_mapping: (N_level_3,) array of int
		maps each level 3 condition to index of a level 2 condition (e.g., individual to genotype-sex)
	'''

	assert len(counts) == len(annotations), 'all spots must be annoated!'

	cumsum_spots = np.zeros(len(spots_per_tissue) + 1, dtype=int)
	cumsum_spots[1:] = np.cumsum(spots_per_tissue)

	### Characteristic expression ###

	# Level 1 priors: all beta_level_1 ~ N(0, 2)
	beta_level_1 = torch.zeros((N_level_1, N_covariates))
	for i in range(N_level_1):
		for j in range(N_covariates):
			beta_level_1[i,j] = pyro.sample('beta_level_1.%d.%d' % (i,j), dist.Normal(0,2))

	bottom_level_beta = beta_level_1

	# Level 2 priors: all beta_level_2 ~ N(beta_level_1, sigma_level_2)
	if N_level_2 > 0:
		beta_level_2 = torch.zeros((N_level_2, N_covariates))
		sigma_level_2 = pyro.sample('sigma_level_2', dist.HalfNormal(1))

		for i in range(N_level_2):
			for j in range(N_covariates):
				beta_level_2[i,j] = pyro.sample('beta_level_2.%d.%d' % (i,j), 
					dist.Normal(beta_level_1[level_2_mapping[i], j], sigma_level_2))

		bottom_level_beta = beta_level_2

	# Level 3 priors: all beta_level_3 ~ N(beta_level_2, sigma_level 3)
	if N_level_3 > 0:
		beta_level_3 = torch.zeros((N_level_3, N_covariates))
		sigma_level_3 = pyro.sample('sigma_level_3', dist.HalfNormal(1))

		for i in range(N_level_3):
			for j in range(N_covariates):
				beta_level_3[i,j] = pyro.sample('beta_level_3.%d.%d' % (i,j),
					dist.Normal(beta_level_2[level_3_mapping[i], j], sigma_level_3))

		bottom_level_beta = beta_level_3

	### Spatial autocorrelation ###
	alpha = pyro.sample('alpha', dist.Uniform(0,1))
	tau = 1 / pyro.sample('tau_inv', dist.Gamma(1,1))

	sparse_car_prior = SparseCARDist(tau, alpha, W_sparse, D_sparse, eig_values, W_n)
	psi = pyro.sample('psi', sparse_car_prior)

	### Spot-level variation ###
	sigma = pyro.sample('sigma', dist.HalfNormal(0.3))
	epsilon = pyro.sample('epsilon', dist.MultivariateNormal(torch.zeros(len(counts), dtype=torch.double), sigma * torch.eye(len(counts), dtype=torch.double)))

	### Likelihood calculation ###

	# Calculate expression rate for each spot
	log_lambda = torch.zeros(len(annotations))

	for i in range(len(spots_per_tissue)):
		for j in range(cumsum_spots[i], cumsum_spots[i+1]):
			log_lambda[j] = bottom_level_beta[tissue_mapping[i], annotations[j]] + psi[j] + epsilon[j]

	# Zero inflation component
	theta = pyro.sample('theta', dist.Beta(1,2))

	# Likelihood evaluation (vectorized)
	expr_rate = log_lambda.exp() * size_factors
	with pyro.plate('data'):
		pyro.sample('obs', dist.ZeroInflatedPoisson(expr_rate, gate=theta), obs=counts)


if __name__ == '__main__':
	parser = ArgumentParser('Pyro implementation of cSplotch')
	parser.add_argument('-i', '--input-file', required=True, type=str,
		help='Path to Rdump file containing cSplotch inputs (from generate_splotch_inputs).')
	parser.add_argument('-o', '--output-file', required=False, type=str,
		help='Path to save CSV summary of model output.')
	parser.add_argument('-n', '--num-samples', required=False, type=int, default=500,
		help='Number of samples to draw in posterior inference.')
	parser.add_argument('-c', '--num-chains', required=False, type=int, default=4,
		help='Number of chains to run in posterior inference.')
	parser.add_argument('-p', '--progress-bar', required=False, action='store_true',
		help='Display a progress bar in the console.')
	args = parser.parse_args()


	# Read in input data
	data_dict = read_rdump(args.input_file)

	count_arr = torch.tensor(data_dict['counts'], dtype=int)
	annot_arr = data_dict['D'].astype(int) - 1
	depth_arr = torch.tensor(data_dict['size_factors'])

	spots_per_tissue = data_dict['N_spots'].astype(int)

	N_covariates = int(data_dict['N_covariates'])
	N_level_1 = int(data_dict['N_level_1'])
	N_level_2 = int(data_dict['N_level_2'])
	N_level_3 = int(data_dict['N_level_3'])

	tissue_mapping = data_dict['tissue_mapping'].astype(int) - 1
	level_2_mapping = data_dict['level_2_mapping'].astype(int) - 1
	level_3_mapping = data_dict['level_3_mapping'].astype(int) - 1

	W_sparse = torch.tensor(data_dict['W_sparse'].astype(int)) - 1
	D_sparse = torch.tensor(data_dict['D_sparse'])
	eig_values = torch.tensor(data_dict['eig_values'])
	W_n = int(data_dict['W_n'])

	# Clear parameter store before inference
	pyro.clear_param_store()

	# Logging options
	if args.progress_bar:
		log_fn = None
	else:
		def log_fn(kernel, samples, stage, i):
			if i % 10 == 0:
				print('\t%s [%d/%d] (%ds)' % (stage, i, args.num_samples, time.time()-start_time), flush=True)

	# Set up NUTS kernel and MCMC sampler
	kernel = pyro.infer.NUTS(splotch_model, max_tree_depth=7)
	mcmc = pyro.infer.MCMC(kernel, 
		num_samples=args.num_samples,
		warmup_steps=args.num_samples,
		num_chains=args.num_chains,
		disable_progbar = not args.progress_bar, hook_fn=log_fn)

	# Perform sampling and check results
	start_time = time.time()
	mcmc.run(count_arr, depth_arr, annot_arr, spots_per_tissue, tissue_mapping, N_covariates,
		W_sparse, D_sparse, eig_values, W_n,
		N_level_1, N_level_2, N_level_3, level_2_mapping, level_3_mapping)
	run_time = time.time() - start_time

	print('Inference ran for %.3f minutes' % (run_time / 60))
	print(mcmc.summary())

	post_df = pd.DataFrame(mcmc.get_samples())
	print(post_df.head())

	if args.output_file is not None:
		post_df.to_csv(args.output_file, sep=',')
