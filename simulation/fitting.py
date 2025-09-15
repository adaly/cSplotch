import os
import pickle
import argparse

import numpy as np
from scipy import sparse

import stan  # Requires pystan 2.19.1.1 or earlier

parser = argparse.ArgumentParser("Generate posteriors from simulated ST data to see if cSplotch can deconvolve cell type expression.")
parser.add_argument(
	'-d', '--data-dir', type=str, required=True, help="Directory containing simulated data and covariates.")
parser.add_argument(
	'-o', '--output-dir', type=str, required=True, help='Directory in which to save output.')
parser.add_argument(
	'-g', '--gene', type=int, required=True, help="Index of gene to be processed.")
parser.add_argument(
	'-s', '--standard', action='store_true', help='Use standard Splotch model instead of cSplotch')
args = parser.parse_args()


if not args.standard:
	model_file = os.path.expanduser('~/cSplotch/stan/comp_splotch_stan_model.stan')
else:
	model_file = os.path.expanduser('~/cSplotch/stan/splotch_stan_model.stan')

splotch_in = pickle.load(open(os.path.join(args.data_dir, 'covariates.p'), 'rb'))

# Read relevant row of input count matrix 
counts = sparse.load_npz(os.path.join(args.data_dir, 'counts.npz'))
counts_row = np.squeeze(counts[args.gene,:].toarray())
splotch_in['counts'] = counts_row

# Fit model using count data from specified gene, then save.
fh = open(model_file, 'r')
model_code = fh.read()
fh.close()
posterior = stan.build(model_code, data=splotch_in, random_seed=1)
fit = posterior.sample(num_samples=250, num_chains=4)

beta_level_1 = fit['beta_level_1']
print("beta_level_1 mean:", beta_level_1.mean(axis=-1))
print("beta_level_1 SD:", beta_level_1.std(axis=-1))

if not os.path.exists(args.output_dir):
	os.mkdir(args.output_dir)

np.save(os.path.join(args.output_dir, 'beta_%d' % args.gene), beta_level_1)
np.save(os.path.join(args.output_dir, 'lambda_%d_mu' % args.gene), np.exp(beta_level_1).mean(axis=-1))
np.save(os.path.join(args.output_dir, 'lambda_%d_sigma' % args.gene), np.exp(beta_level_1).std(axis=-1))
