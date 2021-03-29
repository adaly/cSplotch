import os
import pickle
import argparse

import numpy as np
from scipy import sparse

import pystan

parser = argparse.ArgumentParser("Generate posteriors from simulated ST data to see if cSplotch can deconvolve cell type expression.")
parser.add_argument(
	'-d', '--data-dir', type=str, required=True, help="Directory containing simulated data and covariates.")
parser.add_argument(
	'-o', '--output-dir', type=str, required=True, help='Directory in which to save output.')
parser.add_argument(
	'-g', '--gene', type=int, required=True, help="Index of gene to be processed.")
parser.add_argument(
	'-r', '--recompile', action='store_true', help='Recompile Stan model.')
args = parser.parse_args()

# Load precompiled model, if available.
model_dir = "../stan/"
if os.path.exists(os.path.join(model_dir,"comp_splotch_stan_model.pkl")) and not args.recompile:
	print("Loading precomiled model.")
	stan_model = pickle.load(open(os.path.join(model_dir,"comp_splotch_stan_model.pkl"), "rb"))
else:
	print("Compiling model from source.")
	stan_model = pystan.StanModel(file=os.path.join(model_dir,"comp_splotch_stan_model.stan"))
	pickle.dump(stan_model, open(os.path.join(model_dir,"comp_splotch_stan_model.pkl"), "wb"))

splotch_in = pickle.load(open(os.path.join(args.data_dir, 'covariates.p'), 'rb'))

# Read relevant row of input count matrix 
counts = sparse.load_npz(os.path.join(args.data_dir, 'counts.npz'))
counts_row = np.squeeze(counts[args.gene,:].toarray())

splotch_in['counts'] = counts_row

# Fit model using count data from specified gene, then save.
fit = stan_model.sampling(data=splotch_in, iter=500, chains=4)
post = fit.extract(permuted=True)

beta_level_1 = post['beta_level_1']
print("beta_level_1 mean:", beta_level_1.mean(axis=0))
print("beta_level_1 SD:", beta_level_1.std(axis=0))

np.save(os.path.join(args.output_dir, 'beta_%d_mu' % args.gene), beta_level_1.mean(axis=0))
np.save(os.path.join(args.output_dir, 'beta_%d_sigma' % args.gene), beta_level_1.std(axis=0))
np.save(os.path.join(args.output_dir, 'lambda_%d_mu' % args.gene), np.exp(beta_level_1).mean(axis=0))
np.save(os.path.join(args.output_dir, 'lambda_%d_sigma' % args.gene), np.exp(beta_level_1).std(axis=0))
