import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse

from spots import simarray


# Rosenberg average scRNA cluster profiles from mouse spinal cord
# (44 cell type clusters, 26893 genes)
scdat = "../data/aam8999_TableS10.csv"

# Common cSplotch covariates for one level-one annotation model with no 
# spatial autocorrelation and no zero inflation.
covariates = {
	'N_levels': 1,
	'N_level_1': 1,
	'N_level_2': 0,
	'N_level_3': 0,
	'level_2_mapping': np.array([], dtype=int),
	'level_3_mapping': np.array([], dtype=int),
	'zip': 0,
	'car': 0,
	'W_n': np.array([], dtype=int),
	'W_sparse': np.zeros((0,0), dtype=int),
	'D_sparse': [],
	'eig_values': []
}

# Simulate tissue comprised of a single AAR with all cell types observed
def simdata_a1(n_arrays, spots_per_array=2000, sigma=0., output_dir='.'):
	'''
	Parameters:
	----------
	n_arrays: int
		number of arrays to simulate
	spots_per_array: int
		number of spots to sample per array
	sigma: float
		standard deviation of Gaussian noise to be added to each annotation
	output_dir: path
		directory in which to save output files 'covariates.p' and 'counts.npy'
	'''
	df = pd.read_csv(scdat, sep=",", index_col=0)

	included_types = [
		'6 Astrocyte - Slc7a10',
		'10 Microglia',
		'12 Oligodendroctye Myelinating',
		'16 Alpha motor neurons'
	]
	cmat = df[included_types].values - 1
	n_celltypes = len(included_types)

	# Simulate an array with all cell types present in uniformly random combination
	spot_kwargs = [{'celltypes_present': np.arange(n_celltypes)}]

	count_mat, comp_true, aar_vec = simarray(cmat, spot_kwargs, [1.], 
		n_spots=spots_per_array*n_arrays)

	# Add Gaussian noise to composition matrix and normalize
	if sigma > 0:
		comp_obs = comp_true + np.random.normal(scale=sigma, size=comp_true.shape)
		comp_obs = np.maximum(comp_obs, np.zeros_like(comp_obs))
		comp_obs /= comp_obs.sum(axis=0)
	else:
		comp_obs = comp_true

	# Save covariate data in dictionary formatted as input to stan model
	# (Count data for given gene must be added in 'counts' field)
	covariates['N_tissues'] = n_arrays
	covariates['N_spots'] = n_arrays * [spots_per_array]
	covariates['N_covariates'] = 1
	covariates['N_celltypes'] = n_celltypes
	covariates['tissue_mapping'] = n_arrays * [1]
	covariates['size_factors'] = count_mat.sum(axis=0)
	covariates['D'] = np.ones(n_arrays * spots_per_array, dtype=int)
	covariates['E'] = np.transpose(comp_obs)

	pickle.dump(covariates, open(os.path.join(output_dir, "covariates.p"), "wb"))

	# Save counts as a sparse (CSR) matrix
	sparse_counts = sparse.csr_matrix(count_mat)
	sparse.save_npz(os.path.join(output_dir, 'counts'), sparse_counts)


# Simulate tissue comprised of a single AAR with some cell types observed as a 
# composite signature
def simdata_a1_composite(sigma=0.):
	df = pd.read_csv(scdat, sep=",", index_col=0)

	included_types = [
		'2 Unassigned',                   # 'BG'
		'3 Unassigned',                   # 'BG'
		'6 Astrocyte - Slc7a10',          # 'Glia'
		'10 Microglia',                   # 'Glia'
		'12 Oligodendrocyte Myelinating', # 'Glia'
		'16 Alpha motor neurons',         # 'Neuron'
		'17 Gamma motor neurons'          # 'Neuron'
	]

# Simulate tissue comprised of two AARs (WM and GM) with all cell types observed
def simdata_a2(sigma=0.):
	df = pd.read_csv(scdat, sep=",", index_col=0)

	included_types = [
		'5 Astrocyte - Gfap',    # White matter
		'6 Astrocyte - Slc7a10', # Gray matter
		'10 Microglia',
		'11 Oligo Mature',
		'12 Oligodendrocyte Myelinating',
		'16 Alpha motor neurons', # Gray matter
		'17 Gamma motor neurons', # Gray matter
	]

# Simulate tissue comprised of two AARs (WM and GM) with some cell types observed as a
# composite signature
def simdata_a2_composite(sigma=0.):
	pass


if __name__ == '__main__':
	simdata_a1(12, spots_per_array=2000, sigma=0., output_dir='simdata_a1_sigma_0.0')
	simdata_a1(12, spots_per_array=2000, sigma=0.1, output_dir='simdata_a1_sigma_0.1')
