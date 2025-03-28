import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from spots import simarray, simarray_cells


# Rosenberg average snRNA cluster profiles from mouse spinal cord
# (44 cell type clusters, 26893 genes)
scdat = "../data/aam8999_TableS10.csv"

# Rosenberg individual snRNA-seq profiles from mouse spinal cord
cells_file = '../data/GSM3017261_20000_SC_nuclei.h5ad'

# Common cSplotch covariates for one level-one annotation model with no 
# spatial autocorrelation and no zero inflation.
covariates = {
	'N_levels': 1,
	'N_level_1': 1,
	'N_level_2': 0,
	'N_level_3': 0,
	'level_2_mapping': np.array([], dtype=int),
	'level_3_mapping': np.array([], dtype=int),
	'zi': 0,
	'car': 0,
	'W_n': np.array([], dtype=int),
	'W_sparse': np.zeros((0,0), dtype=int),
	'D_sparse': [],
	'eig_values': []
}

# Simulate tissue comprised of a single AAR with all cell types observed
def simdata_a1(n_arrays, spots_per_array=2000, sigma=0., output_dir='.', 
	sim_mode='clusters', comp_mode='cells'):
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
	sim_mode: 'cells' or 'clusters'
		whether to use individual cells or average cluster profiles to simulate spots
	comp_mode: 'cells' or 'counts'
		for cell simulations, whether to use cells/type or counts/type to characterize 
		spot composition (counts/type is more accurate but less realistic to observe).
	'''

	if sim_mode not in ['cells', 'clusters']:
		raise ValueError('sim_mode must be either "cells" or "clusters"')
	if comp_mode not in ['cells', 'counts']:
		raise ValueError('comp_mode must be either "cells" or "counts"')

	included_types = [
		'6 Astrocyte - Slc7a10',
		'10 Microglia',
		'12 Oligodendrocyte Myelinating',
		'16 Alpha motor neurons'
	]
	n_celltypes = len(included_types)

	# Simulate spots by drawing from characteristic distributions
	if sim_mode == 'clusters':
		covariates['nb'] = 0

		df = pd.read_csv(scdat, sep=",", index_col=0)
		cmat = df[included_types].values - 1
		
		# Simulate arrays with all included cell types present in uniformly random proportions
		spot_kwargs = [{'celltypes_present': np.arange(n_celltypes)}]

		count_mat, comp_true, aar_vec = simarray(cmat, spot_kwargs, [1.], 
			n_spots=spots_per_array*n_arrays)

	# Simulate spots by combining individual cells
	else:
		covariates['nb'] = 1

		adat = sc.read_h5ad(cells_file)
		sc.pp.normalize_total(adat, target_sum=1000)  # scale all cells to have ~1000 total UMIs

		# Filter AnnData to only include specified cell types
		adat = adat[adat.obs['sc_cluster'].isin(included_types)]

		# Simulate arrays with all included cell types present in uniformly random proportions
		spot_kwargs = [{'celltypes_present': included_types, 'ncells_in_spot':10}]

		count_mat, comp_df, aar_vec = simarray_cells(adat, 'sc_cluster', spot_kwargs,
			n_spots=spots_per_array*n_arrays, comp_mode=comp_mode)
		
		# Ensure ordering of celltypes in comp matrix matches that of included_celltypes
		comp_true = np.vstack([comp_df.loc[c,:].values for c in included_types])

		# Account for drop-out events in single-cell data with ZIP
		#covariates['zip'] = 1

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

	covariates['beta_prior_mean'] = np.zeros(n_celltypes)
	covariates['beta_prior_std'] = 2 * np.ones(n_celltypes)

	pickle.dump(covariates, open(os.path.join(output_dir, "covariates.p"), "wb"))

	# Save counts as a sparse (CSR) matrix
	sparse_counts = sparse.csr_matrix(count_mat)
	sparse.save_npz(os.path.join(output_dir, 'counts'), sparse_counts)


# Simulate tissue comprised of a single AAR with some cell types observed as a 
# composite signature
def simdata_a1_composite(n_arrays, spots_per_array=2000, sigma=0., output_dir='.', 
	sim_mode='clusters', comp_mode='cells'):

	if sim_mode not in ['cells', 'clusters']:
		raise ValueError('sim_mode must be either "cells" or "clusters"')
	if comp_mode not in ['cells', 'counts']:
		raise ValueError('comp_mode must be either "cells" or "counts"')

	included_types = [
		'2 Unassigned',                   # 'BG'
		'3 Unassigned',                   # 'BG'
		'6 Astrocyte - Slc7a10',          # 'Glia'
		'10 Microglia',                   # 'Glia'
		'12 Oligodendrocyte Myelinating', # 'Glia'
		'16 Alpha motor neurons',         # 'Neuron'
		'17 Gamma motor neurons'          # 'Neuron'
	]
	n_celltypes = len(included_types)

	if sim_mode == 'clusters':
		covariates['nb'] = 0

		df = pd.read_csv(scdat, sep=",", index_col=0)
		cmat = df[included_types].values - 1

		# Simulate an array with all cell types present in uniformly random combination
		spot_kwargs = [{'celltypes_present': np.arange(n_celltypes)}]

		count_mat, comp_true, aar_vec = simarray(cmat, spot_kwargs, [1.], 
			n_spots=spots_per_array*n_arrays)

	else:
		covariates['nb'] = 1

		adat = sc.read_h5ad(cells_file)
		sc.pp.normalize_total(adat, target_sum=1000)  # scale all cells to have ~1000 total UMIs

		# Filter AnnData to only include specified cell types
		adat = adat[adat.obs['sc_cluster'].isin(included_types)]

		# Simulate arrays with all included cell types present in uniformly random proportions
		spot_kwargs = [{'celltypes_present': included_types, 'ncells_in_spot':10}]

		count_mat, comp_df, aar_vec = simarray_cells(adat, 'sc_cluster', spot_kwargs,
			n_spots=spots_per_array*n_arrays, comp_mode=comp_mode)
		
		# Ensure ordering of celltypes in comp matrix matches that of included_celltypes
		comp_true = np.vstack([comp_df.loc[c,:].values for c in included_types])

	# Create three composite profiles by combining similar scRNA signals
	comp_obs = np.zeros((3, comp_true.shape[1]))
	comp_obs[0,:] = comp_true[0,:] + comp_true[1,:] 
	comp_obs[1,:] = comp_true[2,:] + comp_true[3,:] + comp_true[4,:]
	comp_obs[2,:] = comp_true[5,:] + comp_true[6,:]

	# Add Gaussian noise to composition data and normalize
	if sigma > 0:
		comp_obs = comp_true + np.random.normal(scale=sigma, size=comp_true.shape)
		comp_obs = np.maximum(comp_obs, np.zeros_like(comp_obs))
		comp_obs /= comp_obs.sum(axis=0)

	# Save covariate data in dictionary formatted as input to stan model
	# (Count data for given gene must be added in 'counts' field)
	covariates['N_tissues'] = n_arrays
	covariates['N_spots'] = n_arrays * [spots_per_array]
	covariates['N_covariates'] = 1
	covariates['N_celltypes'] = 3
	covariates['tissue_mapping'] = n_arrays * [1]
	covariates['size_factors'] = count_mat.sum(axis=0)
	covariates['D'] = np.ones(n_arrays * spots_per_array, dtype=int)
	covariates['E'] = np.transpose(comp_obs)

	covariates['beta_prior_mean'] = np.zeros(comp_obs.shape[0])
	covariates['beta_prior_std'] = 2 * np.ones(comp_obs.shape[0])

	pickle.dump(covariates, open(os.path.join(output_dir, "covariates.p"), "wb"))

	# Save counts as a sparse (CSR) matrix
	sparse_counts = sparse.csr_matrix(count_mat)
	sparse.save_npz(os.path.join(output_dir, 'counts'), sparse_counts)


# Simulate tissue comprised of two AARs (WM and GM) with some cell types observed as a
# composite signature
def simdata_a2_composite(n_arrays, spots_per_array=2000, sigma=0., output_dir='.', 
	sim_mode='clusters', comp_mode='cells'):

	if sim_mode not in ['cells', 'clusters']:
		raise ValueError('sim_mode must be either "cells" or "clusters"')
	if comp_mode not in ['cells', 'counts']:
		raise ValueError('comp_mode must be either "cells" or "counts"')

	'''included_types = [
		'2 Unassigned',          # BG (WM + GM)
		'3 Unassigned',          # BG (WM + GM)
		'5 Astrocyte - Gfap',    # Glia (WM)
		'6 Astrocyte - Slc7a10', # Glia (GM)
		'8 Endothelial',         # BG (WM + GM)
		'9 VLMC',                # BG (WM)
		'10 Microglia',          # Glia (WM + GM)
		'11 Oligo Mature',       # Glia (WM + GM?)
		'12 Oligodendrocyte Myelinating', # Glia (WM)
		'16 Alpha motor neurons', # Neuron (GM)
		'17 Gamma motor neurons', # Neuron (GM)
	]'''
	included_types = [
		'5 Astrocyte - Gfap',    # Glia (WM)
		'6 Astrocyte - Slc7a10', # Glia (GM)
		'10 Microglia',          # Glia (WM + GM)
		'11 Oligo Mature',       # Glia (WM + GM?)
		'12 Oligodendrocyte Myelinating', # Glia (WM)
		'16 Alpha motor neurons', # Neuron (GM)
		'17 Gamma motor neurons', # Neuron (GM)
	]
	n_celltypes = len(included_types)

	if sim_mode == 'clusters':
		covariates['nb'] = 0

		df = pd.read_csv(scdat, sep=",", index_col=0)
		cmat = df[included_types].values - 1	

		'''
		spot_kwargs = [
			{'celltypes_present': np.array([0,1,2,4,5,6,7,8])},  # WM
			{'celltypes_present': np.array([0,1,3,4,6,7,9,10])}  # GM
		]'''
		spot_kwargs = [
			{'celltypes_present': np.array([0,2,3,4])},   # WM
			{'celltypes_present': np.array([1,2,3,5,6])}, # GM
		]

		count_mat, comp_true, aar_vec = simarray(cmat, spot_kwargs, [.5, .5], 
			n_spots=spots_per_array*n_arrays)

	else:
		covariates['nb'] = 1
		
		adat = sc.read_h5ad(cells_file)
		sc.pp.normalize_total(adat, target_sum=1000)  # scale all cells to have ~1000 total UMIs

		# Filter AnnData to only include specified cell types
		adat = adat[adat.obs['sc_cluster'].isin(included_types)]

		'''
		spot_kwargs = [
			{'celltypes_present': ['2 Unassigned', '3 Unassigned', '5 Astrocyte - Gfap', '8 Endothelial', 
								   '9 VLMC', '10 Microglia', '11 Oligo Mature', '12 Oligodendrocyte Myelinating'],
			 'ncells_in_spot': 10},  # WM
			{'celltypes_present': ['2 Unassigned', '3 Unassigned', '6 Astrocyte - Slc7a10', '8 Endothelial', 
			                       '10 Microglia', '11 Oligo Mature', '16 Alpha motor neurons', '17 Gamma motor neurons'],
			 'ncells_in_spot': 10}  # GM
		]'''
		spot_kwargs = [
			{'celltypes_present': ['5 Astrocyte - Gfap', '10 Microglia', '11 Oligo Mature', '12 Oligodendrocyte Myelinating'],
			 'ncells_in_spot': 10},
			{'celltypes_present': ['6 Astrocyte - Slc7a10', '10 Microglia', '11 Oligo Mature', 
								   '16 Alpha motor neurons', '17 Gamma motor neurons'],
			 'ncells_in_spot': 10}
		]

		count_mat, comp_df, aar_vec = simarray_cells(adat, 'sc_cluster', spot_kwargs,
			n_spots=spots_per_array*n_arrays, comp_mode=comp_mode)

		# Ensure ordering of celltypes in comp matrix matches that of included_celltypes
		comp_true = np.vstack([comp_df.loc[c,:].values for c in included_types])

	# Create 3 composite profiles by combining similar scRNA-seq profiles
	'''comp_obs = np.zeros((3, comp_true.shape[1]))
	comp_obs[0,:] = comp_true[0] + comp_true[1] + comp_true[4] + comp_true[5]
	comp_obs[1,:] = comp_true[2] + comp_true[3] + comp_true[6] + comp_true[7] + comp_true[8]
	comp_obs[2,:] = comp_true[9] + comp_true[10]
	'''
	# Create 2 composite profiles by combining similar scRNA-seq profiles
	comp_obs = np.zeros((2, comp_true.shape[1]))
	comp_obs[0,:] = comp_true[0] + comp_true[1] + comp_true[2] + comp_true[3] + comp_true[4]
	comp_obs[1,:] = comp_true[5] + comp_true[6]

	# Add Gaussian noise to composition data and normalize
	if sigma > 0:
		comp_obs = comp_true + np.random.normal(scale=sigma, size=comp_true.shape)
		comp_obs = np.maximum(comp_obs, np.zeros_like(comp_obs))
		comp_obs /= comp_obs.sum(axis=0)

	# Save covariate data in dictionary formatted as input to stan model
	# (Count data for given gene must be added in 'counts' field)
	covariates['N_tissues'] = n_arrays
	covariates['N_spots'] = n_arrays * [spots_per_array]
	covariates['N_covariates'] = 2
	#covariates['N_celltypes'] = 3
	covariates['N_celltypes'] = 2
	covariates['tissue_mapping'] = n_arrays * [1]
	covariates['size_factors'] = count_mat.sum(axis=0)
	covariates['D'] = aar_vec + 1
	covariates['E'] = np.transpose(comp_obs)

	covariates['beta_prior_mean'] = np.zeros(comp_obs.shape[0])
	covariates['beta_prior_std'] = 2 * np.ones(comp_obs.shape[0])

	pickle.dump(covariates, open(os.path.join(output_dir, "covariates.p"), "wb"))

	# Save counts as a sparse (CSR) matrix
	sparse_counts = sparse.csr_matrix(count_mat)
	sparse.save_npz(os.path.join(output_dir, 'counts'), sparse_counts)


if __name__ == '__main__':
	#simdata_a1(12, spots_per_array=2000, sigma=0, output_dir='simdata_cells/simdata_a1_s0_Ecells', sim_mode='cells', comp_mode='cells')	
	#simdata_a1(12, spots_per_array=2000, sigma=0, output_dir='simdata_cells/simdata_a1_s0_Ecounts', sim_mode='cells', comp_mode='counts')

	#simdata_a1_composite(12, spots_per_array=2000, sigma=0, output_dir='simdata_cells/simdata_a1_s0_composite', sim_mode='cells', comp_mode='cells')

	#simdata_a2_composite(12, spots_per_array=2000, sigma=0, output_dir='simdata_cells/simdata_a2_s0_composite', sim_mode='cells', comp_mode='cells')

	# Simualate without BG class
	simdata_a2_composite(12, spots_per_array=2000, sigma=0, output_dir='simdata_clusters/simdata_a2_nobg_s0', 
		sim_mode='clusters')
	simdata_a2_composite(12, spots_per_array=2000, sigma=0, output_dir='simdata_cells/simdata_a2_nobg_s0',
		sim_mode='cells', comp_mode='cells')
