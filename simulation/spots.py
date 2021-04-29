import os
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

def round_series_retain_integer_sum(xs):
    N = np.sum(xs)
    Rs = [np.rint(x) for x in xs]
    K = N - np.sum(Rs)
    fs = [x - np.rint(x) for x in xs]
    indices = [i for order, (e, i) in enumerate(reversed(sorted((e,i) for i,e in enumerate(fs)))) if order < K]
    ys = np.array([R + 1 if i in indices else R for i,R in enumerate(Rs)])
    return ys.astype(int)

#########################################################################################
##         Simulation from average expression profiles (one per cell type)             ##
#########################################################################################

# Simulate an ST spot from expression profiles of component cell types
def simspot(celltype_profiles, spot_depth=10000, celltype_wts=None, 
	celltypes_present=None):
	'''
	Parameters:
	----------
	celltype_profiles: (n_genes, n_celltypes) ndarray
		matrix containing characteristic expression of each gene in each cell type
	spot_depth: int
		number of total UMI counts to simulate
	celltype_wts: (n_celltypes,) ndarray of dtype float, function, or None
		if ndarray, proportion of each cell type present in spot.
		if function, returns sample from distribution over celltype proportions (no arguments).
		if None, proportions will be randomly sampled from a uniform distribution.
	celltypes_present: int, array-like of dtype int, or None
		if celltype_wts is None, used to specify either the number of unique cell types 
		present (int), or the specific cell types present (array of int in range 
		[0,n_celltypes-1]). If None, all cell types may be included in random sample
	
	Returns:
	----------
	count_vec: (n_genes,) ndarray of dtype int
		array containing integer counts per gene summing to spot_depth
	celltype_wts: (n_celltypes,) ndarray of dtype float
		simplex of cell type proportions used to sample spot
	'''

	n_genes, n_celltypes = celltype_profiles.shape

	# Normalize expression profiles within each cell type
	expmat = celltype_profiles / celltype_profiles.sum(axis=0)

	if celltype_wts is None:
		if celltypes_present is None:
			W = np.random.random(n_celltypes)
		
		else:
			W = np.zeros(n_celltypes)
			
			# Number of component cell types specified
			if isinstance(celltypes_present, int):
				selected = np.random.randint(n_celltypes, size=celltypes_present)
				W[selected] = np.random.random(celltypes_present)

			# Indices of component cell types specified
			elif isinstance(celltypes_present, np.ndarray):
				W[celltypes_present] = np.random.random(len(celltypes_present))
			
			else:
				raise ValueError('celltypes_present must be integer or array of integers')

	elif callable(celltype_wts):
		W = celltype_wts()

		if not hasattr(W, 'shape') or W.shape != (n_celltypes,):
			raise ValueError('celltype_wts function must return a 1d array of length n_celltypes')

		W = np.maximum(W, np.zeros_like(W))

	elif hasattr(celltype_wts, 'shape') and celltype_wts.shape == (n_celltypes,):
		W = celltype_wts
	else:
		raise ValueError('celltype_wts must be an array of length n_celltypes')

	# Ensure weights sum to 1 (valid simplex)
	W /= np.sum(W)

	# Sample wt*total_count counts from each celltype according to a multinomial distr.
	count_vec = np.zeros(n_genes, dtype=int)
	for i, w in enumerate(W):
		counts = int(np.rint(w * spot_depth))
		if w>0:
			count_vec += np.random.multinomial(counts, expmat[:,i])

	return count_vec, W

# Simulate a tissue composed of spots belonging to multiple AARs in a known proportion
def simarray(celltype_profiles, aar_kwargs, aar_freq, n_spots=2000):
	'''
	Parameters:
	----------
	celltype_profiles: (n_genes, n_celltypes) ndarray
		matrix containing characteristic expression of each gene in each cell type
	aar_kwargs: list of dict
		keyword inputs to simspot corresponding to each AAR
	aar_freq: (n_aars,) ndarray of dtype float
		frequency of each AAR among spots in the tissue
	n_spots: int
		number of spots in simulated array

	Returns:
	----------
	count_mat: (n_genes, n_spots) ndarray of type int
		array containing the number of counts for each gene in each spot
	comp_mat: (n_celltypes, n_spots) ndarray of type float
		array containing the simplex cell type compositions for each spot
	aar_vec: (n_spots,) ndarray of type int
		array containing the AAR index for each spot in the count/comp matrices
	'''

	n_genes, n_celltypes = celltype_profiles.shape
	n_aars = len(aar_freq)

	if len(aar_kwargs) != n_aars:
		raise ValueError('Number of AARs must match between aar_kwargs and aar_freq!')

	count_mat = np.zeros((n_genes, n_spots), dtype=int)
	comp_mat = np.zeros((n_celltypes, n_spots), dtype=float)
	aar_vec = np.zeros(n_spots, dtype=int)

	csum = 0
	for aind, freq in enumerate(aar_freq):
		spots_in = int(np.rint(freq * n_spots))

		for s in range(csum, csum+spots_in):
			count_vec, celltype_wts = simspot(celltype_profiles, **aar_kwargs[aind])

			count_mat[:,s] = count_vec
			comp_mat[:,s] = celltype_wts
			aar_vec[s] = aind

		csum += spots_in

	return count_mat, comp_mat, aar_vec


#########################################################################################
##               Simulation from cell-level data (sn/scRNA-seq matrix)                 ##
#########################################################################################

def simspot_cells(adata_cells, celltype_label, ncells_in_spot=10, max_spot_depth=None,
	celltypes_present=None, celltype_wts=None):
	'''
	Parameters:
	----------
	adata_cells: AnnData object
		expression profiles from sn/scRNA-seq experiment, with n_obs=n_cells and n_var=n_genes
	celltype_label: str
		column of adata_cells.obs in which celltype labels are found
	ncells_in_spot: int
		total number of cells present in spot
	max_spot_depth: int or None
		if int, maximum number of counts captured from cells present
		if None, all counts from included cells are captured
	celltypes_present: int, array-like of str, or None
		if int, number of unique cell types present (chosen at random)
		if array-like, set of unique celltypes to use
		if None, cells are drawn at random from all present
	celltype_wts: (n_celltypes,) ndarray of dtype float, function, or None
		if ndarray, relative proportions of celltypes specified by celltypes_present
		if function, returns simplex over celltypes specified by celltypes_present
		if None, weights drawn uniformly randomly

	Returns:
	----------
	count_vec: (n_genes,) ndarray 
		vector containing counts for each gene in the simulated spot
	ncounts_per_type: pd.Series
		total UMI counts derived from each included cell type 
	ncells_per_type: pd.Series
		total number of cells from each included cell type
	'''

	if celltypes_present is not None and celltype_label is None:
		raise ValueError('A celltype_label must be provided if celltypes_present is specified')

	if celltypes_present is None:
		celltypes_present = np.unique(adata_cells.obs[celltype_label])
	
	if isinstance(celltypes_present, int):
		celltypes_present = np.random.choice(np.unique(adata_cells.obs[celltype_label]), celltypes_present, 
			replace=False)

	if not hasattr(celltypes_present, '__iter__'):
		raise ValueError('celltypes_present must be int, iterable, or None')

	# Determine proportions of cell types present
	if celltype_wts is None:
		W = np.random.random(len(celltypes_present))
	elif callable(celltype_wts):
		W = celltype_wts()

		if not hasattr(W, 'shape') or W.shape != (len(celltypes_present),):
			raise ValueError('celltype_wts function must return a 1d array of length n_celltypes')

		W = np.maximum(W, np.zeros_like(W))
	elif hasattr(celltype_wts, 'shape') and celltype_wts.shape == (len(celltypes_present,)):
		W = celltype_wts
	else:
		raise ValueError('celltype_wts must be an array of length n_celltypes')

	W /= np.sum(W)  # ensure weights sum to 1 (valid simplex)

	# Determine number of cells from each type and sample
	ncells_per_type = round_series_retain_integer_sum(W * ncells_in_spot)
	ncounts_per_type = np.zeros_like(ncells_per_type)
	selected_cells = []

	for i, (ct, n) in enumerate(zip(celltypes_present, ncells_per_type)):
		if n > 0:
			in_type = adata_cells[adata_cells.obs[celltype_label]==ct]

			# If we try to sample more cells of the current type than exist
			if len(in_type) < n:
				logging.warning('Attempting to sample %d cells of type %s (%d present)' % (n, ct, len(in_type)))
				
				n = len(in_type)
				ncells_per_type[i] = n

			s = in_type[np.random.choice(len(in_type), n, replace=False)]
			cmat = np.array(s.X.todense())   # (cells, genes)
			cvec = np.rint(cmat.sum(axis=0)) # rounding after summation over cells minimizes deviation from desired total

			selected_cells.append(cvec)
			ncounts_per_type[i] = cvec.sum()
		else:
			selected_cells.append(np.zeros(len(adata_cells.var)))
			ncounts_per_type[i] = 0

	pooled_counts = np.vstack(selected_cells)

	# Stochastically remove counts until we are within spot_depth
	if max_spot_depth is not None:
		for i in range(max(0, int(pooled_counts.sum()-max_spot_depth))):
			c,g = pooled_counts.nonzero()
			sel = np.random.choice(len(c), p=pooled_counts[c,g]/np.sum(pooled_counts[c,g]))

			pooled_counts[c[sel], g[sel]] -= 1
			ncounts_per_type[c[sel]] -= 1

	assert ncounts_per_type.sum() == pooled_counts.sum(), "something has gone horribly wrong..."

	pooled_counts = pooled_counts.sum(axis=0)
	ncounts_per_type = pd.Series(data=ncounts_per_type, index=celltypes_present)
	ncells_per_type = pd.Series(data=ncells_per_type, index=celltypes_present)

	return pooled_counts, ncounts_per_type, ncells_per_type

def simarray_cells(adata_cells, celltype_label, aar_kwargs, aar_freq=None, n_spots=1000,
	comp_mode='cells'):
	'''
	Parameters:
	----------
	adata_cells: AnnData object
		expression profiles from sn/scRNA-seq experiment, with n_obs=n_cells and n_var=n_genes
	aar_kwargs: list of dict
		keyword inputs to simspot_cell for each AAR
	aar_freq: (n_aars,) ndarray of dtype float
		frequency of each AAR among spots in the tissue
	n_spots: int
		number of spots in simulated array
	comp_mode: 'cells' or 'counts'
		whether to use the normalized number of cells per cell type, or counts from each cell 
		type, to calculate the cell type composition of each spot

	Returns:
	----------
	count_mat: (n_genes, n_spots) ndarray of type int
		array containing the number of counts for each gene in each spot
	comp_df: DataFrame containing (n_celltypes, n_spots) array of type float
		simplex cell type compositions for each spot indexed by celltype
	aar_vec: (n_spots,) ndarray of type int
		AAR index for each spot in the count/comp matrices
	'''

	if aar_freq is None:
		aar_freq = np.ones(len(aar_kwargs)) / len(aar_kwargs)
	
	if hasattr(aar_freq, '__len__'):
		if len(aar_freq) != len(aar_kwargs):
			raise ValueError('Number of AARs implied by lengths of aar_freq and aar_kwargs must match')
	else:
		raise ValueError('aar_freq must be either an iterable or None')

	if not comp_mode in ['cells', 'counts']:
		raise ValueError('comp_mode must be either "cells" or "counts"')

	n_genes = len(adata_cells.var)
	celltypes = np.unique(adata_cells.obs[celltype_label])
	n_celltypes = len(celltypes)

	count_mat = np.zeros((n_genes, n_spots), dtype=int)
	aar_vec = np.zeros(n_spots, dtype=int)

	comp_mat = np.zeros((n_celltypes, n_spots), dtype=float)
	comp_df = pd.DataFrame(data=comp_mat, index=celltypes)

	# For each AAR, simulate a number of spots equal to p(AAR) * n_spots
	csum = 0
	for aind, freq in enumerate(aar_freq):
		spots_in = int(np.rint(freq * n_spots))

		for s in range(csum, csum+spots_in):
			count_vec, counts_per_type, cells_per_type = simspot_cells(
				adata_cells, celltype_label, **aar_kwargs[aind])

			count_mat[:,s] = np.rint(count_vec).astype(int)
			aar_vec[s] = aind

			cells_per_type /= cells_per_type.values.sum()
			counts_per_type /= counts_per_type.values.sum()

			for ct in cells_per_type.index:
				if comp_mode == 'cells':
					comp_df.loc[ct, s] = cells_per_type[ct]
				else:
					comp_df.loc[ct, s] = counts_per_type[ct]

		csum += spots_in

	return count_mat, comp_df, aar_vec


if __name__ == '__main__':
	# Rosenberg average scRNA cluster profiles from mouse spinal cord
	# (44 cell type clusters, 26893 genes)
	# Note: Cat's mouse Visium data has 24 arrays, ~2k spots/array, 13990 genes
	'''scdat = "../data/aam8999_TableS10.csv"
	df = pd.read_csv(scdat, sep=",", index_col=0)
	cmat = (df.values-1)[:,:7]

	def ctw1():
		w = np.random.normal(loc=[.5,.4,.1,0,0,0,0], scale=[.2,.2,.1,0,0,0,0])
		return np.maximum(w, np.zeros_like(w))

	#counts, wts = simspot(cmat, celltypes_present=np.array([0,1]), spot_depth=10000)
	counts, wts = simspot(cmat, celltype_wts=None, spot_depth=10000)

	print(counts.sum())
	print(wts)'''

	'''spot_args = [
		{'celltypes_present': np.array([1,2,4])},
		{'celltypes_present': np.array([1,2,5,6])}
	]
	aar_freqs = [.2, .8]

	count_mat, comp_mat, aar_vec = simarray(cmat, spot_args, aar_freqs, n_spots=6)

	print(count_mat, count_mat.sum(axis=0), count_mat.max(axis=0))
	print(comp_mat, comp_mat.sum(axis=0))
	print(aar_vec)'''

	# Rosenberg snRNA-seq cell data
	cells_file = '../data/GSM3017261_20000_SC_nuclei.h5ad'
	adat = sc.read_h5ad(cells_file)
	sc.pp.normalize_total(adat, target_sum=1000)  # scale all cells to have 1000 total UMIs

	#count_vec, cells_per_type, counts_per_type = simspot_cells(adat, celltype_label='sc_cluster', celltypes_present=4)
	#count_vec, cells_per_type, counts_per_type = simspot_cells(adat, celltype_label='sc_cluster', celltypes_present=['5 Astrocyte - Gfap', '6 Astrocyte - Slc7a10'])

	#print(count_vec.shape, count_vec.sum(), count_vec.min(), count_vec.max())
	#print(cells_per_type)
	#print(counts_per_type)

	aar_kwargs = [
		{'celltypes_present': 4},
		{'celltypes_present': ['5 Astrocyte - Gfap', '6 Astrocyte - Slc7a10']}
	]

	count_mat, comp_df, aar_vec = simarray_cells(adat, celltype_label='sc_cluster', 
		aar_kwargs=aar_kwargs, n_spots=1000, comp_mode='counts')

	print(count_mat.sum(axis=0))

	

