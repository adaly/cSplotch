import os
import numpy as np
import pandas as pd


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
	celltype_wts: (n_celltypes,) ndarray of dtype float, or None
		if provided, proportion of each cell type present in spot.
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
			celltype_wts = np.random.random(n_celltypes)
		
		else:
			celltype_wts = np.zeros(n_celltypes)
			
			# Number of component cell types specified
			if isinstance(celltypes_present, int):
				selected = np.random.randint(n_celltypes, size=celltypes_present)
				celltype_wts[selected] = np.random.random(celltypes_present)

			# Indices of component cell types specified
			elif isinstance(celltypes_present, np.ndarray):
				celltype_wts[celltypes_present] = np.random.random(len(celltypes_present))
			
			else:
				raise ValueError('celltypes_present must be integer or array of integers')

	celltype_wts /= np.sum(celltype_wts)

	# Sample wt*total_count counts from each celltype according to a multinomial distr.
	count_vec = np.zeros(n_genes, dtype=int)
	for i, w in enumerate(celltype_wts):
		counts = int(np.rint(w * spot_depth))
		if w>0:
			count_vec += np.random.multinomial(counts, expmat[:,i])

	return count_vec, celltype_wts

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


if __name__ == '__main__':
	# Rosenberg average scRNA cluster profiles from mouse spinal cord
	# (44 cell type clusters, 26893 genes)
	# Note: Cat's mouse Visium data has 24 arrays, ~2k spots/array, 13990 genes
	scdat = "../data/aam8999_TableS10.csv"
	df = pd.read_csv(scdat, sep=",", index_col=0)
	cmat = df.values-1

	counts, wts = simspot(cmat, celltypes_present=6, spot_depth=10000)

	spot_args = [
		{'celltypes_present': np.array([1,2,4])},
		{'celltypes_present': np.array([1,2,5,6])}
	]
	aar_freqs = [.2, .8]

	count_mat, comp_mat, aar_vec = simarray(cmat, spot_args, aar_freqs, n_spots=6)

	print(count_mat, count_mat.sum(axis=0), count_mat.max(axis=0))
	print(comp_mat, comp_mat.sum(axis=0))
	print(aar_vec)
