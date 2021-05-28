import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ttest_ind

from splotch.utils import savagedickey

# Determine which genes are differentially expressed between two cell types in sc/snRNA-seq data
def de_genes_cells(adata_cells, celltype_label, equal_var=True, pool_others=True):
	'''
	Parameters:
	----------
	adata_cells: AnnData object
		expression profiles from sn/scRNA-seq experiment, with n_obs=n_cells and n_var=n_genes
	celltype_label: str
		column of adata_cells.obs in which celltype labels are found
	equal_var: bool
		whether to perform standard independent 2-sample t-test (True), or Welch's t-test, which
		does not assume equal population variance
	pool_others: bool
		whether to return LFC between each celltype and pool of others (True),
		or between each celltype and the least significantly different among others (False)

	Returns:
	----------
	de_genes: (n_genes, (n_celltypes, 2)) DataFrame
		one-vs-rest log fold change ('lfc') and associated p-values ('p') for each gene between
		unique groups of celltype_label
	'''

	celltypes = np.unique(adata_cells.obs[celltype_label])
	ngenes = len(adata_cells.var)

	de_genes = pd.DataFrame(
		columns=pd.MultiIndex.from_product([celltypes, ['lfc', 'p']], names=['cell_type', 'statistic']),
		index=adata_cells.var.index)

	def _lfc_ttest(cells_in, cells_out, g):
		s1 = np.array(cells_in[:,g].X.todense()).squeeze()
		s2 = np.array(cells_out[:,g].X.todense()).squeeze()

		# Two-sample t-test between raw counts from each group
		_,p = ttest_ind(s1, s2, equal_var=equal_var)

		# LFC between the means of the two groups
		s1m, s2m = s1.mean(), s2.mean()
		if s2m == 0.:
			if s1m == 0.:
				lfc = 0
			else:
				lfc = np.inf
		elif s1m == 0.:
			lfc = -np.inf
		else:
			lfc = np.log2(s1m/s2m)

		return lfc, p

	for ct in celltypes:

		cells_in = adata_cells[adata_cells.obs[celltype_label]==ct]
		if pool_others:
			cells_out = adata_cells[adata_cells.obs[celltype_label]!=ct]

		for g in adata_cells.var.index:
			# Calculate LFC + p-value between current celltype and all others
			if pool_others:
				de_genes.loc[g, ct] = _lfc_ttest(cells_in, cells_out, g)

			# Calculate LFC + p-value between current celltype and most similar celltype
			else:
				max_p = -np.inf
				for ct2 in [x for x in celltypes if x != ct]:
					cells_out = adata_cells[adata_cells.obs[celltype_label]==ct2]
					lfc, p = _lfc_ttest(cells_in, cells_out, g)

					if p > max_p:
						de_genes.loc[g, ct] = (lfc, p)
						max_p = p

	return de_genes

def de_splotch(beta_post):
	'''
	Parameters:
	----------
	beta_post: (n_samples, n_aars, n_celltypes) ndarray
		array containing posterior over Beta for a single condition

	'''
	_, n_aars, n_celltypes = beta_post.shape

	bfmat = np.zeros((n_aars, n_celltypes))

	for a in range(n_aars):
		for ct in range(n_celltypes):
			inds1 = ct
			inds2 = [i for i in range(n_celltypes) if i != ct]

			bf = savagedickey(beta_post[:,a,inds1].flatten(), beta_post[:,a,inds2].flatten())
			bfmat[a,ct] = bf
	return bfmat


if __name__ == "__main__":
	# Rosenberg snRNA-seq cell data
	cells_file = '../data/GSM3017261_20000_SC_nuclei.h5ad'
	adat = sc.read_h5ad(cells_file)
	sc.pp.normalize_total(adat, target_sum=1000)  # scale all cells to have 1000 total UMIs

	selected_profiles = [
	    '6 Astrocyte - Slc7a10',
	    '10 Microglia',
	    '12 Oligodendrocyte Myelinating',
	    '16 Alpha motor neurons'
	]
	adat4 = adat[adat.obs['sc_cluster'].isin(selected_profiles)]

	de_genes = de_genes_cells(adat4, 'sc_cluster', pool_others=False, equal_var=False)

	print(de_genes)

	#beta_post = np.load('beta_level_1_10.npy')
	#beta_post = beta_post[:,0,:,:]  # Remove condition dimension

	#de_splotch(beta_post)
