import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ttest_ind

from splotch.utils import savagedickey

# Returns lfc and p-value to evaluate whether gene "g" is DE between two groups of single cells.
def lfc_ttest(cells_in, cells_out, g, equal_var=True):
	'''
	Parameters:
	----------
	cells_in: AnnData 
		expression profiles for cells in which we are searching for DE genes (n_obs=n_cells_in and n_var=n_genes)
	cells_out: AnnData
		expression profiles for cells in comparative group (n_obs=n_cells_out, n_var=n_genes).
	equal_var: bool
		whether to perform standard independent 2-sample t-test (True), or Welch's t-test, which
		does not assume equal population variance

	Returns:
	----------
	lfc: float
		log2-fold change between the mean expression in cells_in and cells_out.
	p: float
		p-value quantifying significance of difference between mean log expression in each group.
	'''
	s1 = np.array(cells_in[:,g].X.todense()).squeeze()
	s2 = np.array(cells_out[:,g].X.todense()).squeeze()

	# Perform log-transform prior to significance testing -- Luecken & Theis (2019).
	# Reduces skewness of data; better approximates normality assumed by Welch's t-test.
	# Two-sample t-test between log-tranformed counts from each group
	_,p = ttest_ind(np.log2(s1+1), np.log2(s2+1), equal_var=equal_var)

	# Welch's t-test with unequal variances yields NaN when applied to two arrays of 0s.
	if np.isnan(p):
		p = 1.0

	# LFC between the means of the two groups
	s1m, s2m = s1.mean(), s2.mean()
	lfc = np.log2((s1m+1)/(s2m+1))

	return lfc, p

# Determine which genes are differentially expressed between each distinct cell type in adata_cells, and either:
# - All other cells (pool_others=True)
# - The next most similar group of cells, as measured by p-val on significance test (pool_others=False)
def de_genes_cells(adata_cells, celltype_label, equal_var=True, pool_others=True):
	'''
	Parameters:
	----------
	adata_cells: AnnData
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
		maps each gene to DE statistics (lfc, p-value) assessing DE in each unique cell type
		(as compared to all others or the next most similar group, depending on pool_others)
	'''

	celltypes = np.unique(adata_cells.obs[celltype_label])
	ngenes = len(adata_cells.var)

	de_genes = pd.DataFrame(
		columns=pd.MultiIndex.from_product([celltypes, ['lfc', 'p']], names=['cell_type', 'statistic']),
		index=adata_cells.var.index)

	for ct in celltypes:

		cells_in = adata_cells[adata_cells.obs[celltype_label]==ct]
		if pool_others:
			cells_out = adata_cells[adata_cells.obs[celltype_label]!=ct]

		for g in adata_cells.var.index:
			# Calculate LFC + p-value between current celltype and all others
			if pool_others:
				de_genes.loc[g, ct] = lfc_ttest(cells_in, cells_out, g, equal_var)

			# Calculate LFC + p-value between current celltype and most similar celltype
			else:
				max_p = -np.inf
				for ct2 in [x for x in celltypes if x != ct]:
					cells_out = adata_cells[adata_cells.obs[celltype_label]==ct2]
					lfc, p = lfc_ttest(cells_in, cells_out, g, equal_var)

					if p > max_p:
						de_genes.loc[g, ct] = (lfc, p)
						max_p = p

	return de_genes

# Determine which genes are DE between two predefined groups of single cells
def de_genes_cells_2(cells_in, cells_out, equal_var=True):
	'''
	Parameters:
	----------
	cells_in: AnnData 
		expression profiles for cells in which we are searching for DE genes (n_obs=n_cells_in and n_var=n_genes)
	cells_out: AnnData
		expression profiles for cells in comparative group (n_obs=n_cells_out, n_var=n_genes).
		-> Must have the same genes as cells_in!
	equal_var: bool
		whether to perform standard independent 2-sample t-test (True), or Welch's t-test, which
		does not assume equal population variance

	Returns:
	----------
	de_genes: (n_genes, 2) DataFrame
		maps each gene to DE statistics (lfc, p-value) assessing DE between the two sets of cells.
	'''
	if not cells_in.var.index.equals(cells_out.var.index):
		raise ValueError('cells_in and cells_out should have the same genes in the same order')

	de_genes = pd.DataFrame(columns=['lfc', 'p'], index=cells_in.var.index)

	for g in cells_in.var.index:
		# Calculate LFC + p-value between two groups of cells
		de_genes.loc[g] = lfc_ttest(cells_in, cells_out, g, equal_var)

	return de_genes
	
# Given a posterior over Beta output by cSplotch, calculate Bayes factors between each cell type
# and all others (separately within each AAR) in order to determine when/if the associated gene is DE.
def de_splotch_celltypes(beta_post):
	'''
	Parameters:
	----------
	beta_post: (n_samples, n_aars, n_celltypes) ndarray
		array containing posterior over Beta output by cSplotch for a single condition

	Returns:
	----------
	bf_ma: (n_aars, n_celltypes) ndarray
		Savage-Dickey estimated Bayes factor between each cell type and all others within each AAR.

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
