import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ttest_ind

from splotch.utils import savagedickey

# Determine which genes are differentially expressed between two cell types in sc/snRNA-seq data
def de_genes_cells(adata_cells, celltype_label):
	'''
	Parameters:
	----------
	adata_cells: AnnData object
		expression profiles from sn/scRNA-seq experiment, with n_obs=n_cells and n_var=n_genes
	celltype_label: str
		column of adata_cells.obs in which celltype labels are found

	Returns:
	----------
	de_genes: (n_genes, 2) DataFrame
		one-vs-rest log fold change ('lfc') and associated p-values ('p') for each gene between
		unique groups of celltype_label
	'''

	celltypes = np.unique(adata_cells.obs[celltype_label])
	ngenes = len(adata_cells.var)

	de_genes = {}

	for ct in celltypes:
		df = pd.DataFrame(np.empty((ngenes, 2)), index=adata_cells.var.index, columns=['lfc', 'p'])

		cells_in = adata_cells[adata_cells.obs[celltype_label]==ct]
		cells_out = adata_cells[adata_cells.obs[celltype_label]!=ct]

		#for g in adata_cells.var.index:
		for g in adata_cells.var.index[:1000]:
			s1 = np.array(cells_in[:,g].X.todense())
			s2 = np.array(cells_out[:,g].X.todense())

			t,p = ttest_ind(s1, s2, equal_var=False)

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

			df.loc[g, 'lfc'] = lfc
			df.loc[g, 'p'] = p

		#print(ct)
		#print(df.loc[(np.abs(df['lfc']) > 1) & (df['p'] < 0.05)])

		de_genes[ct] = df

	return de_genes

def de_splotch(beta_post):
	'''
	Parameters:
	----------
	beta_post: (n_samples, n_aars, n_celltypes) ndarray
		array containing posterior over Beta for a single condition

	'''
	_, n_aars, n_celltypes = beta_post.shape

	bfmat = np.zeros(n_aars, n_celltypes)

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

	'''
	de_genes = de_genes_cells(adat, 'csplotch_annot')

	for k,v in de_genes.items():
		print(k)
		filtered = v.loc[(np.abs(v['lfc'] > 1)) & (v['p'] < 0.05)]

		print(filtered.sort_values('lfc', ascending=False))
	'''

	beta_post = np.load('beta_level_1_10.npy')
	beta_post = beta_post[:,0,:,:]  # Remove condition dimension

	de_splotch(beta_post)
