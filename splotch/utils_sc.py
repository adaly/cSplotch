import os
import logging
import numpy as np
import pandas as pd
import scanpy as sc

from scanpy._utils import check_nonnegative_integers


# Compute mean and standard deviation of expression within a cell group.
def grouped_obs_mean_std(adata, group_key, layer=None, gene_symbols=None):
	'''
	Parameters:
	----------
	adata: AnnData
		single cell data to be analyzed.
	group_key: str
		value in adata.obs defining cell types of interest.
	layer: str
		layer of adata containing expression data to use.
	gene_symbols: str
		value in adata.var containing gene names to use in returned DataFrame.
	
	Returns:
	----------
	means: DataFrame
		mean value of the expression of each gene (row) in each cell type (column).
	stds: DataFrame
		standard deviation of the expression of each gene (row) in each cell type (column).
	'''

	if layer is not None:
		getX = lambda x: x.layers[layer]
	else:
		getX = lambda x: x.X
	if gene_symbols is not None:
		new_idx = adata.var[gene_symbols].values
	else:
		new_idx = adata.var_names

	grouped = adata.obs.groupby(group_key)
	means = pd.DataFrame(
		np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
		columns=grouped.groups.keys(),
		index=new_idx
	)
	stds = pd.DataFrame(
		np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
		columns=grouped.groups.keys(),
		index=new_idx
	)

	for group, idx in grouped.indices.items():
		X = getX(adata[idx])
		means[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
		stds[group] = np.ravel(X.todense().std(axis=0, dtype=np.float64))
	return means, stds


def priors_from_celldata(adata, sc_group_key, st_cell_types, st_gene_list, 
	sc_gene_symbols=None, sc_layer=None, use_raw=True, target_depth=1e4,
	filter_cells_kws=None, filter_genes_kws=None):
	'''
	Parameters:
	----------
	adata: AnnData
		single-cell (or single-nuclear) data from which to infer priors over log expression.
	sc_group_key: str
		entry in adata.var containing cell type annotations of interest.
	st_cell_types: iterable of str
		cell types for which empirical priors over expression should be calculated.
	st_gene_list: iterable of str
		genes for which empirical priors over expression should be calculated.
	sc_gene_symbols: str or None
		value in adata.var containing gene names to use in returned DataFrame, or None for index.
	sc_layer: str or None
		value in adata.layers containing expression data, or None for adata.X.
	use_raw: bool
		if True, preprocess counts from adata.raw. If False, use adata.X as-is (expected to be log-normalized).
	target_depth: float
		target depth for all cells -- should be set to median spot depth across all ST spots.
	filter_cells_kws: dict or None
		keyword arguments to scanpy.pp.filter_cells, or None to consider all cells.
	filter_genes_kws: dict or None
		keyword arguments to scanpy.pp.filter_genes, or None to consider all genes.

	Returns:
	----------
	beta_prior_means: DataFrame
		prior mean over log expression to use for each gene (row) and cell type (column).
	beta_prior_stds: DataFrame
		prior standard deviation over log expression to use for each gene (row) and cell type (column).
	'''

	if use_raw and adata.raw is None:
		raise ValueError("Received `use_raw=True`, but `adata.raw` is empty.")

	if not use_raw and check_nonnegative_integers(adata.X):
		raise ValueError("Count data appears to not be log transformed!")

	if use_raw:
		adata = adata.raw.to_adata()

		if filter_genes_kws is not None:
			sc.pp.filter_genes(adata, **filter_genes_kws)

		if filter_cells_kws is not None:
			sc.pp.filter_cells(adata, **filter_cells_kws)

		sc.pp.normalize_total(adata, target_depth)
		sc.pp.log1p(adata)

	sc_group_means, sc_group_stds = grouped_obs_mean_std(adata, sc_group_key, 
		layer=sc_layer, gene_symbols=sc_gene_symbols)

	# Count number of genes shared between ST and single-cell
	n_shared_genes, n_shared_celltypes = 0, 0
	for gene in st_gene_list:
		if gene in sc_group_means.index:
			n_shared_genes += 1
	logging.info('%d of %d genes shared between ST and single-cell data.' % (n_shared_genes, len(st_gene_list)))

	# Count number of cell types shared between ST and single-cell, and throw a warning if some are missing!
	for ct in st_cell_types:
		if ct in sc_group_means.columns:
			n_shared_celltypes += 1
	if n_shared_celltypes < len(st_cell_types):
		logging.warning('%d ST cell types not found in single-cell data!' % len(st_cell_types-n_shared_celltypes))

	# For any unobserved (gene, celltype) pair, default to prior of N(0,2)
	beta_prior_means = np.zeros((len(st_gene_list), len(st_cell_types)))
	beta_prior_stds = 2 * np.ones((len(st_gene_list), len(st_cell_types)))

	# For all observed (gene, celltype) pairs, use empirical mean/std as prior hyperparameters.
	for i, gene in enumerate(st_gene_list):
		for j, ct in enumerate(st_cell_types):
			if gene in sc_group_means.index and ct in sc_group_means.columns:
				beta_prior_means[i,j] = sc_group_means.loc[gene, ct]
				beta_prior_stds[i,j] = sc_group_stds.loc[gene, ct]

	beta_prior_means = pd.DataFrame(beta_prior_means,
		columns=st_cell_types, index=st_gene_list)
	beta_prior_stds = pd.DataFrame(beta_prior_stds,
		columns=st_cell_types, index=st_gene_list)

	return beta_prior_means, beta_prior_stds


if __name__ == '__main__':
	snrna_dir = '/Users/adaly/Dropbox (Simons Foundation)/cell_segmentation_colon/snrna_anndata/'
	adata = sc.read_h5ad(os.path.join(snrna_dir, 'adata_larger_relabeling_after_tsne_jan18.h5ad'))

	ilastik_classes = ['Colonocyte', 'Immune', 'Interstitial', 'Muscle', 'Rest']

	st_dir = '/Users/adaly/Documents/mouse_colon/counts/'
	df_count = pd.read_csv(os.path.join(st_dir, 'L9CN82_E2_stdata_under_tissue_IDs.txt.unified.tsv'),
		header=0, index_col=0, sep='\t')
	st_genes = df_count.index 

	sel_classes = ['Colonocyte', 'T', 'Myocyte']  # For now -- eventually need to create an obs label with Ilastik class
	df_mu, df_sigma = priors_from_celldata(adata, 'pheno_major_cell_types', sel_classes, st_genes,
		sc_gene_symbols='gene_ids-0')

	print(df_mu)
	print(df_sigma)
