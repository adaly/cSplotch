import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# https://www.statsmodels.org/devel/install.html
from statsmodels.stats.multitest import fdrcorrection


# Given a (genes, MCPs) correlation matrix, return n_genes with strongest positive/negative
#  correlation to each MCP.
def get_top_corr(df_corr, n_genes, min_abs_corr=0.05):
	'''
	Parameters:
	----------
	df_corr: (genes, MCPs) DataFrame
		correlation between each gene and each MCP.
	n_genes: int
		number of positively and negatively correlated genes to return.
	min_abs_corr: float
		minimum absolute value for correlation to be considered.

	Returns:
	-------
	genes_corr: dict
		mapping of MCPs to arrays of genes with strongest positive/negative correlation.
	'''

	genes_corr = {}

	#print(df_corr)

	for mcp in df_corr.columns:
		# Negatively correlated genes
		df_sorted = df_corr.sort_values(mcp, ascending=True)
		ci = min(df_sorted[mcp][n_genes], -min_abs_corr)
		genes_neg = df_corr.index[df_corr[mcp] < ci].values

		#print('neg <', ci)
		#print(df_corr.loc[genes_neg, mcp].sort_values())

		# Positively correlated genes
		df_sorted = df_corr.sort_values(mcp, ascending=False)
		ci = max(df_sorted[mcp][n_genes], min_abs_corr)
		genes_pos = df_corr.index[df_corr[mcp] > ci].values

		#print('pos >', ci)
		#print(df_corr.loc[genes_pos, mcp].sort_values())

		genes_corr[mcp+'_up'] = genes_pos
		genes_corr[mcp+'_down'] = genes_neg

	return genes_corr

# For a given cell type, identifies genes whose expression is most strongly correlated (positive or negative) 
#   with each MCP within the same cell type.
def single_cell_corr(tpm_cell, mcp_cell, n_genes=100):
	'''
	Parameters:
	----------
	tpm_cell: (genes, samples) DataFrame
		for given cell type, matric containing the expression level of each gene at each time point (in TPM).
	mcp_cell: (samples, MCPs) DataFrame
		for given cell type, matrix containing the expression level of each MCP at each time point.
	n_genes: int
		number of positively and negatively correlated genes to return.

	Returns:
	-------
	genes_corr: dict
		mapping of MCPs to arrays of genes with strongest positive/negative correlation in the same cell type.
	'''
	G, M = tpm_cell.shape[0], mcp_cell.shape[1]
	corr_mat = np.zeros((G, M))

	for g in range(G):
		for m in range(M):
			r, p = pearsonr(tpm_cell.iloc[g,:], mcp_cell.iloc[:,m])
			corr_mat[g,m] = r

	df_corr = pd.DataFrame(corr_mat, index=tpm_cell.index, columns=mcp_cell.columns)
	genes_corr = get_top_corr(df_corr, n_genes)

	return genes_corr

# For a given cell type, identifies genes whose expression is most strongly correlated (positive or negative) 
#   with each MCP across all cell types.
def multi_cell_corr(celltype, all_cell_tpm, all_cell_mcp, candidate_genes, n_genes=100, min_abs_corr=0.05):
	'''
	Parameters:
	----------
	celltype: str
		name of "source" cell type (cell type of interest).
	all_cell_tpm: dict of (str, DataFrame)
		(genes, samples) DataFrames containing gene expression data for each cell type in the niche.
	all_cell_mcp: dict of (str, DataFrame)
		(samples, MCPs) DataFrames containing MCP expression data for each cell type in the niche.
	candidate_genes: array of str
		set of genes to consider for significant correlation.
	n_genes: int
		maximum number of top correlated genes to consider for each cell type.
	min_abs_corr: float
		minimum absolute value of rho to be considered as correlated.

	Returns:
	-------
	genes_corr: dict
		mapping of MCPs to arrays of genes with strongest positive/negative correlation across all cell types.
	'''
	tpm = all_cell_tpm[celltype].loc[candidate_genes]

	# Genes that show significant correlation to each MCP across all cell types in niche.
	genes_corr_all = None

	for comp_ct in all_cell_tpm.keys():
		#print(comp_ct)
		
		mcp = all_cell_mcp[comp_ct]
		G, M = tpm.shape[0], mcp.shape[1]

		# Calculate Spearman correlation (rho and p-val) between each gene in source cell type
		#  and each candidate cell type (including source). 
		R, P = np.zeros((G, M)), np.zeros((G, M))
		for g in range(G):
			for m in range(M):
				r, p = spearmanr(tpm.iloc[g,:], mcp.iloc[:,m])
				R[g,m] = r
				P[g,m] = p
		
		# Perform BH (FDR) correction on Spearman p-values.
		for m in range(M):
			reject, padj = fdrcorrection(P[:,m])
			P[:,m] = padj

		# Mark all genes without significant correlation as uncorrelated.
		R[P > 0.05] = 0

		# Find top genes from source cell type correlated with each MCP in dest cell type.
		genes_corr = get_top_corr(
			pd.DataFrame(data=R, index=tpm.index, columns=mcp.columns), 
			n_genes, min_abs_corr)
		# ...then, update running list of genes in source cell type that show strong positive/negative 
		#  correlation with each MCP across *all* cell types.
		if genes_corr_all is None:
			genes_corr_all = genes_corr
		else:
			for k in genes_corr_all.keys():
				genes_corr_all[k] = np.intersect1d(genes_corr_all[k], genes_corr[k])

	return genes_corr_all

if __name__ == '__main__':
	mcca_dir = 'mcca_pilot'
	region = 'crypt_apex'

	# Read in latent variables (MCPs) for each cell type, as well as TPM matrices (derived 
	#  from betas, after dropping genes with zero variance across measured time points).
	mcp, tpm = {}, {}
	for ct in ['colonocyte', 'interstitial', 'epithelial']:
		mcp_file = os.path.join(mcca_dir, region, 'mcp_%s.csv' % ct)
		tpm_file = os.path.join(mcca_dir, region, 'tpm_%s.csv' % ct)

		mcp[ct] = pd.read_csv(mcp_file, header=0, index_col=0)
		tpm[ct] = pd.read_csv(tpm_file, header=0, index_col=0)

	# Find initial set of genes -- strongest correlation with MCP level in the same cell type.
	genes_corr = single_cell_corr(tpm['colonocyte'], mcp['colonocyte'])
	candidate_genes = np.unique(np.concatenate([genes_corr[k] for k in genes_corr.keys()]))

	genes_corr_all = multi_cell_corr('colonocyte', tpm, mcp, candidate_genes, min_abs_corr=0.5)

	print(genes_corr_all)
	