# For each MCP in a given spatial niche, export up-regulated genes in each cell type for KEGG analysis.
import os
import glob
import pickle
import scipy.stats
import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from argparse import ArgumentParser
from mcp_analysis import single_cell_corr, multi_cell_corr


##### Files for KEGG analysis #####

KEGG_DIR = 'kegg_files/'
GENE_IDS = os.path.join(KEGG_DIR, 'gene_ids.txt')
KEGG_IDS = os.path.join(KEGG_DIR, 'kegg_ids.txt')
KEGG_DATABASE = os.path.join(KEGG_DIR, 'kegg_database.txt')

ENS_TO_COMMON = os.path.join(KEGG_DIR, 'Gene_names_mm.txt')

###################################

# Find top genes by strength of correlation with MCP
def mcp_kegg_genes(all_cell_tpm, all_cell_mcp, n_genes=100, min_abs_corr=0.5):
	df_kegg = pd.DataFrame(columns=['gene', 'MCP', 'celltype'])

	for ct in all_cell_mcp.keys():
		tpm = all_cell_tpm[ct]
		mcp = all_cell_mcp[ct]

		# For a given cell type, find genes correlated with MCP within the same cell.
		genes_corr = single_cell_corr(tpm, mcp, n_genes=n_genes)
		candidate_genes = np.unique(np.concatenate([genes_corr[k] for k in genes_corr.keys()]))

		# Then, find genes within this candidate set correlated with MCP in all cells.
		genes_corr_all = multi_cell_corr(ct, all_cell_tpm, all_cell_mcp, candidate_genes, 
			min_abs_corr=min_abs_corr, n_genes=n_genes)

		for m in genes_corr_all.keys():
			if len(genes_corr_all[m]) > 0:
				df = pd.DataFrame({
					'gene': genes_corr_all[m],
					'MCP': m,
					'celltype': ct
					})
				df_kegg = pd.concat((df_kegg, df))

	return df_kegg

# Find top genes by magnitude of weight contributing to MCP
def mcp_kegg_genes_bywt(all_cell_wts, n_genes=100, min_abs_wt=0, maxk=None):
	df_kegg = pd.DataFrame(columns=['gene', 'MCP', 'celltype'])

	for ct in all_cell_wts.keys():
		wts = all_cell_wts[ct]

		if maxk is None:
			mcp_list = wts.columns
		else:
			mcp_list = ['MCP%d' % k for k in range(1, maxk+1)]

		for m in mcp_list:
			# Top n_genes most positively correlated genes
			wts_decr = wts[m].sort_values(ascending=False)
			wts_decr = wts_decr[wts_decr > 0]
			genes_pos = wts_decr.index.values
			if len(genes_pos) > 0:
				df = pd.DataFrame({
					'gene': genes_pos[:np.minimum(n_genes, len(genes_pos))],
					'wt': wts_decr[:np.minimum(n_genes, len(genes_pos))],
					'MCP': m+'_up',
					'celltype': ct
					})
				df_kegg = pd.concat((df_kegg, df))

			# Top n_genes most negatively correlated genes
			wts_incr = wts[m].sort_values(ascending=True)
			wts_incr = wts_incr[-wts_incr > 0]
			genes_neg = wts_incr.index.values
			if len(genes_neg) > 0:
				df = pd.DataFrame({
					'gene': genes_neg[:np.minimum(n_genes, len(genes_neg))],
					'wt': wts_incr[:np.minimum(n_genes, len(genes_neg))],
					'MCP': m+'_down',
					'celltype': ct
					})
				df_kegg = pd.concat((df_kegg, df))

	return df_kegg


def kegg_analysis(genes_of_interest):
	'''
	Parameters:
	---------
	genes_of_interest: iterable of str
		common names of genes to test for KEGG pathway enrichment

	Returns:
	-------
	df_kegg_path: DataFrame
		enriched pathways, along with pval, p_adj, t (t-statistic of Fisher's exact test),
		and overlap between genes of interest and total pathway size.
	'''

	# Translate ID no's to gene names
	gene_mapping = {}
	with open(GENE_IDS,'r') as f:
		for line in f:
			gene_mapping[line.split('\t')[0]] = line.split('\t')[1].strip()

	# Translate ID no's to pathway names
	kegg_mapping = {}
	with open(KEGG_IDS,'r') as f:
		for line in f:
			kegg_mapping[line.split('\t')[0]] = line.split('\t')[1].strip()

	genes_per_kegg_pathway = {}  # pathway name -> gene names
	kegg_pathways_per_gene = {}  # gene name -> pathway names

	with open(KEGG_DATABASE,'r') as f:
		for line in f:
			kegg_pathway_id = line.split('\t')[0].replace('mmu','')
			kegg_pathway_genes = line.strip().split('\t')[1].split(',')[:-1]

			genes_per_kegg_pathway[kegg_mapping[kegg_pathway_id]] = [gene_mapping[gene] for gene in kegg_pathway_genes]

			for gene in kegg_pathway_genes:
				if not gene_mapping[gene] in kegg_pathways_per_gene:
					kegg_pathways_per_gene[gene_mapping[gene]] = []
				kegg_pathways_per_gene[gene_mapping[gene]].append(kegg_mapping[kegg_pathway_id])

	# Discard genes not found in KEGG database
	genes_of_interest_in_kegg = []
	for gene in genes_of_interest:
		if gene in kegg_pathways_per_gene:
			genes_of_interest_in_kegg.append(gene)

	pvals = []
	test_statistics = []
	tested_kegg_pathways = []
	genes_in_kegg_pathways = {}
	for kegg_pathway in genes_per_kegg_pathway.keys():
		genes_in_kegg_pathways[kegg_pathway] = []
		for gene in genes_of_interest_in_kegg:
			if gene in genes_per_kegg_pathway[kegg_pathway]:
				genes_in_kegg_pathways[kegg_pathway].append(gene) 

		genes_in_kegg_pathway = len(genes_in_kegg_pathways[kegg_pathway])
		contingency_table = [[genes_in_kegg_pathway,
							  len(genes_per_kegg_pathway[kegg_pathway])],
							 [len(genes_of_interest_in_kegg)-genes_in_kegg_pathway,
							  len(kegg_pathways_per_gene)-len(genes_per_kegg_pathway[kegg_pathway])]]

		t,p = scipy.stats.fisher_exact(contingency_table,'greater')
		tested_kegg_pathways.append(kegg_pathway)
		pvals.append(p)
		test_statistics.append(t)

	Q = 0.1
	m = len(genes_per_kegg_pathway)

	BF_critical_values = (scipy.stats.rankdata(pvals))/float(m)*Q

	max_pval = 0
	for idx,pval in enumerate(pvals):
		if pval < BF_critical_values[idx] and max_pval < pval:
			max_pval = pval

	pvals = np.array(pvals)
	test_statistics = np.array(test_statistics)
	tested_kegg_pathways = np.array(tested_kegg_pathways)
	
	sort_inds = np.argsort(pvals)

	df_kegg = pd.DataFrame({
		'pathway': tested_kegg_pathways[sort_inds],
		'pval': pvals[sort_inds],
		'p_adj': np.minimum(pvals[sort_inds] * m / np.arange(1, len(pvals)+1), 1),
		't': test_statistics[sort_inds],
		'genes_in': [len(genes_in_kegg_pathways[t]) for t in tested_kegg_pathways[sort_inds]],
		'genes_tot': [len(genes_per_kegg_pathway[t]) for t in tested_kegg_pathways[sort_inds]]
		})
	df_kegg['overlap'] = df_kegg['genes_in'] / df_kegg['genes_tot']

	return df_kegg[df_kegg['pval'] <= max_pval]

def create_kegg_recarray(kegg_paths_per_mcp):
	compiled_metrics = {}

	for mcp, kegg in kegg_paths_per_mcp.items():
		for metric in ['pval', 'p_adj', 't', 'overlap']:
			df_m = pd.DataFrame({mcp: kegg[metric].values}, index=kegg['pathway'])

			if metric not in compiled_metrics:
				compiled_metrics[metric] = df_m
			else:
				compiled_metrics[metric] = compiled_metrics[metric].join(df_m, how='outer')

	dtype_vals = [(t,'f4') for t in compiled_metrics['pval'].columns.values]
	dtype_names = [(t,'O') for t in compiled_metrics['pval'].columns.values]

	compiled_recarrays = {}

	print(compiled_metrics['overlap'])

	# recarray of pathway names
	rec_arr = []
	for path_name in compiled_metrics['pval'].index:
		row = compiled_metrics['pval'].loc[path_name].values
		rec_arr.append(tuple([path_name if not np.isnan(x) else "nan" for x in row]))
	compiled_recarrays['name'] = np.array(rec_arr, dtype=dtype_names)

	# recarrays of metrics for pathway association (pval, p_adj, t, overlap)
	for metric, df_m in compiled_metrics.items():
		rec_arr = []
		for i in range(len(df_m)):
			rec_arr.append(tuple(df_m.iloc[i].values))
		compiled_recarrays[metric] = np.array(rec_arr, dtype=dtype_vals)

	return compiled_recarrays


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-m', '--mcp-dirs', type=str, nargs='+', required=True,
		help='Directory/directories containing "mcp_[celltype].csv" and \
		"tpm_[celltype].csv" files output by multi-CCA for given region(s).')
	parser.add_argument('-a', '--adata-file', type=str, default=None,
		help='Path to AnnData file in which to save recarray representation of KEGG analysis.')
	parser.add_argument('-k', '--maxk', type=int, default=None,
		help='Maximum number of MCPs to keep in each MROI')
	args = parser.parse_args()

	maxk = args.maxk

	df_map = pd.read_csv(ENS_TO_COMMON, header=None, index_col=0, sep='_', names=['common'])

	kegg_paths_per_mcp = {}

	# Read in latent variables (MCPs) for each cell type, as well as TPM matrices (derived 
	#  from betas, after dropping genes with zero variance across measured time points).
	for rdir in args.mcp_dirs:
		mcp, tpm, wts = {}, {}, {}

		region = Path(rdir).stem
		print(region)

		for mcp_file in glob.glob(os.path.join(rdir, 'mcp_*.csv')):
			tpm_file = Path(mcp_file).parent / Path(mcp_file).name.replace('mcp', 'tpm')
			wts_file = Path(mcp_file).parent / Path(mcp_file).name.replace('mcp', 'ws')

			if not os.path.exists(tpm_file):
				raise ValueError('Could not find corresponding TPM file for %s' % mcp_file)

			ct = Path(mcp_file).stem.split('_')[1]
			mcp[ct] = pd.read_csv(mcp_file, header=0, index_col=0)  # (conditions x MCPs)
			tpm[ct] = pd.read_csv(tpm_file, header=0, index_col=0)  # (genes x conditions)
			wts[ct] = pd.read_csv(wts_file, header=0, index_col=0)  # (genes x MCPs)

		#df_kegg = mcp_kegg_genes(tpm, mcp, n_genes=250)
		df_kegg = mcp_kegg_genes_bywt(wts, n_genes=250, maxk=maxk)
		df_kegg = df_kegg.join(df_map, how='left', on='gene')

		print(df_kegg)

		for mcp in df_kegg['MCP'].unique():
			genes_in = np.unique(df_kegg['common'][df_kegg['MCP']==mcp].values)
			df_kegg_path = kegg_analysis(genes_in)

			if len(df_kegg_path) > 0:
				kegg_paths_per_mcp['%s %s' % (region, mcp)] = df_kegg_path

	print(kegg_paths_per_mcp.keys())
	#fh = open('kegg_paths_per_mcp.dat', 'wb')
	#pickle.dump(kegg_paths_per_mcp, fh)


	if args.adata_file is not None:
		adata = sc.read_h5ad(args.adata_file)

		#kegg_paths_per_mcp = pickle.load(open('kegg_paths_per_mcp.dat', 'rb'))

		recarrays = create_kegg_recarray(kegg_paths_per_mcp)

		params = {'groupby': 'annotation',
				  'method': 'mcp_kegg',
				  'reference': 'rest',
				  'use_raw': False}
		adata.uns['mcp_kegg'] = {
			'logfoldchanges': recarrays['overlap'], 
			'pvals': recarrays['pval'],
			'pvals_adj': recarrays['p_adj'],
			'scores': recarrays['t'],
			'names': recarrays['name'], 
			'params': params
		}
		
		adata.write(args.adata_file)
