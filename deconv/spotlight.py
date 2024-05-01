import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from pathlib import Path
from argparse import ArgumentParser
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy
from matplotlib import pyplot as plt


#################################
# 0. Data Processing Functions	#
#################################

def st_to_anndata(countfiles):
	'''
	Parameters:
	----------
	countfiles: list of str
		paths to Splotch-formatted count files containing raw ST counts.

	Returns:
	-------
	adata: AnnData
	'''
	adata_list = []

	for cfile in countfiles:
		df_st = pd.read_csv(cfile, header=0, index_col=0, sep='\t')
		
		name = Path(cfile).name
		obs = pd.DataFrame({'array': name, 'coords': df_st.columns},
			index=pd.Index(np.array([name+'_'+cstr for cstr in df_st.columns])))
		adata = ad.AnnData(X=df_st.values.T, obs=obs, var=pd.DataFrame(index=df_st.index))
		
		if len(adata.obs) > 0:
			adata_list.append(adata)

	return ad.concat(adata_list)

def json_to_dict(json_file):
	with open(json_file) as fh:
		return json.load(fh)


#########################################################################
# 1. NMF Helper Functions (topic distributions over single-cell data)   #
#########################################################################

# Normalize and scale single-cell data across listed genes
def preprocess_single_cell(adata, ensembl_list, groupby):
	# 0. Select genes of interest
	adata_pp = adata[:, ensembl_list].copy()

	# 0.1. Balance the number of cells per cell type
	celltypes = adata_pp.obs[groupby].unique()
	min_ct = np.min([np.sum(adata_pp.obs[groupby]==ct) for ct in celltypes])
	adata_pp = ad.concat([
		sc.pp.subsample(adata_pp[adata_pp.obs[groupby]==ct], n_obs=min_ct, copy=True) 
		for ct in celltypes])

	# 1. Depth-normalize count data
	sc.pp.normalize_total(adata_pp, 1e4)

	# 2. log(x+1)-transform count data and save as a layer
	adata_pp.layers['X_log1p'] = adata_pp.X.copy()
	sc.pp.log1p(adata_pp, layer='X_log1p')

	# 3. Scale both count data and log count data to unit variance
	sc.pp.scale(adata_pp, zero_center=False)
	sc.pp.scale(adata_pp, zero_center=False, layer='X_log1p')

	return adata_pp

# Create an initialized (genes, topics) W matrix for NMF of snRNA-seq data, with:
# - W[g,t] = 1-P_{g,t}, where P_{g,t} is the p-value of gene g being a marker for cell type t.
# - We treat cell types as equivalent to topics for the purposes of initialization only!
def W_init(adata, groupby):
	
	if not 'rank_genes_groups' in adata.uns.keys():
		print('Performing differential expression analysis...')
		sc.tl.rank_genes_groups(adata, groupby, use_raw=False, layer='X_log1p')
	
	de_df = sc.get.rank_genes_groups_df(adata, None)
	
	n_genes = adata.var.shape[0]
	n_topics = adata.obs[groupby].unique().shape[0]
	
	W_df = pd.DataFrame(data=np.zeros((n_genes, n_topics), dtype=np.float64), 
						index=adata.var.index, 
						columns=adata.obs[groupby].unique())
	
	for i in range(len(de_df)):
		row = de_df.iloc[i]
		W_df.loc[row['names'], row['group']] = 1-row['pvals_adj']
	
	return W_df

# Create initialized binary (topics, cells) H matrix for NMF of snNRA-seq data, with:
# - W[t,c] = 1 if cell c is assigned to cell type t
# - We treat cell types as equivalent to topics for the purposes of initialization only!
def H_init(obs, groupby):
	n_cells = obs.shape[0]
	n_topics = obs[groupby].unique().shape[0]
	
	H_df = pd.DataFrame(data=np.zeros((n_topics, n_cells), dtype=np.float64),
						index = obs[groupby].unique(),
						columns = obs.index)
	for i,bc in enumerate(obs.index):
		row = obs.iloc[i]
		H_df.loc[row[groupby], bc] = 1.0
	
	return H_df

# Calculate the median topic profile Q within each cell type
def Q_init(obs, groupby, H):
	n_ctypes = obs[groupby].unique().shape[0]
	n_topics = H.shape[0]
	
	assert n_ctypes == n_topics, "Number of topics must match number of cell types!"
	
	Q_df = pd.DataFrame(data=np.zeros((n_topics, n_ctypes), dtype=np.float64),
						columns=obs[groupby].unique())
	for ct in Q_df.columns:
		H_ct = H[:, np.array(obs[groupby]==ct, dtype=bool)]        
		Q_df[ct] = np.median(H_ct, axis=1)
	
	return Q_df

# Use NMF to decompose (normalized) counts into W, H
def nmf_infer_topics(adata, W_0, H_0):
	'''
	Parameters:
	----------
	adata: AnnData
		AnnData object containing normalized, variance-scaled (genes, cells) count matrix in X
	W_0: DataFrame
		initialization of (genes, topics) matrix W for NMF
	H_0: DataFrame
		initialization of (topics, cells) matrix H for NMF

	Returns:
	-------
	W: DataFrame
		output (genes, topics) matrix from NMF
	H: DataFrame
		output (topics, cells) matrix from NMF
	'''

	nmf_model = NMF(init='custom', n_components=H_0.shape[0], max_iter=500)

	W = nmf_model.fit_transform(adata.X.T.astype(np.float32), 
							W=W_0.values.astype(np.float32), 
							H=H_0.values.astype(np.float32))
	H = pd.DataFrame(data=nmf_model.components_, index=H_0.index, columns=H_0.columns)
	W = pd.DataFrame(data=W, index=W_0.index, columns=W_0.columns)
	
	return W, H
		

###########################################################################
# 2. NNLS Helper Functions (topic/cell distributions over spatial data)   #
###########################################################################

# Normalize and scale spatial data across listed genes
def preprocess_spatial(adata, ensembl_list, min_counts=100, mode='counts'):
	adata_pp = adata[:, ensembl_list].copy()

	# Only consider spots with >min_counts UMIs; terminate if no spots pass filter.
	sc.pp.filter_cells(adata_pp, min_counts=min_counts)
	if len(adata_pp.obs) == 0:
		return None

	sc.pp.normalize_total(adata_pp, 1e4)
	sc.pp.scale(adata_pp, zero_center=False)

	return adata_pp

# Clean (celltypes, spots) proportion matrix P by dropping minimally present cells & normalizing.
def clean_proportions(P, min_prop=0.01):
	P_norm = P.copy()
	P_norm[P_norm < min_prop] = 0
	return P_norm / P_norm.sum(axis=0)

# Use NNLS to find nonzero topic distributions over spots.
def nnls_infer_topics(adata, W):
	'''
	Parameters:
	----------
	adata: AnnData
		AnnData object containing normalized, variance-scaled (genes, spots) count matrix in X.
	W: DataFrame
		(genes, topics) matrix inferred by NMF on single-cell data

	Returns:
	-------
	H_prime: DataFrame
		output (topics, spots) matrix from NNLS.
	'''

	nnls_model = LinearRegression(positive=True, fit_intercept=False)
	nnls_model.fit(W.values, adata.X.T)

	H_prime = pd.DataFrame(data=nnls_model.coef_.T,
		index=W.columns, columns=adata.obs.coords)
	return H_prime

# Use NNLS to find nonzero cell type distributions over spots.
def nnls_infer_proportions(H_prime, Q):
	'''
	Parameters:
	----------
	H_prime: DataFrame
		(topics, spots) matrix inferred by previous NNLS
	Q: DataFrame
		median (topics, celltypes) matrix from single-cell data

	Returns:
	-------
	P: DataFrame
		output (celltypes, spots) matrix from NNLS
	'''

	nnls_model = LinearRegression(positive=True, fit_intercept=False)
	nnls_model.fit(Q.values, H_prime.values)

	P = pd.DataFrame(data=nnls_model.coef_.T,
		index=Q.columns, columns=H_prime.columns)
	return P


###########################################################
# 3. Constrained NNLS for Ilastik-informed deconvolution  #
###########################################################

# Remove any columns containing NaNs and assure all columns sum to 1.0
def filter_comp_mat(df_ilastik):
	df_clean = df_ilastik.dropna(axis='columns')
	df_clean = df_clean / df_clean.sum(axis=0)
	return df_clean.dropna(axis='columns')

# Return a binary matrix S mapping single-cell types to morphological cell types.
def S_init(Q, mapping_dict):
	superclasses = mapping_dict.keys()
	n_super = len(superclasses)
	n_ctypes = Q.shape[1]

	S = pd.DataFrame(data=np.zeros((n_super, n_ctypes), dtype=int),
					index=superclasses,
					columns=Q.columns)

	for sc, ctypes in mapping_dict.items():
		for ct in ctypes:
			S.loc[sc, ct] = 1

	# Ensure valid mapping
	if np.any(S.sum(axis=1) != 1):
		raise ValueError('Invalid mapping: each cell type must be mapped to exactly one superclass')

	return S

# Given Q and S matrices, returns reconstructions of H_prime and L.
# Stores spot composition matrix P as only parameters.
class SPOTlightConstrained(nn.Module):
	def __init__(self, n_spots, n_types):
		super().__init__()

		P = torch.rand(n_types, n_spots)
		self.P = nn.Parameter(P)
	
	def forward(self, Q, S):
		# Use the sigmoid function to ensure that all values lie between 0, 1
		reconst = torch.matmul(Q.to(torch.float32), torch.sigmoid(self.P))
		ilastik = torch.matmul(S.to(torch.float32), torch.sigmoid(self.P))
		
		return reconst, ilastik

# Perform NNLS with added Ilastik constraints using gradient descent	
def spotlight_train(H_prime, Q, L, S, 
					alpha=1.0, n_iter=1000, lr=1e-3):
	'''
	Parameters:
	----------
	H_prime: torch.tensor
		(topics, spots) matrix inferred by NNLS
	Q: torch.tensor
		(topics, celltypes) matrix from single-cell data.
	L: torch.tensor
		(superclass, spots) composition matrix from Ilastik.
	S: torch.tensor
		(superclass celltypes) binary matrix encoding superclass membership for each celltype.
	alpha: float
		weight given to the Ilastik constraint term during training.
	n_iter: int
		number of iterations for stochastic gradient descent (SGD).
	lr: float
		learning rate for SGD.

	Returns:
	-------
	P: ndarray
		(celltypes, spots) non-negative composition matrix inferred by constrained optimization.
	hist_r: ndarray
		H_prime reconstruction loss at each training iteration.
	hist_l: ndarray
		L reconstruction loss at each training iteration.
	'''

	n_spots = H_prime.shape[1]
	n_types = Q.shape[1]
	
	slc = SPOTlightConstrained(n_spots, n_types)
	optim = Adam(slc.parameters(), lr=lr)
	
	hist_r, hist_i = [],[]
	for i in range(n_iter):
		optim.zero_grad()
		
		reconst, ilastik = slc(Q, S)
		
		l_r = ((reconst - H_prime) ** 2).mean()
		l_i = ((ilastik - L) ** 2).mean(axis=1)
		loss = l_r + alpha * l_i.mean()
		
		hist_r.append(l_r.data.cpu().numpy())
		hist_i.append(l_i.data.cpu().numpy())
		
		loss.backward()
		optim.step()
		
	return torch.sigmoid(slc.P).data.cpu().numpy(), np.array(hist_r), np.array(hist_i)


#####################
# 4. Main function  #
#####################

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-x', '--st-countfiles', type=str, nargs="+", required=True,
		help='Path to Splotch-formatted count files (genes x spots TSV) containing ST data to be deconvolved.')
	parser.add_argument('-d', '--dest-dir', type=str, required=True,
		help='Directory in which to save output.')
	parser.add_argument('-f', '--filter-spots-umi', type=int, default=100,
		help='Filter out spots with less than the specified number of UMIs.')

	parser.add_argument('-t', '--topic-inference', action='store_true',
		help='Perform topic inference on single-cell data & save results prior to deconvolution.\
		Only need to do once unless changing single-cell reference and/or markers')
	parser.add_argument('-s', '--sc-file', type=str,
		help='Path to file containing HDF5-formatted snRNA-seq raw counts.')
	parser.add_argument('-c', '--cell-types', type=str,
		help='Column in AnnData.obs containing cell type annotations.')
	parser.add_argument('-g', '--gene-names', type=str,
		help='Column in AnnData.var containing gene names corresponding to those in ST files.')
	parser.add_argument('-m', '--marker-genes', type=str,
		help='Path to CSV file with 0th column indicating marker genes (matched to index of AnnData).')

	parser.add_argument('-W', '--W_file', type=str, 
		help='Path to W matrix computed by "--topic-inference"')
	parser.add_argument('-Q', '--Q_file', type=str, 
		help='Path to Q matrix computed by "--topic-inference"')

	parser.add_argument('-e', '--superclass-compositions', type=str, nargs="+", default=None,
		help='Path to (superclasses x spots) TSV files containing superclass compositions of ST data, if known.')
	parser.add_argument('-S', '--superclass-mapping', type=str, default=None,
		help='Path to JSON file mapping superclass names to member cell types in "sc-file".')
	parser.add_argument('-a', '--alpha', type=float, default=1.0,
		help='Weight given to superclass reconstruction during deconvolution')
	args = parser.parse_args()


	W_file, Q_file = None, None

	# Calculate topic profiles, save W and Q matrices, and exit.
	if args.topic_inference:
		if args.sc_file is None or args.cell_types is None:
			raise ValueError('Must provide "sc-file" and "cell-types" for topic inference.')

		adata = sc.read_h5ad(args.sc_file)
		if adata.raw is not None:
			adata = adata.raw.to_adata()

		# Limit to marker genes, if provided
		if args.marker_genes is not None:
			df_markers = pd.read_csv(args.marker_genes, header=None, index_col=0)
			markers_in = df_markers.index.intersection(adata.var.index)
			print('%d of %d marker genes found in single-cell data' % (len(markers_in), len(df_markers)))
			adata = adata[:, markers_in]

		# Switch gene indexing to specified name set.
		if args.gene_names is not None:
			adata.var['common'] = adata.var.index.values
			adata.var.index = adata.var[args.gene_names]

		# Read example ST count file and determine set of shared genes
		df_st = pd.read_csv(args.st_countfiles[0], header=0, index_col=0, sep='\t')
		genes_shared = df_st.index.intersection(adata.var.index)
		print('%d genes shared between single-cell and ST' % len(genes_shared))

		# Pre-processing on single-cell count data
		adata_pp = preprocess_single_cell(adata, genes_shared, args.cell_types)

		print('Calculating W_0', flush=True)
		W_0 = W_init(adata_pp, args.cell_types)

		print('Calculating H_0...', flush=True)
		H_0 = H_init(adata_pp.obs, args.cell_types)

		print('Performing NMF...', flush=True)
		W, H = nmf_infer_topics(adata_pp, W_0, H_0)
		Q = Q_init(adata_pp.obs, args.cell_types, H.values)

		W_file = os.path.join(args.dest_dir, args.cell_types+'_W.csv')
		Q_file = os.path.join(args.dest_dir, args.cell_types+'_Q.csv')
		W.to_csv(W_file)
		Q.to_csv(Q_file)

	# Perform deconvolution using pre-computed W and Q matrices.
	if W_file is None and Q_file is None:
		if args.W_file is None or args.Q_file is None:
			raise ValueError('Must provide "W_file" and "Q_file" to perform deconvolution.')

	sdir = args.dest_dir
	if not os.path.exists(sdir):
		os.mkdir(sdir)

	W = pd.read_csv(W_file, header=0, index_col=0)
	Q = pd.read_csv(Q_file, header=0, index_col=0)
	genes_shared = W.index

	# Read in data for superclass-informed deconvolution, if provided
	if args.superclass_mapping is not None:
		S = S_init(Q, json_to_dict(args.superclass_mapping))

		if args.superclass_compositions is not None:
			if len(args.superclass_compositions) != len(args.st_countfiles):
				raise ValueError("Must provide one superclass composition file per count file")
		superclass_constrained = True
	else:
		superclass_constrained = False

	# Construct AnnData object containing counts from all ST arrays, then normalize/scale.
	adata_st = st_to_anndata(args.st_countfiles)
	adata_st_pp = preprocess_spatial(adata_st, genes_shared, min_counts=args.filter_spots_umi)

	for idx, cfile in enumerate(args.st_countfiles):
		arr = Path(cfile).name
		adata_array = adata_st_pp[adata_st_pp.obs.array == arr]

		# Infer cell proportions by NNLS
		if superclass_constrained:
			compfile = args.superclass_compositions[idx]
			df_comp = pd.read_csv(compfile, header=0, index_col=0, sep='\t')
			df_comp = filter_comp_mat(df_comp)  # remove spots with invalid composition & normalize

			# Ensure ordering of superclasses matches that in S
			df_comp = df_comp.loc[S.index]

			# Limit to coordinates shared between count and composition file
			common_coords = np.intersect1d(adata_array.obs.coords, df_comp.columns)
			adata_array = adata_array[adata_array.obs.coords.isin(common_coords)]
			df_comp = df_comp.loc[:, adata_array.obs.coords]
			if len(adata_array.obs) == 0:
				continue

			# Infer topic distributions over spots using NNLS (SPOTlight classic)
			H_prime = nnls_infer_topics(adata_array, W)

			# Refine estimation with superclass constraint
			P, hist_r, hist_i = spotlight_train(
				torch.tensor(H_prime.values), torch.tensor(Q.values),
				torch.tensor(df_comp.values), torch.tensor(S.values),
				n_iter=100_000, lr=1e-2, alpha=args.alpha)
			
			P = pd.DataFrame(data=P, index=Q.columns, columns=H_prime.columns)

		else:
			# Infer topic distributions over spots using NNLS (SPOTlight classic)
			H_prime = nnls_infer_topics(adata_array, W)
			P = nnls_infer_proportions(H_prime, Q)

		P_norm = clean_proportions(P)  # Remove noisy contributions & normalize
		P_norm.to_csv(os.path.join(sdir, arr.split('.')[0]+'.tsv'), sep='\t')


