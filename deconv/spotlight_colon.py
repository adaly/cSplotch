import os
import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

import anndata
from anndata import AnnData
from argparse import ArgumentParser
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy
from matplotlib import pyplot as plt


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
	adata_pp = anndata.concat([
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
def S_init(Q, groupby='pheno_major_cell_types'):
	ilastik_classes = ['Colonocyte', 'Immune', 'Interstitial', 'Muscle', 'Rest']
	n_ilastik = len(ilastik_classes)
	n_ctypes = Q.shape[1]

	S = pd.DataFrame(data=np.zeros((n_ilastik, n_ctypes), dtype=int),
					index=ilastik_classes,
					columns=Q.columns)
	for i, ct in enumerate(Q.columns):
		if groupby == 'pheno_major_cell_types':
			if ct == 'Colonocyte':
				S.loc['Colonocyte', ct] = 1
			elif ct == 'Myocyte':
				S.loc['Muscle', ct] = 1
			elif ct in ['Cycling', 'Enteroendocrine', 'Goblet', 'Stem', 'TA', 'Tuft']:
				S.loc['Rest', ct] = 1
			elif ct in ['Fibroblast', 'Glia', 'Lymphatic', 'Macrophage', 'Mesothelial', 'Neuron', 'Vascular']:
				S.loc['Interstitial', ct] = 1
			elif ct in ['B', 'T']:
				S.loc['Immune', ct] = 1
			else:
				raise ValueError('Unknown cell type encountered in AnnData')

		elif groupby == 'pheno_cell_types':
			if ct.startswith('Colonocyte'):
				S.loc['Colonocyte', ct] = 1
			elif ct == 'Myocyte':
				S.loc['Muscle', ct] = 1
			elif np.any([ct.startswith(snt) for snt in ['Cycling', 'Enteroendocrine', 'Goblet', 'Stem', 'TA', 'Tuft']]):
				S.loc['Rest', ct] = 1
			elif np.any([ct.startswith(snt) for snt in ['Fibroblast', 'Glia', 'Lymphatic', 'Macrophage', 'Mesothelial', 'Neuron', 'Vascular']]):
				S.loc['Interstitial', ct] = 1
			elif ct in ['B_cell', 'T_cell']:
				S.loc['Immune', ct] = 1
			else:
				raise ValueError('Unknown cell type encountered in AnnData')

		else:
			raise NotImplementedError

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
	parser.add_argument('-t', '--topic-inference', action='store_true',
		help='Perform topic inference on single-cell data and save results.')
	parser.add_argument('-m', '--counts-mode', type=str, default='counts',
		help='ST data to perform deconvolution on: "counts" or "lambdas"')
	parser.add_argument('-c', '--constrained', action='store_true',
		help='Perform SPOTlight with constraint that cell proportions should match Ilastik output')
	parser.add_argument('-a', '--alpha', type=float, default=1.0,
		help='Weight given to Ilastilk reconstruction loss during training')
	parser.add_argument('-g', '--marker-genes', action='store_true',
		help='Limit SPOTlight to celltype marker genes only.')
	args = parser.parse_args()

	data_dir = '/mnt/home/adaly/ceph/datasets/mouse_colon/'

	#snrna_file = os.path.join(data_dir, 'snrna', 'adata_larger_relabeling_after_tsne_jan18.h5ad')
	#snrna_file = os.path.join(data_dir, 'snrna', 'adata_larger_relabeling_after_tsne_jan18_stem_lgr5.h5ad')
	snrna_file = os.path.join(data_dir, 'snrna', 'adata_larger_relabeling_after_tsne_Oct2022_stemfiltered.h5ad')
	groupby = 'pheno_major_cell_types'  # entry in adata.obs denoting cell types
	#groupby = 'pheno_cell_types'

	meta = pd.read_csv(os.path.join(data_dir, 'Metadata_cSplotch_all.tsv'), header=0, index_col=0, sep='\t')

	# Files in which to save W, Q matrices for ease of future use
	tag = groupby + ('_markers' if args.marker_genes else '')
	W_file = os.path.join('data', '%s_W.csv' % tag)
	Q_file = os.path.join('data', '%s_Q.csv' % tag)

	# Directory in which to save inferred celltype proportions
	if args.marker_genes:
		cellcomp_dir = os.path.join(data_dir, 'celltype_annotations_spotlight_constrained_markers')
	else:
		cellcomp_dir = os.path.join(data_dir, 'celltype_annotations_spotlight_constrained')
	#cellcomp_dir = os.path.join(data_dir, 'celltype_annotations_spotlight_constrained_k30')

	if args.topic_inference:
		# Read in snRNA-seq AnnData
		adata = sc.read_h5ad(snrna_file)
		marker_genes = adata.var['gene_ids-0']
		if adata.raw is not None:
			adata = adata.raw.to_adata()

		# Switch snRNA-seq index to ENSEMBL IDs to match ST
		adata.var['common'] = adata.var.index
		adata.var.index = adata.var['gene_ids-0']

		# Import an example ST count file and find shared genes
		if args.counts_mode == 'counts':
			df_st = pd.read_csv(os.path.join(data_dir, meta['Count file'].iloc[0]), header=0, index_col=0, sep='\t')
		else:
			df_st = pd.read_csv(os.path.join(data_dir, 'lambda_means_combined', '%s_lambdas.csv' % meta.index[0]),
				header=0, index_col=0, sep=',')
		if args.marker_genes:
			ensembl_shared = df_st.index.intersection(marker_genes)
		else:
			ensembl_shared = df_st.index.intersection(adata.var.index)

		adata_pp = preprocess_single_cell(adata, ensembl_shared, groupby)

		print('Calculating W_0', flush=True)
		W_0 = W_init(adata_pp, groupby)

		print('Calculating H_0...', flush=True)
		H_0 = H_init(adata_pp.obs, groupby)

		print('Performing NMF...', flush=True)
		W, H = nmf_infer_topics(adata_pp, W_0, H_0)
		Q = Q_init(adata_pp.obs, groupby, H.values)

		W.to_csv(W_file)
		Q.to_csv(Q_file)
	else:
		print('Loading W, Q from previous NMF...', flush=True)
		W = pd.read_csv(W_file, header=0, index_col=0)
		Q = pd.read_csv(Q_file, header=0, index_col=0)
		ensembl_shared = W.index

	# Construct AnnData object containing counts from all ST arrays, then normalize/scale.
	adata_list = []

	for i in range(len(meta)):
		row = meta.iloc[i]
		name = meta.index[i]

		if args.counts_mode == 'counts':
			cfile = os.path.join(data_dir, row['Count file'])
			if not os.path.exists(cfile):
				continue
			df_st = pd.read_csv(cfile, header=0, index_col=0, sep='\t')
		else:
			cfile = os.path.join(data_dir, 'lambda_means_combined', 
				'%s_lambdas.csv' % name)
			if not os.path.exists(cfile):
				continue
			df_st = pd.read_csv(cfile, header=0, index_col=0, sep=',')

		obs = pd.DataFrame({'array': [name]*len(df_st.columns), 'coords':df_st.columns},
			index=pd.Index(np.array([name+'_'+cstr for cstr in df_st.columns])))
		adata = AnnData(X=df_st.values.T, dtype=np.float32, 
			obs=obs, var=pd.DataFrame(index=df_st.index))
		if len(adata.obs) > 0:
			adata_list.append(adata)

	adata_st = anndata.concat(adata_list)
	adata_st_pp = preprocess_spatial(adata_st, ensembl_shared, args.counts_mode)

	
	# Perform deconvolution separately for each array, using either NNLS or custom minimization
	spot_entropies, spot_iloss, spot_rloss = [], [], []
	S = S_init(Q, groupby)
	for ct in S.columns:
		assert S[ct].sum() == 1, 'snRNA-seq cell type %s must be assigned to exactly one Ilastik class'

	for arr in meta.index:
		adata_array = adata_st_pp[adata_st_pp.obs.array == arr]

		# Load morphological cell proportions predicted by Ilastik
		comp_file = os.path.join(data_dir, meta.loc[arr, 'Composition file'])
		if not os.path.exists(comp_file):
			continue
		df_ilastik = pd.read_csv(comp_file, header=0, index_col=0, sep='\t')
		df_ilastik = filter_comp_mat(df_ilastik)  # Remove spots w/invalid composition & norm.

		# Limit to coordinates shared between count and composition file
		common_coords = np.intersect1d(adata_array.obs.coords, df_ilastik.columns)
		adata_array = adata_array[adata_array.obs.coords.isin(common_coords)]
		df_ilastik = df_ilastik.loc[:, adata_array.obs.coords]
		if len(adata_array.obs) == 0:
			continue

		# Infer topic distributions over spots using NNLS (SPOTlight classic)
		H_prime = nnls_infer_topics(adata_array, W)

		# Infer cell proportions either by NNLS or Ilastik-constrained minimization.
		if args.constrained:
			P, hist_r, hist_i = spotlight_train(
				torch.tensor(H_prime.values), torch.tensor(Q.values),
				torch.tensor(df_ilastik.values), torch.tensor(S.values),
				n_iter=100_000, lr=1e-2, alpha=args.alpha)
			
			P = pd.DataFrame(data=P, index=Q.columns, columns=H_prime.columns)

			print(arr, hist_r[-1], hist_i[-1], flush=True)
		else:
			P = nnls_infer_proportions(H_prime, Q)
		P_norm = clean_proportions(P)  # Remove noisy contributions & normalize

		# Save inferred compositions as cell annotation file
		lbl = 'alpha0' if not args.constrained else 'alpha%.2f' % args.alpha
		cellcomp_subdir = os.path.join(cellcomp_dir, lbl)
		if not os.path.exists(cellcomp_subdir):
			os.mkdir(cellcomp_subdir)
		P_norm.to_csv(os.path.join(cellcomp_subdir, arr+'.tsv'), sep='\t')

		# Calculate entropies of spot composition vectors
		spot_entropies.append(entropy(P_norm, base=P_norm.shape[0], axis=0))

		# Aggregate into morphological superclasses and compare with Ilastik
		L_pred = S.values @ P_norm.values
		spot_iloss.append(((df_ilastik - L_pred)**2).mean(axis=0))

		assert P_norm.columns.equals(df_ilastik.columns)

		# Compare reconstructed counts against input counts
		V_prime_pred = W.values @ (Q.values @ P.values)
		spot_rloss.append(((adata_array.X.T - V_prime_pred)**2).mean(axis=0))


	# Plot histograms over spot entropy and Ilastik loss
	spot_entropies = np.concatenate(spot_entropies)
	spot_iloss = np.concatenate(spot_iloss)
	spot_rloss = np.concatenate(spot_rloss)
	print(spot_entropies.shape, spot_iloss.shape, spot_rloss.shape)

	lbl = 'alpha0' if not args.constrained else 'alpha%.2f' % args.alpha

	np.save('outputs/spotlight_%s_%s_entropy' % (lbl, args.counts_mode), spot_entropies)
	np.save('outputs/spotlight_%s_%s_sclass_mse' % (lbl, args.counts_mode), spot_iloss)
	np.save('outputs/spotlight_%s_%s_reconst_mse' % (lbl, args.counts_mode), spot_rloss)

	fig, ax = plt.subplots(1)
	ax.hist(spot_entropies, bins=50)
	ax.set_xlabel('Composition Entropy')
	ax.set_ylabel('# Spots')
	plt.savefig('outputs/spotlight_%s_%s_entropy.png' % (lbl, args.counts_mode), 
		format='PNG', dpi=300)
	plt.close()

	fig, ax = plt.subplots(1)
	ax.hist(spot_iloss, bins=50)
	ax.set_xlabel('Superclass MSE')
	ax.set_ylabel('# Spots')
	plt.savefig('outputs/spotlight_%s_%s_sclass_mse.png' % (lbl, args.counts_mode),
		format='PNG', dpi=300)
	plt.close()

	fig, ax = plt.subplots(1)
	ax.hist(spot_rloss, bins=50)
	ax.set_xlabel('Norm Counts MSE')
	ax.set_ylabel('# Spots')
	plt.savefig('outputs/spotlight_%s_%s_reconst_mse.png' % (lbl, args.counts_mode),
		format='PNG', dpi=300)
	plt.close()


