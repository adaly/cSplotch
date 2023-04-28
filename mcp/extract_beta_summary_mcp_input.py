# Construct an AnnData object containing posterior mean/std for betas in each cell type and region
import os
import h5py 
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-i', '--splotch-inputs', type=str, required=True)
parser.add_argument('-o', '--splotch-outputs', type=str, required=True)
parser.add_argument('-d', '--dest-file', type=str, required=True)
parser.add_argument('-l', '--level', type=str, default='beta_level_1')
args = parser.parse_args()

sinfo = pickle.load(open(os.path.join(args.splotch_inputs, 'information.p'), 'rb'))

IS_COMP = False
if 'celltype_mapping' in sinfo.keys():
	IS_COMP = True

n_aars = len(sinfo['annotation_mapping'])
n_conds = len(sinfo['beta_mapping'][args.level])
if IS_COMP:
	n_ctypes = len(sinfo['celltype_mapping'])
n_genes = len(sinfo['genes'])

if IS_COMP:
	X_means = np.zeros((n_conds*n_aars*n_ctypes, n_genes))
	X_stds = np.zeros((n_conds*n_aars*n_ctypes, n_genes))
else:
	X_means = np.zeros((n_conds*n_aars, n_genes))
	X_stds = np.zeros((n_conds*n_aars, n_genes))

# Row-major "flattening" of label array -- store observations in beta, aar, ct order
#  to match flattened array of beta means, stds.
label_array = []
for beta in sinfo['beta_mapping'][args.level]:
	for aar in sinfo['annotation_mapping']:
		if IS_COMP:
			for ct in sinfo['celltype_mapping']:
				label_array.append((beta, aar, ct))
		else:
			label_array.append((beta, aar))

label_array = np.array(label_array)
if IS_COMP:
	obs = pd.DataFrame(label_array, columns=['condition', 'aar', 'celltype'])
else:
	obs = pd.DataFrame(label_array, columns=['condition', 'aar'])

for gid in range(1, n_genes+1):
	if (gid-1) % 1000 == 0:
		print('%d/%d genes...' % (gid-1, n_genes), flush=True)

	postfile = os.path.join(args.splotch_outputs, str(gid//100), 'combined_%d.hdf5' % gid)
	
	if not os.path.exists(postfile):
		print('Warning: %s does not exist' % postfile, flush=True)
		continue
	else:
		post = h5py.File(postfile, 'r')

		X_means[:, gid-1] = post[args.level]['mean'][:].flatten()
		X_stds[:, gid-1] = post[args.level]['std'][:].flatten()

		post.close()

var = pd.DataFrame(index=sinfo['genes'])

adata = ad.AnnData(X=X_means, obs=obs, var=var)
adata.layers['X_stds'] = X_stds

adata.write(args.dest_file)
