import os
import pickle
import argparse

import numpy as np
import scanpy as sc
from scipy import sparse


parser = argparse.ArgumentParser("Generate posteriors from simulated ST data to see if cSplotch can deconvolve cell type expression.")
parser.add_argument(
	'-d', '--data-dir', type=str, required=True, help="Directory containing simulated data and covariates.")
parser.add_argument(
	'-o', '--output-dir', type=str, required=True, help='Directory in which to save output.')
parser.add_argument(
	'-f', '--filename', type=str, required=True, help='Path to save disbatch file.')
parser.add_argument(
	'-n', '--num-samples', type=int, default=1000, help="Number of genes to fit.")

parser.add_argument(
	'-c', '--cells-file', type=str, help='Path to scanpy h5ad file from which ST data were simulated. Used to filter out low-abundance genes in each cell type.')
parser.add_argument(
	'-l', '--cells-label', type=str, help='Label to consider when splitting cells by type.')
parser.add_argument(
	'-t', '--cell-types', type=str, nargs='+', help='Exclude genes not expressed in at least 20%% of cells in one of these types.')
args = parser.parse_args()


covariates = pickle.load(open(os.path.join(args.data_dir, 'covariates.p'), 'rb'))
counts = sparse.load_npz(os.path.join(args.data_dir, 'counts.npz'))

num_genes, num_spots = counts.shape

# Low-abundance genes may cause artificial disagreement between cSplotch and single-cell methods for assessing DE.
# For DE analysis, Rosenberg et al. limited themselves to genes expressed in >20% of cells in at least one of the groups being compared.
if args.cells_file is not None:
	adata = sc.read_h5ad(args.cells_file)

	if adata.raw is not None:
		adata = adata.raw.to_adata()

	if not len(adata.var) == num_genes:
		raise ValueError('Number of genes in simulated ST data and single-cell dataset do not match!')

	if args.cells_label is not None:
		print('Filtering out genes not expressed in >20%% of at least one %s cell type' % args.cells_label)
		ab_inds = np.zeros(len(adata.var), dtype=bool)
	
		# Keep all genes expressed in >20% of at least one (specified) cell type.
		if args.cell_types is None:
			args.cell_types = np.unique(adata.obs[args.cells_label])
		for ct in args.cell_types:
			cells_in = adata[adata.obs[args.cells_label] == ct]
			num_nz_cells = np.array((cells_in.X > 0).sum(axis=0)).squeeze()		
			ab_inds = np.logical_or(ab_inds, ((num_nz_cells / len(cells_in)) > 0.2))
	else:
		print('Filtering out genes not expressed in >20 cells (across all types).')
		raise NotImplementedError('Currently must provide a celltype label to filter on.')
else:
	ab_inds = np.ones(num_genes, dtype=bool)

print('%d genes retained after filtering.' % np.sum(ab_inds))


# Draw random sample of genes expressed in >2% of spots
num_nonzero_spots = (counts>0).sum(axis=1)

selected_genes = []

fh = open(args.filename, 'w+')

while len(selected_genes) < args.num_samples:
	rint = np.random.randint(num_genes)

	if rint not in selected_genes and num_nonzero_spots[rint] > (num_spots*.02) and ab_inds[rint]:
		selected_genes.append(rint)

		fh.write('python3 fitting.py -d %s -o %s -g %d\n' % (args.data_dir, args.output_dir, rint))

fh.close()
