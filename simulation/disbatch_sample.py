import os
import pickle
import argparse

import numpy as np
from scipy import sparse


parser = argparse.ArgumentParser("Generate posteriors from simulated ST data to see if cSplotch can deconvolve cell type expression.")
parser.add_argument('-d', '--data-dir', type=str, required=True, help="Directory containing simulated data and covariates.")
parser.add_argument('-o', '--output-dir', type=str, required=True, help='Directory in which to save output.')
parser.add_argument('-f', '--filename', type=str, required=True, help='Path to save disbatch file.')
parser.add_argument('-n', '--num-samples', type=int, default=1000, help="Number of genes to fit.")
args = parser.parse_args()

covariates = pickle.load(open(os.path.join(args.data_dir, 'covariates.p'), 'rb'))
counts = sparse.load_npz(os.path.join(args.data_dir, 'counts.npz'))

num_genes, num_spots = counts.shape
num_nonzero_spots = (counts>0).sum(axis=1)

fh = open(args.filename, 'w+')

# Draw random sample of genes expressed in >2% of spots
selected_genes = []

while len(selected_genes) < args.num_samples:
	rint = np.random.randint(num_genes)

	if rint not in selected_genes and num_nonzero_spots[rint] > (num_spots*.02):
		selected_genes.append(rint)

		fh.write('python3 fitting.py -d %s -o %s -g %d\n' % (args.data_dir, args.output_dir, rint))

fh.close()