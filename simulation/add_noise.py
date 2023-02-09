import os
import pickle
import argparse
import numpy as np


parser = argparse.ArgumentParser("Add noise to the cell compositions in a covariate file to be input to cSplotch.")
parser.add_argument(
	'-d', '--data-dir', type=str, required=True, help="Directory containing noise-free covariates.")
parser.add_argument(
	'-o', '--output-dir', type=str, required=True, help='Directory in which to save noisy covariates.')
parser.add_argument(
	'-s', '--scale', type=float, required=True, help='Magnitude (sdev) of Gaussain noise to be added, or a negative value for random labels.')
args = parser.parse_args()


cov = pickle.load(open(os.path.join(args.data_dir, 'covariates.p'), 'rb'))

if args.scale > 0:
	delta = np.random.normal(scale=args.scale, size=cov['E'].shape)
	E2 = cov['E'] + delta
	E2 = np.maximum(E2, 0)  # remove negative proportions
else:
	E2 = np.random.uniform(size=cov['E'].shape)
E2 = E2 / E2.sum(axis=1)[:, None]  # valid simplexes

cov['E'] = E2
pickle.dump(cov, open(os.path.join(args.output_dir, 'covariates.p'), 'wb'))