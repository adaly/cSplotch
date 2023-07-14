import os
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from argparse import ArgumentParser

from apply_ilastik import CELLCLASS_SUFFIX


# Create a dictionary mapping pixel values to class names
def map_unique_labels(labeled_imfiles, class_names):
	'''
	Parameters:
	----------
	labeled_imfiles: iterable of path
		set of all labeled object files -- each class should be represented at least once
	class_names: iterable of str
		list of class names ordered by increasing pixel value

	Returns:
	-------
	lbl_dict: dict
		mapping of pixel values to class names
	'''
	unique_labels = np.array([])

	for lfile in labeled_imfiles:
		label_img = np.array(Image.open(lfile).convert('L'))
		lbls = np.unique(label_img[label_img > 0])

		unique_labels = np.union1d(unique_labels, lbls)

	unique_labels = sorted(unique_labels)

	if not len(unique_labels) == len(class_names):
		raise ValueError("# of detected FG classes (%d) does not match # of class labels (%d)" % 
			(len(unique_labels), len(class_names)))

	lbl_dict = dict([(x,y) for x,y in zip(unique_labels, class_names)])

	return lbl_dict


# Create dictionary mapping class names to fraction of foreground covered.
def quantify_spot_composition(lfile, label_dict):
	'''
	Parameters:
	----------
	lfile: path
		path to single-channel image with integer labeling of foreground objects 
	label_dict: dict
		mapping of pixel values to class names
	
	Returns:
	-------
	spot_comp: dict
		mapping of class names to fraction of foreground image
	'''
	label_img = np.array(Image.open(lfile).convert('L'))

	fg_area = np.sum([np.sum(label_img==pval) for pval in label_dict.keys()])

	spot_comp = {}
	for pval, class_name in label_dict.items():
		spot_comp[class_name] = np.sum(label_img == pval) / fg_area

	return spot_comp


# Extract spot coordinates from filename of label map
def spot_coords_from_labelfile(lfile):
	return '_'.join(lfile.split(CELLCLASS_SUFFIX)[0].split('_')[-2:])


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-m', '--metadata', type=str, required=True,
		help='cSplotch metadata file')
	parser.add_argument('-l', '--labeled-imgs', type=str, nargs='+', required=True,
		help='Cell type label maps for spots in metadata table.'\
		'Files should be named as [array_name]_[xcoord]_[ycoord].jpg')
	parser.add_argument('-k', '--celltypes-ordered', type=str, nargs='+', required=True,
		help='Cell classes ordered according to listing in Ilastik model (increasing pixel value)')
	parser.add_argument('-o', '--output-dir', type=str, required=True,
		help='Destination in which to save composition files.')
	parser.add_argument('-n', '--array-name', type=str, default=None,
		help='Column in metadata table containing array name. Defaults to Annotation file name.')
	parser.add_argument('-c', '--classic-st', action='store_true', default=False,
		help='Use classic ST format instead of Visium')
	parser.add_argument('-p', '--pxvals-ordered', type=int, nargs='+',
		help='List of pixel values corresponding to class names -- defaults to all nonzero values.'\
		'Use to exclude select classes from foreground consideration')
	args = parser.parse_args()

	meta = pd.read_csv(args.metadata, header=0, sep='\t')

	# Map pixel values to class labels
	if args.pxvals_ordered is not None:
		assert len(args.pxvals_ordered) == len(args.celltypes_ordered)
		lbl_dict = dict([(x,y) for x,y in zip(args.pxvals_ordered, args.celltypes_ordered)])
	else:
		lbl_dict = map_unique_labels(args.labeled_imgs, args.celltypes_ordered)

	for i, afile in enumerate(meta['Annotation file']):
		if args.array_name is None:
			array_name = Path(afile).stem
		else:
			array_name = meta.iloc[i][args.array_name]

		# Find set of label maps associated with current array
		array_labels = [lfile for lfile in args.labeled_imgs if Path(lfile).name.startswith(array_name)]

		# Compile DataFrame of spot compositions for current array
		if len(array_labels) > 0:
			df_spot_comp = []

			for lfile in array_labels:
				spot_comp = quantify_spot_composition(lfile, lbl_dict)
				spot_coords = spot_coords_from_labelfile(lfile)
				
				df_spot_comp.append(pd.DataFrame(spot_comp, index=[spot_coords]))
			df_spot_comp = pd.concat(df_spot_comp).T

			# Save output file
			df_spot_comp.to_csv(os.path.join(args.output_dir, array_name+'.tsv'),
				sep='\t')
