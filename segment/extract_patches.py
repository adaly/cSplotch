import os
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 

from pathlib import Path
from scipy.spatial import distance_matrix

from argparse import ArgumentParser


# Filters patches in position file to those matching a set of annotations
def filter_patches_by_annotation(pos, df_annot, annotations):
	'''
	Parameters:
	----------
	pos: DataFrame
		one row per spot; must include columns "pixel_row", "pixel_col", "array_row", "array_col"
		index should match that of df_annot (barcodes)
	df_annot: DataFrame
		one row per spot; first column contains MROI annotatins
		index should match that of pos
	annotations: iterable
		set of annotations to include

	Returns:
	-------
	pos: DataFrame
		filtered DataFrame of spot locations
	'''
	barcodes_keep = df_annot.index[df_annot.iloc[:,0].isin(annotations).values]
	barcodes_keep = pos.index.intersection(barcodes_keep)
	return pos.loc[barcodes_keep]

# Extracts image patches of a specified size + resolution and saves as separate JPEG files.
def extract_patches(img_file, pos, output_dir, patch_size_um=55, output_resolution=1.4,
	array_name=None):
	'''
	Parameters:
	----------
	img_file: path
		path to full-resolution image used in Spaceranger pipeline.
	pos: DataFrame
		one row per spot; must include columns: "pxl_row_in_fullres", "pxl_col_in_fullres", 
		  "array_row", "array_col"
	patch_size_um: float
		size of tissue patch to be sampled at each spot (in um).
	output_resolution: float
		resolution of patches to be output in pixels per um -- pixel height/width of patch 
		will be chosen accordingly.

	'''
	# Automatically determine the resolution:
	# - Visium spots that are 2 units apart in array_col are adjacent (they double scale of x-axis
	#   so that they can offset every odd-numbered row and create a pseudo-hex indexing).
	# - Adjacent spots are 100um apart.
	arr_cols = np.array(pos['array_col'])[:,None]
	pix_cols = np.array(pos['pxl_col_in_fullres'])[:,None]
	arr_dist = distance_matrix(arr_cols, arr_cols)
	pix_dist = distance_matrix(pix_cols, pix_cols)

	x,y = np.where(arr_dist==2)
	assert len(x)>0 and len(y)>0, 'Unable to locate adjacent spots on array'
	d100 = pix_dist[y[0],x[0]]
	print('100um = %d pixels' % d100)

	visium_resolution = d100/100
	patch_size_px = int(np.rint(visium_resolution * patch_size_um))

	# Only consider spots that are covered by tissue.
	pos = pos[pos['in_tissue']==1]

	# Create a sub-directory for each image:
	if array_name is None:
		subdir = Path(img_file).stem
	else:
		subdir = array_name
	if not os.path.isdir(os.path.join(output_dir, subdir)):
		os.mkdir(os.path.join(output_dir, subdir))

	img_arr = np.array(Image.open(img_file))

	for i in range(len(pos)):
		row = pos.iloc[i]

		# Extract image patch
		x_px, y_px = int(row['pxl_col_in_fullres']), int(row['pxl_row_in_fullres'])
		patch = img_arr[(y_px-patch_size_px//2):(y_px+patch_size_px//2), 
			(x_px-patch_size_px//2):(x_px+patch_size_px//2)]

		# Resize patch to achieve desired resolution
		output_size = int(np.rint(patch_size_px / visium_resolution * output_resolution))
		patch = Image.fromarray(patch).resize((output_size, output_size))

		# Save patch as separate JPEG file
		dest_file = os.path.join(output_dir, subdir, 
			'%s_%d_%d.jpg' % (subdir, row['array_col'], row['array_row']))
		patch.save(dest_file, 'JPEG')


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-m', '--metadata', type=str, required=True,
		help='cSplotch metadata file')
	parser.add_argument('-i', '--img-col', type=str, required=True,
		help='Name of column in metadata table containing WSI image locations for each array')
	parser.add_argument('-o', '--output-dir', type=str, required=True,
		help='Directory in which to save output image patches (one subdirectory per array)')
	parser.add_argument('-a', '--annotations', type=str, nargs='+', default=None,
		help='List of MROIs from which to extract spots (defaults to all spots)')
	parser.add_argument('-c', '--classic-st', action='store_true', default=False,
		help='Use classic ST format instead of Visium')
	parser.add_argument('-u', '--patch-size-um', type=float, default=55,
		help='Size of image patches centered at spot locations, in um')
	parser.add_argument('-p', '--pixels-per-um', type=float, default=1.4,
		help='Resolution of output patches, in pixels per um')
	parser.add_argument('-n', '--array-name', type=str, default=None,
		help='Column in metadata table containing array name. Defaults to image filename.')
	args = parser.parse_args()

	if args.classic_st:
		raise NotImplementedError

	meta = pd.read_csv(args.metadata, header=0, sep='\t')

	for i, (img_file, srd) in enumerate(zip(meta[args.img_col], meta['Spaceranger output'])): 
		
		# Read spot position file
		if os.path.exists(os.path.join(srd, "outs/spatial/tissue_positions.csv")):
			positions_path = os.path.join(srd, "outs/spatial/tissue_positions.csv")  # Spaceranger >=2.0
			positions = pd.read_csv(positions_path, index_col=0, header=0)
		elif os.path.exists(os.path.join(srd, "outs/spatial/tissue_positions_list.csv")):
			positions_path = os.path.join(srd, "outs/spatial/tissue_positions_list.csv") # Spaceranger <2.0
			positions = pd.read_csv(positions_path, index_col=0, header=None,
				names=["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"])
		else:
			raise ValueError("Could not locate tissue position listfile in Spaceranger directory %s" % srd)

		# Optionally filter spots by MROI
		if args.annotations is not None:
			annot_file = meta.iloc[i]['Annotation file']
			annot = pd.read_csv(annot_file, header=0, index_col=0)
			positions = filter_patches_by_annotation(positions, annot, args.annotations)

			if len(positions) == 0:
				print('No %s patches found in %s (SKIPPING...)' % (', '.join(args.annotations),
					img_file))
				continue

		array_name = None
		if args.array_name is not None:
			array_name = meta.iloc[i][args.array_name]

		# Read slide image file and extract spot images
		extract_patches(img_file, positions, args.output_dir, patch_size_um=args.patch_size_um, 
			output_resolution=args.pixels_per_um, array_name=array_name)
