import os, sys
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.ndimage import label

from splotch.utils import watershed_tissue_sections, get_spot_adjacency_matrix

# Read in a series of Loupe annotation files and return the set of all unique categories.
# NOTE: "Undefined" 
def unique_annots_loupe(loupe_files):
	all_annots = []

	for fh in loupe_files:
		df = pd.read_csv(fh, header=0, sep=",")
		df = df.dropna()
		for a in df.iloc[:,1].values:
			a = str(a)
			if len(a)>0 and a.lower() != "undefined":
				all_annots.append(a)

	return sorted(list(set(all_annots)))

def visium_find_position_file(spaceranger_dir, hd_binning=None):
	# Visium HD
	if hd_binning is not None:
		position_file = os.path.join(spaceranger_dir, 'outs', 'binned_outputs', hd_binning, 'spatial', 
			'tissue_positions.parquet')
	else:
		# Spaceranger >= 2.0
		if os.path.exists(os.path.join(spaceranger_dir, 'outs', 'spatial', 'tissue_positions.csv')):
			position_file = os.path.join(spaceranger_dir, 'outs', 'spatial', 'tissue_positions.csv')
		# Spaceranger <= 1.9
		else:
			position_file = os.path.join(spaceranger_dir, 'outs', 'spatial', 'tissue_positions_list.csv')
	return position_file

# Annotataion matrix from Loupe annotation file
def read_annot_matrix_loupe(loupe_file, position_file, unique_annots):
	annots = pd.read_csv(loupe_file, header=0, sep=",")

	if position_file.endswith('.parquet'):
		positions = pd.read_parquet(position_file)
		positions = positions.set_index('barcode')
	elif position_file.endswith('_list.csv'):
		# Spaceranger <=1.9 has no header row
		positions = pd.read_csv(position_file, index_col=0, header=None,
		  names=["in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"])
	else:
		positions = pd.read_csv(position_file, index_col=0, header=0)

	annot_matrix = np.zeros((len(unique_annots), len(annots['Barcode'])), dtype=int)

	positions_list = []
	for i,b in enumerate(annots['Barcode']):
		xcoor = positions.loc[b,'array_col']
		ycoor = positions.loc[b,'array_row']
		positions_list.append('%d_%d' % (xcoor, ycoor))
		a = str(annots.iloc[i,1])

		if a in unique_annots:
			annot_matrix[unique_annots.index(a),i] = 1

	annot_frame = pd.DataFrame(annot_matrix, index=unique_annots, columns=positions_list)

	return annot_frame

# Converts from pseudo-hex indexing of Visium (in which xdim is doubled and odd-indexed)
#   rows are offset by 1) to standard array indexing with odd rows implicitly shifted.
def pseudo_hex_to_oddr(c):
	x,y = c
	if int(np.rint(y)) % 2 == 1:
		x -= 1
	return [int(np.rint(x//2)),int(np.rint(y))]

# Converts from pseudo-hex indexing of Visium (in which xdim is doubled and odd-indexed)
#   rows are offset by 1) to Cartesian coordinates where neighbors are separated by unit distance.
def pseudo_hex_to_true_hex(c):
	x_arr, y_arr = pseudo_hex_to_oddr(c)

	x = x_arr
	y = y_arr * np.sqrt(3)/2
	if y_arr % 2 == 1:
		x += 0.5
	return [x,y]

''' Determines connected components by recursively checking neighbors in a hex grid.
	bin_oddr_matrix - binary odd-right indexed matrix where 1 indicates annotated spot.
'''
def connected_components_hex(bin_oddr_matrix):
	lmat = np.zeros_like(bin_oddr_matrix, dtype=int)
	lmax = 0

	# Returns immediate neighbors of a coordinate in an odd-right hex grid index.
	def neighbors(cor):
		N = []
		# Spots on even-numbered rows have the following adjacency:
		# [[1,1,0],[1,1,1],[1,1,0]]
		if cor[1] % 2 == 0:
			offsets = [[-1,-1],[0,-1],[-1,0],[1,0],[-1,1],[0,1]]
		# Spots on odd-numbered rows have the following adjacency:
		# [[0,1,1],[1,1,1],[0,1,1]]
		else:
			offsets = [[0,-1],[1,-1],[-1,0],[1,0],[0,1],[1,1]]
		# Find all valid neighbors (within image bounds and present in binary array).
		for o in offsets:
			q = np.array(cor) + np.array(o)
			if q[0]>=0 and q[1]>=0 and q[0]<bin_oddr_matrix.shape[1] and q[1]<bin_oddr_matrix.shape[0]:
				if bin_oddr_matrix[q[1],q[0]] == 1:
					N.append(q)
		return N

	# Find set of all spots connected to a given coordinate.
	def neighborhood(cor, nmat):
		nmat[cor[1],cor[0]] = True

		N = neighbors(cor)
		if len(N)==0:
			return nmat
		for q in N:
			if not nmat[q[1],q[0]]:
				neighborhood(q, nmat)
		return nmat

	# Default recursion limit is 999 -- if there are more than 1k spots on grid we want to
	# allow for all of them to be traversed.
	sys.setrecursionlimit(np.size(bin_oddr_matrix))

	# Determine neighborhood of each unlabled spot, assign a label, and proceed.
	for y in range(bin_oddr_matrix.shape[0]):
		for x in range(bin_oddr_matrix.shape[1]):
			if bin_oddr_matrix[y,x]==1 and lmat[y,x]==0:
				nmat = neighborhood([x,y], np.zeros_like(bin_oddr_matrix, dtype=bool))
				lmax += 1
				lmat[nmat] = lmax

	return lmat, lmax

''' Analog of detect_tissue_sections for hexagonally packed ST grids (Visium)
'''
def detect_tissue_sections_hex(coordinates, check_overlap=False, threshold=120):	
	
	# Convert from spatial hexagonal coordinates to odd-right indexing:
	oddr_indices = np.array(list(map(pseudo_hex_to_oddr, coordinates)))

	xdim, ydim = 64, 78  # Visium arrays have 78 rows of 64 spots each
	bin_oddr_matrix = np.zeros((ydim,xdim))
	for ind in oddr_indices:
		bin_oddr_matrix[ind[1],ind[0]]=1

	labels, n_labels = connected_components_hex(bin_oddr_matrix)

	''' From here on, copy-pasta from utils.detect_tissue_section for removing small components
		and detecting overlap.
	'''

	# get the labels of original spots (before dilation)
	unique_labels,unique_labels_counts = np.unique(labels*bin_oddr_matrix,return_counts=True)

	logging.info('Found %d candidate tissue sections'%(unique_labels.max()+1))

	# this is used to label new tissue sections obtained by watershedding
	max_label = unique_labels.max()+1

	# let us see if there are any tissue sections with unexpected many spots
	if check_overlap:
		for unique_label,unique_label_counts in zip(unique_labels,unique_labels_counts):
			# skip background
			if unique_label == 0:
				continue
			# most likely two tissue sections are slightly overlapping
			elif unique_label_counts >= threshold:
				logging.warning('Tissue section has %d spots. Let us try to break the tissue section into two.'%(unique_label_counts))

				labels = watershed_tissue_sections(unique_label,labels,max_label)
				max_label = max_label + 1

	unique_labels,unique_labels_counts = np.unique(labels*bin_oddr_matrix,return_counts=True)

	# discard tissue sections with less than 10 spots
	for idx in range(0,len(unique_labels_counts)):
		if unique_labels_counts[idx] < 10:
			labels[labels == unique_labels[idx]] = 0
	spots_labeled = labels*bin_oddr_matrix

	# get labels of detected tissue sections
	# and discard skip the background class
	unique_labels = np.unique(spots_labeled)
	unique_labels = unique_labels[unique_labels > 0]

	logging.info('Keeping %d tissue sections'%(len(unique_labels)))

	return unique_labels, spots_labeled

''' Create a boolean vector indicating which spots from the coordinate list belong to the
	tissue section being considered (tissue_idx, spots_tissue_section_labeled obtained by
	connected component analysis in detect_tissue_sections_hex).
'''
def get_tissue_section_spots_hex(tissue_idx, array_coordinates_float, spots_tissue_section_labeled):
	tissue_section_spots = np.zeros(array_coordinates_float.shape[0],dtype=bool)

	for n, chex in enumerate(array_coordinates_float):
		cor = pseudo_hex_to_oddr(chex)

		if spots_tissue_section_labeled[cor[1],cor[0]] == tissue_idx:
			tissue_section_spots[n] = True

	return tissue_section_spots

''' Return spot adjacency matrix given a list of coordinates in pseudo-hex:
'''
def get_spot_adjacency_matrix_hex(coordinates):
	cartesian_coords = np.array(list(map(pseudo_hex_to_true_hex, coordinates)))

	return get_spot_adjacency_matrix(cartesian_coords)


from scipy.ndimage.measurements import label
from splotch.utils import read_array, filter_arrays, detect_tissue_sections, get_tissue_section_spots

import glob

if __name__ == "__main__":
	annot_files = glob.glob('../data/Visium_test/*.csv')
	aars = unique_annots_loupe(annot_files)

	loupe_file = '../data/Visium_test/V014-CGND-MA-00765-A_loupe_AARs.csv'
	position_file = '../data/Visium_test/V014-CGND-MA-00765-A/outs/spatial/tissue_positions_list.csv'

	annot_frame = read_annot_matrix_loupe(loupe_file, position_file, aars)

	array_coordinates_float = np.array([list(map(float, c.split("_"))) for c in annot_frame.columns.values])

	unique_labels, spots_labeled = detect_tissue_sections_hex(array_coordinates_float, True, 600)

	plt.figure()
	plt.imshow(spots_labeled)
	plt.show()

	for tissue_idx in unique_labels:
		tissue_section_spots = get_tissue_section_spots_hex(tissue_idx,array_coordinates_float,
			spots_labeled)

		tissue_section_coordinates_float = array_coordinates_float[tissue_section_spots]
		tissue_section_coordinates_string = ["%.2f_%.2f" % (c[0],c[1]) for c in tissue_section_coordinates_float]

		tissue_section_W = get_spot_adjacency_matrix_hex(tissue_section_coordinates_float)
		print(np.sum(tissue_section_W))

		df = pd.DataFrame(tissue_section_W, index=tissue_section_coordinates_string, 
			columns=tissue_section_coordinates_string)
