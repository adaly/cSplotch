import os
import glob
import logging
import numpy as np
import pandas as pd

from utils import read_array, read_aar_matrix, get_tissue_section_spots
from utils_visium import get_tissue_sections_hex


def overlay_tissues(data_files, aar_files, max_iter=10000, align_aars=None, sep='\t', Visium=False):
	'''
	Parameters:
	-----------
	data_files: list of paths
		list of paths to files containing "sep"-delimited (features, spots) matrices.
		header must specify spot coordinates as "x_y", while index column contains feature names.
	'''

	get_tissue_sections(data_files, aar_files, Visium=Visium)

# Return a dict mapping data files to a list of dicts (one for each tissue section on the array).
# Each tissue dict contains ST coordinates + annotation for each spot in the tissue.
def get_tissue_sections(data_files, aar_files, minimum_spot_val=None, 
	max_spots_per_tissue=2000.0, Visium=False):
	data = {}

	_,aar_names = read_aar_matrix(aar_files[0])

	for df, af in zip(data_files, aar_files): 
		feature_names, array_coordinates_str, array_coordinates_float, feature_vals, total_spot_val = \
		read_array(df)

		# Read the spot annotations
		array_aar_matrix, array_aar_names = read_aar_matrix(af)

		if not np.array_equal(array_aar_names,aar_names):
			logging.critical('Mismatch with AAR names! Order of the AARs must match!')
			sys.exit(1)

		# Optionally mark whether spots exceed a total feature value (e.g., spot depth for count data)
		if minimum_spot_val is not None:
			good_spots = total_spot_val >= minimum_spot_val
		else:
			good_spots = np.ones(len(total_spot_val), dtype=bool)

		# Mark un-annotated spots
		for n,coord in enumerate(array_coordinates_str):
			if (coord not in array_aar_matrix.columns) or (array_aar_matrix[coord].sum() == 0):
				good_spots[n] = False

		# Skip any arrays possessing fewer than 10 annotated spots
		if good_spots.sum() < 10:
			logging.warning('The array %s will be skipped because it has less than 10 annotated spots!'%(filename))
			continue

		# Perform filtering
		array_coordinates_str, array_coordinates_float, feature_vals, array_counts_per_spot = \
		filter_arrays(good_spots,coordinates_str=array_coordinates_str,
			coordinates_float=array_coordinates_float,counts=feature_vals,
			counts_per_spot=total_spot_val)

		# Detect distinct tissue sections
		if not Visium:
			tissue_section_labels, spots_tissue_section_labeled = \
				detect_tissue_sections(array_coordinates_float, True, max_spots_per_tissue)
		else:
			tissue_section_labels, spots_tissue_section_labeled = \
				detect_tissue_sections_hex(array_coordinates_float, True, max_spots_per_tissue)

		# Loop over detected tissue sections on the slide
		for tissue_idx in tissue_section_labels:
			tissue_section_spots = get_tissue_section_spots(tissue_idx, array_coordinates_float,
				spots_tissue_section_labeled)

			# Get the coordinates & features of the spots assigned to the current tissue section
			tissue_section_coordinates_str, tissue_section_coordinates_float, tissue_section_features = \
			filter_arrays(tissue_section_spots,coordinates_str=array_coordinates_str, 
				coordinates_float=array_coordinates_float,counts=feature_vals)

		# Get aar matrix for the spots on the current tissue section
		tissue_section_aar_matrix = array_aar_matrix[tissue_section_coordinates_str].values

		if df not in data:
			data[df] = []
		data[df].append({'coords_str':tissue_section_coordinates_str,
			'coords_float':numpy.asarray([list(map(float,spot.split('_'))) for spot in tissue_section_coordinates_str]),
			'annotations':numpy.asarray([numpy.where(spot)[0][0] for spot in tissue_section_aar_matrix.T])})

	return data, aar_names


if __name__ == '__main__':
	pass
	