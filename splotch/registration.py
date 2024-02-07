import os, sys
import glob
import logging
import numpy as np
import pandas as pd

import jax.numpy as jnp
from jax import grad, jit
try:
  from jax.example_libraries import optimizers
except ImportError:
  from jax.experimental import optimizers

from splotch.utils import read_array, read_aar_matrix, detect_tissue_sections, get_tissue_section_spots, filter_arrays
from splotch.utils_visium import detect_tissue_sections_hex, get_tissue_section_spots_hex, pseudo_hex_to_true_hex
from sklearn.preprocessing import LabelEncoder


############### ANNDATA-BASED WORKFLOW ###############

# Given an AnnData object containing annotated data from multiple ST arrays, transform coordinates such
#   that the centroids of the specified annotation categories overlay.
def anndata_overlay_tissues(adata, array_col, x_col, y_col, annot_col, max_spots_per_tissue=None, 
	max_iter=10000, align_aars=None, Visium=True):
	'''
	Parameters:
	----------
	adata: AnnData
		ST data; expects a 'tissue_label' column in adata.obs if max_spots_per_tissue is None
	array_col: str
		column in adata.obs denoting ST array name for each spot
	x_col: str
		column in adata.obs denoting array x-coordinate for each spot
	y_col: str
		column in adata.obs denoting array y-coordinate for each spot
	annot_col: str
		column in adata.obs denoting annotation for each spot
	max_spots_per_tissue: int
		maximum number of spots in an individual tissue section (for watershed separation of overlapping tissues)
	max_iter: int
		maximum number of iterations for alignment procedure
	align_aars: list of str or None
		align tissues by centroids of specified annotation categories, or all annotation categories if None
	Visium: bool
		whether the data is Visium formatted (hex-packed spots) or not (Cartesian)

	Returns:
	-------
	adata: AnnData
		input AnnData with added 'x_reg' and 'y_reg' columns in .obs denoting tissue ID for each spot
	'''
	
	# Separate multiple tissue sections on the same array
	if max_spots_per_tissue is not None:
		adata = anndata_get_tissue_sections(adata, array_col, x_col, y_col, 
			max_spots_per_tissue=max_spots_per_tissue, Visium=Visium)
	elif 'tissue_label' not in adata.obs.columns:
		logging.info('Warning: no "tissue_label" column in adata.obs; assuming one tissue per array')
		adata.obs['tissue_label'] = adata.obs[array_col]

	coords_float, annotations, spot_indices = [],[],[]
	array_filenames = []

	for tissue_idx in adata.obs['tissue_label'].unique():
		# reserved label for tissues that failed watershed segmentation
		if tissue_idx == -1:
			continue

		tissue_spots = adata.obs.index[adata.obs['tissue_label'] == tissue_idx]
		tissue_coordinates_float = np.vstack((adata.obs.loc[tissue_spots, x_col].values, 
			adata.obs.loc[tissue_spots, y_col].values))
		if Visium:
			tissue_coordinates_float = np.array(list(map(pseudo_hex_to_true_hex, tissue_coordinates_float.T))).T
		coords_float.append(tissue_coordinates_float)
		annotations.append(adata.obs.loc[tissue_spots, annot_col])
		spot_indices.append(tissue_spots.values)

	annotations = np.concatenate(annotations)
	spot_indices = np.concatenate(spot_indices)

	# Convert annotations to integer indexes
	le = LabelEncoder()
	annotations_int = le.fit_transform(annotations)
	aar_names = list(le.classes_)

	# Mean-center coordinates of each tissue section prior to registration:
	coords_reg = [c - c.mean(1, keepdims=True) for c in coords_float]

	# Align tissue sections on centroids of desired AARs
	coords_reg = register_individuals(coords_reg, annotations_int, aar_names, align_aars=align_aars, 
		max_iter=max_iter)

	# Rotate consensus spot cloud
	coords_reg = rotate_consensus(coords_reg, annotations_int, aar_names)

	adata.obs['x_reg'] = 0.
	adata.obs.loc[spot_indices, 'x_reg'] = coords_reg[0,:]
	adata.obs['y_reg'] = 0.
	adata.obs.loc[spot_indices, 'y_reg'] = coords_reg[1,:]

	return adata


# Given an AnnData containing ST data, label individual tissues on each array
def anndata_get_tissue_sections(adata, array_col, x_col, y_col, max_spots_per_tissue=2000, Visium=True):
	'''
	Parameters:
	----------
	adata: AnnData
		ST data
	array_col: str
		column in adata.obs denoting ST array name for each spot
	x_col: str
		column in adata.obs denoting array x-coordinate for each spot
	y_col: str
		column in adata.obs denoting array y-coordinate for each spot
	max_spots_per_tissue: int
		maximum number of spots in an individual tissue section (for watershed separation of overlapping tissues)
	Visium: bool
		whether the data is Visium formatted (hex-packed spots) or not (Cartesian)

	Returns:
	-------
	adata: AnnData
		input AnnData with an added 'tissue_label' column in .obs denoting tissue ID for each spot
	'''

	adata.obs['tissue_label'] = -1
	max_tissue_idx = 0

	for arr in adata.obs[array_col].unique():
		array_inds = adata.obs.index[adata.obs[array_col] == arr]
		array_coordinates_float = np.vstack((adata.obs.loc[array_inds, x_col].values, 
			adata.obs.loc[array_inds, y_col].values)).T

		# Detect distinct tissue sections
		try:
			if not Visium:
				tissue_section_labels, spots_tissue_section_labeled = \
					detect_tissue_sections(array_coordinates_float, True, max_spots_per_tissue)
			else:
				tissue_section_labels, spots_tissue_section_labeled = \
					detect_tissue_sections_hex(array_coordinates_float, True, max_spots_per_tissue)

			# Loop over detected tissue sections on the slide
			for tissue_idx in tissue_section_labels:
				if not Visium:
					tissue_section_spots = get_tissue_section_spots(tissue_idx, array_coordinates_float,
						spots_tissue_section_labeled)
				else:
					tissue_section_spots = get_tissue_section_spots_hex(tissue_idx, array_coordinates_float,
						spots_tissue_section_labeled)

				tissue_inds = array_inds[tissue_section_spots]
				adata.obs.loc[tissue_inds, 'tissue_label'] = max_tissue_idx + tissue_idx

			max_tissue_idx += max(tissue_section_labels)+1
		except:
			logging.warning('Watershedding failed -- dropping array %s' % arr)

	return adata


############### TSV FILE WORKFLOW (LEGACY) + HELPER FUNCTIONS ###############

# Accepts a set of annotated data files and transforms coordinates such that centroids of specified
#   annotation categories overlap.
def overlay_tissues(data_files, aar_files, max_iter=10000, align_aars=None, sep_data='\t', sep_aar='\t', Visium=False):
	'''
	Parameters:
	-----------
	data_files: list of paths
		files containing "sep"-delimited (features, spots) matrices. 
		Header must specify spot coordinates as "x_y", while index column contains feature names.
	aar_files: list of paths
		files containing "sep"-delimited (AARs, spots) matrices.
		Ordering of AARs must be consistent across all files.
	max_iter: int
		maximum number of iterations for registration procedure (register_individuals)
	sep_data: char
		delimiting character for data files.
	sep_aar: char
		delimiting character for aar files.
	Visium: bool
		whether coordinates are in Visium format (pseudo-hex) or not (Cartesian).
		Important for connected component analysis used to separate individual tissues.

	Returns:
	----------
	registration_map: dict
		mapping of data files to dict, which in turn maps points in data file 'x_y' to 
		their registered counterparts 'xreg_yreg'.
	coords_reg: (2, N_spots) ndarray
		float array containing concatenation of all registered spot coordinates.
	annotations: (N_spots,) ndarray
		int array containing annotation class for each point in coords_reg.
	aar_names: list of str
		str array containing names of annotation classes.
	'''

	tissue_dict, aar_names = get_tissue_sections(data_files, aar_files, 
		sep_data=sep_data, sep_aar=sep_aar, Visium=Visium)

	coords_float, coords_str, annotations = [],[],[]
	array_filenames = []

	for df in tissue_dict:
		for section in tissue_dict[df]:
			coords_float.append(section['coords_float'].T)
			coords_str += list(section['coords_str'])
			annotations.append(section['annotations'])
			array_filenames += [df]*len(section['annotations'])
	annotations = np.hstack(annotations)

	# Mean-center coordinates of each tissue section prior to registration:
	coords_reg = [c - c.mean(1, keepdims=True) for c in coords_float]

	# Align tissue sections on centroids of desired AARs
	coords_reg = register_individuals(coords_reg, annotations, aar_names, align_aars=align_aars)

	# Rotate consensus spot cloud
	coords_reg = rotate_consensus(coords_reg, annotations, aar_names)

	registration_map = {}
	for idx in range(0,coords_reg.shape[1]):
		if array_filenames[idx] not in registration_map:
			registration_map[array_filenames[idx]] = {}
		registration_map[array_filenames[idx]][coords_str[idx]] = '_'.join(map(str,coords_reg[:,idx]))

	logging.info('Finished')   
	return registration_map, coords_reg, annotations, aar_names


# Plot a desired feature on coordinates obtained from overlay_tissues
def plot_registered_feature(registration_map, data_files, feature_name, sep='\t', 
							vmin=None, vmax=None, alpha=0.9):
	'''
	Parameters:
	-----------
	registration_map: dict
		mapping of data files to dict, which in turn maps points in data file 'x_y' to 
		their registered counterparts 'xreg_yreg'.
	data_files: list of paths
		files containing 'sep'-delimited (features, spots) matrices. 
		Header must specify spot coordinates as "x_y", while index column contains feature names.
	feature_name: str
		name of feature to plot.
	sep: char
		delimiting character in data files.
	vmin: float or None
		minimum value to display on feature map (defaults to min over feature values)
	vmax: float or None
		maximum value to display on feature map (defaults to max over feature values)
	alpha: float
		transparency of points on feature map.

	Returns:
	-----------
	fig, ax: pointers to matplotlib figure and axis objects.
	'''
	reg_coords, feat_vals = [],[]
	
	for c, df in enumerate(data_files):
		if df in registration_map:
			dat = pd.read_csv(df, sep=sep, index_col=0, header=0)
			
			for cstr in registration_map[df]:
				if cstr in dat.columns:
					reg_coords.append(list(map(float, registration_map[df][cstr].split('_'))))
					feat_vals.append(dat.loc[feature_name, cstr])
		
	reg_coords = np.array(reg_coords)
	feat_vals = np.array(feat_vals)
	
	if vmin is None:
		vmin = feat_vals.min()
	if vmax is None:
		vmax = np.percentile(feat_vals, 95)
	
	fig, ax = plt.subplots(1, figsize=(8,8))
	cbmap = ax.scatter(reg_coords[:,0], reg_coords[:,1], c=feat_vals, cmap='viridis', s=4, alpha=alpha,
					  vmin=vmin, vmax=vmax)
	cbar = plt.colorbar(cbmap)
	cbar.set_label(feature_name)
	
	return fig, ax


# Takes in list of (centered) coordinates for each tissue, and outputs transformed coordinates
#   aligned on centroids of specified AARs (align_aars)
def register_individuals(coords, annot_inds, aar_names, align_aars=None, max_iter=10000):
	'''
	Parameters:
	-----------
	coords: list of (N_spots_j, 2) ndarrays
		list of arrays where each entry contains centered coordinates for all spots in array j.
	annot_inds: (N_spots,) ndarray
		array containing concatenation of all spot annotation indices.
	aar_names: list of str
		list mapping AAR indices to names.
	align_aars: list of str, list of int, or None
		list of AARs to use for registration -- either str, which must be members of aar_names,
		or int, which will index into aar_names. If None, all AARs will be used.
	max_iter: int
		maximum number of iterations for registration optimizer.

	Returns:
	----------
	coords_reg: (2, N_spots) ndarray
		float array containing concatenation of all registered spot coordinates.
	'''
	if align_aars == None:
		align_aars = range(0,len(aar_names))
	elif isinstance(align_aars[0], str):
		align_aars = [aar_names.index(aar) for aar in align_aars]

	aar_indices = [annot_inds == aar for aar in align_aars]
	uti_indices = [jnp.triu_indices(sum(annot_inds == aar), k=1) for aar in align_aars]

	def cost_function(x, y):
		def foo(x, uti): 
			dr = (x[:, uti[0]] - x[:, uti[1]])
			return jnp.sqrt(jnp.sum(dr*dr, axis=0)).sum()
		return sum([foo(x[:, aar_indices[aar]], uti_indices[aar]) for aar in range(0, len(align_aars))])

	def transform(param, x):
		thetas = param[0:len(x)]
		delta_ps = jnp.reshape(param[len(x):], (2, len(x)))
		return jnp.hstack(
			[jnp.dot(jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]), x_s) + 
			 jnp.expand_dims(delta_p, 1) for theta, delta_p, x_s in zip(thetas, delta_ps.T, x)])

	def transformed_cost(param, x, y):
		value = cost_function(transform(param, x), y)
		return value

	loss = lambda param: transformed_cost(param, coords, annot_inds)

	opt_init, opt_update, get_params = optimizers.adagrad(step_size=1, momentum=0.9)

	@jit
	def step(i, opt_state):
		params = get_params(opt_state)
		g = grad(loss)(params)
		return opt_update(i, g, opt_state)

	net_params = np.hstack((np.random.uniform(-np.pi, np.pi, len(coords)), np.zeros(2*len(coords))))
	previous_value = loss(net_params)
	logging.info('Iteration 0: loss = %f' % (previous_value))
	opt_state = opt_init(net_params)

	for i in range(max_iter):
		opt_state = step(i, opt_state)
		
		if i > 0 and i % 10 == 0:
			net_params = get_params(opt_state)
			current_value = loss(net_params)
			logging.info('Iteration %d: loss = %f'%(i+1,current_value))

			if np.isclose(previous_value/current_value,1):
				logging.info('Converged after %d iterations'%(i+1))
				net_params = get_params(opt_state)
				return transform(net_params, coords)

			previous_value = current_value

	logging.warning('Not converged after %d iterations'%(i+1))
	net_params = get_params(opt_state)
	return transform(net_params, coords)


# Take in list of aligned coordinates and rotate them s.t. principle axis of variation aligns with Cartesian axis.
def rotate_consensus(coords, annot_inds, aar_names):
	'''
	Parameters:
	-----------
	coords: (2, N_spots) ndarray
		float array containing registered coordinates output by register_individuals.
	annots: (N_spots,) ndarray
		array containing concatenation of all spot annotation indices.
	aar_names: list of str
		list mapping AAR indices to names.

	Returns:
	-----------
	coords_rot: (2, N_spots) ndarray
		float array containing rotated versions of coords.
	'''
	def transform(theta, x):
		return jnp.dot(jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]), x)

	values = np.zeros((len(aar_names), 2))
	for aar in range(0, len(aar_names)):
		Sigma = np.cov(coords[:, annot_inds == aar])
		u,v = np.linalg.eigh(Sigma)

	# Store the greatest eigenvalue and the angle of the corresponding eigenvector
	values[aar,:] = [u[-1], np.pi + np.arctan(v[0,-1]/v[1,-1])]

	# Take the weighted average
	theta = (values[:,0] * values[:,1]).sum() / values[:,0].sum()

	x_registered = transform(theta, coords)
	return x_registered - x_registered.mean(1, keepdims=True)


# Return a dict mapping data files to a list of dicts (one for each tissue section on the array).
# Each tissue dict contains ST coordinates + annotation for each spot in the tissue.
def get_tissue_sections(data_files, aar_files, minimum_spot_val=None, 
	max_spots_per_tissue=2000.0, Visium=False, sep_data='\t', sep_aar='\t'):
	'''
	Parameters:
	-----------
	data_files: list of paths
		files containing "sep"-delimited (features, spots) matrices. 
		Header must specify spot coordinates as "x_y", while index column contains feature names.
	aar_files: list of paths
		files containing "sep"-delimited (AARs, spots) matrices.
		Ordering of AARs must be consistent across all files.
	minimum_spot_val: float or None
		if provided, filter out spots whos feature vectors sum to less than specified value.
		Used to remove low-UMI spots in count data.
	max_spots_per_tissue: int
		maximum number of spots in a distinct tissue section. Used to separate overlapping tissues
		by watershed segmentation.
	Visium: bool
		whether coordinates are in Visium format (pseudo-hex) or not (Cartesian).
		Important for connected component analysis used to separate individual tissues.
	sep_data: char
		delimiting character for data files.
	sep_aar: char
		delimiting character for aar files.
	
	Returns:
	-----------
	tissue_dict: dict
		mapping of data files to sub-dicts for each tissue section contained in said file.
		Each sub-dict contains arrays of str and float representations of spot coordinates
		within a tissue section, as well as an array of corresponding annotations.
	aar_names: list
		list of AAR names as specified in any given annotation file.
	'''
	
	data = {}

	_,aar_names = read_aar_matrix(aar_files[0])

	for df, af in zip(data_files, aar_files): 

		# Read the spot annotations
		array_aar_matrix, array_aar_names = read_aar_matrix(af, sep=sep_aar)

		if not np.array_equal(array_aar_names,aar_names):
			logging.critical('Mismatch with AAR names! Order of the AARs must match!')
			sys.exit(1)

		# Read the spot features, preserving spot ordering from annotation file
		dat = pd.read_csv(df, sep=sep_data, header=0, index_col=0)
		in_features = np.array([s in dat.columns for s in array_aar_matrix.columns])
		feature_matrix = dat[array_aar_matrix.columns[in_features]]

		array_coordinates_str = feature_matrix.columns
		array_coordinates_float = np.array([list(map(float,c.split('_'))) for c in array_coordinates_str])

		# Optionally mark whether spots exceed a total feature value (e.g., spot depth for count data)
		total_spot_val = feature_matrix.values.sum(axis=0)
		if minimum_spot_val is not None:
			good_spots = total_spot_val >= minimum_spot_val
		else:
			good_spots = np.ones(len(total_spot_val), dtype=bool)

		# Mark un-annotated spots
		for n,coord in enumerate(array_coordinates_str):
			if array_aar_matrix[coord].sum() == 0:
				good_spots[n] = False

		# Skip any arrays possessing fewer than 10 annotated spots
		if good_spots.sum() < 10:
			logging.warning('The array %s will be skipped because it has less than 10 annotated spots!'%(df))
			continue

		# Perform filtering
		array_coordinates_str, array_coordinates_float, feature_vals, array_counts_per_spot = \
		filter_arrays(good_spots,coordinates_str=array_coordinates_str,
			coordinates_float=array_coordinates_float,counts=feature_matrix.values.T,
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
			if not Visium:
				tissue_section_spots = get_tissue_section_spots(tissue_idx, array_coordinates_float,
					spots_tissue_section_labeled)
			else:
				tissue_section_spots = get_tissue_section_spots_hex(tissue_idx, array_coordinates_float,
					spots_tissue_section_labeled)

			# Get the coordinates & features of the spots assigned to the current tissue section
			tissue_section_coordinates_str, tissue_section_coordinates_float, tissue_section_features = \
			filter_arrays(tissue_section_spots,coordinates_str=array_coordinates_str, 
				coordinates_float=array_coordinates_float,counts=feature_vals)

			# Get aar matrix for the spots on the current tissue section
			tissue_section_aar_matrix = array_aar_matrix[tissue_section_coordinates_str].values

			if df not in data:
				data[df] = []
			data[df].append({
				'coords_str':tissue_section_coordinates_str,
				'coords_float':np.asarray([list(map(float,spot.split('_'))) for spot in tissue_section_coordinates_str]),
				'annotations':np.asarray([np.where(spot)[0][0] for spot in tissue_section_aar_matrix.T]),
				})

	return data, aar_names


if __name__ == '__main__':
	metafile = os.path.expanduser('~/ceph/datasets/csplotch_mouse_st/metadata_compositional_WTG93A.tsv')
	metadata = pd.read_csv(metafile, sep='\t')[:20]

	count_files = [s.replace('..', '/mnt/home/adaly/ceph/datasets/') for s in metadata['Count file']]
	comp_files = ['/mnt/home/adaly/ceph/datasets/csplotch_mouse_st/' + s for s in metadata['Composition file']]
	annot_files = [s.replace('..', '/mnt/home/adaly/ceph/datasets/') for s in metadata['Annotation file']]

	overlay_tissues(count_files, annot_files, Visium=False, align_aars=['Dors_Horn', 'Vent_Horn'])
	#overlay_tissues(comp_files, annot_files, Visium=False, align_aars=['Dors_Horn', 'Vent_Horn'])



	