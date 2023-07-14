import os
import glob
import subprocess
import numpy as np
import scipy.ndimage as ndi

from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from skimage import feature, measure, segmentation
from skimage.morphology import closing, remove_small_objects

from argparse import ArgumentParser


############### HEADLESS APPLICATION OF ILASTIK MODELS ###############

DENSITY_SUFFIX = '_Probabilities.tiff'
SEGMENT_SUFFIX = '_Objects.tiff'
CELLCLASS_SUFFIX = '_Object Predictions.tiff'

def apply_density_model(ilastik_dir, ilastik_model, he_imgfiles, dest_dir):
	ilastik_exec = os.path.join(ilastik_dir, 'run_ilastik.sh')
	sub_str = '%s --headless --project=%s ' % (ilastik_exec, ilastik_model)
	sub_str += ' '.join(he_imgfiles)
	sub_str += ' --output_filename_format=%s/{nickname}%s' % (dest_dir, DENSITY_SUFFIX)

	subprocess.run(sub_str, shell=True, check=True)

def apply_cellclass_model(ilastik_dir, ilastik_model, he_imgfiles, seg_imgfiles, dest_dir):
	ilastik_exec = os.path.join(ilastik_dir, 'run_ilastik.sh')
	sub_str = '%s --headless --project=%s' % (ilastik_exec, ilastik_model)
	sub_str += ' --raw_data %s' % ' '.join(he_imgfiles)
	sub_str += ' --prediction_maps %s' % ' '.join(seg_imgfiles)
	sub_str += ' --output_filename_format="%s/{nickname}%s"' % (dest_dir, CELLCLASS_SUFFIX)

	subprocess.run(sub_str, shell=True, check=True)


############### SEGMENTATION AND CLEANING OF FOREGROUND ###############

# Segments nuclei from a nuclear density map and removes small objects
def segment_density_map(dens_img, otsu_connectivity=7, min_size=50):
	'''
	Parameters:
	----------
	dens_img: ndarray
		single-channel image with pixel values representing probability of nucleus
	otsu_connectivity: int
		connectivity used in identification of local maxima (potential nuclear centroids)
	min_size: int
		minimum size (pixels) for object to keep

	Returns:
	-------
	ofile: path
		path to saved segmented image file
	'''
	maskimg = Image.open(dens_img).convert('L')
	maskimg = np.array(maskimg)

	# Initial thresholding
	cells = maskimg > 0.02

	# Distance between cell centroids and background
	distance = ndi.distance_transform_edt(cells)

	# Local maxima from which to watershed
	peak_idx = feature.peak_local_max(distance, min_distance=otsu_connectivity)
	peak_mask = np.zeros_like(maskimg, dtype=bool)
	peak_mask[peak_idx] = True
	markers = measure.label(peak_mask)

	# Segmentation
	segmented_cells = segmentation.watershed(-distance, markers, mask=cells)

	# Remove small objects & binarize
	cleaned = remove_small_objects(segmented_cells, min_size=min_size)
	cleaned = cleaned > 0

	ofile = dens_img.replace(DENSITY_SUFFIX, SEGMENT_SUFFIX)
	Image.fromarray(cleaned).save(ofile)

	return ofile


############### MAIN FUNCTION ###############

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-I', '--ilastik-location', type=str, required=True,
		help='Path to top-level Ilastik directory (Linux) or /Applications/[ilastik_version].app/Contents/ilastik-release/ (Mac)')
	parser.add_argument('-i', '--he-imgfiles', type=str, nargs='+', required=True,
		help='HE images extracted from spot locations in ST data')
	parser.add_argument('-c', '--cellclass-model', type=str, required=True,
		help='Trained cell classification model to be applied to segmented density maps')
	parser.add_argument('-o', '--output-dir', type=str, required=True,
		help='Destination in which to save nuclear density and cell label maps')
	parser.add_argument('-d', '--density-model', type=str, required=False,
		help='Trained nuclear density Ilastik model to be applied to HE images')
	parser.add_argument('-s', '--segment-first', action='store_true', default=False,
		help='If Ilastik model expects segmented image input, perform segmentation programmatically.'\
		'Otherwise, allow Ilastik to segment density map.')
	parser.add_argument('-x', '--otsu-connectivity', type=int, default=7,
		help='Minimum distance (pixels) used in identifying local maxima (nuclear centroids)')
	parser.add_argument('-k', '--min-size', type=int, default=50,
		help='Minimum size (pixels) of foreground objects (nuclei) to be classified')
	args = parser.parse_args()

	apply_density_model(args.ilastik_location, args.density_model,
		args.he_imgfiles, args.output_dir)

	dens_imgfiles = [os.path.join(args.output_dir, Path(he).stem+DENSITY_SUFFIX) for he in args.he_imgfiles]

	if args.segment_first:
		seg_imgfiles = []
		for df in dens_imgfiles:
			sf = segment_density_map(df, otsu_connectivity=args.otsu_connectivity, 
				min_size=args.min_size)
			seg_imgfiles.append(sf)
	else:
		seg_imgfiles = dens_imgfiles

	if args.cellclass_model is not None:
		apply_cellclass_model(args.ilastik_location, args.cellclass_model,
			args.he_imgfiles, seg_imgfiles, args.output_dir)

	'''
	cellclass_lblfiles = [os.path.join(args.output_dir, Path(he).stem+CELLCLASS_SUFFIX) for he in args.he_imgfiles]

	fig, ax = plt.subplots(len(args.he_imgfiles), 3)

	for i,he in enumerate(args.he_imgfiles):
		he_img = Image.open(he)
		ax[i,0].imshow(he_img)

		# Density map
		d1 = dens_imgfiles[i]
		d1_img = Image.open(d1)
		ax[i,1].imshow(d1_img)

		# Object labels
		d2 = cellclass_lblfiles[i]
		d2_img = Image.open(d2)
		ax[i,2].imshow(d2_img)

	plt.show()
	'''
	
