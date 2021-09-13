import os, sys
import pandas as pd
import scanpy as sc
from pathlib import Path
from argparse import ArgumentParser


parser = ArgumentParser('Convert from Splotch-formatted count files to Stereoscope-formatted input files')
parser.add_argument('-d', '--dest-dir', type=str, required=True, help='Directory in which to save generated Stereoscope files.')
parser.add_argument('-s', '--splotch-files', type=str, nargs='+', help='Splotch-formatted count file(s).')
parser.add_argument('-c', '--cell-file', type=str, help='Scanpy-formatted AnnData h5ad file.')
parser.add_argument('-l', '--label', type=str, help='Observational label to use from cell_file.')
args = parser.parse_args()

if args.splotch_files is not None:
	for sf in args.splotch_files:
		df_splotch = pd.read_csv(sf, sep='\t', header=0, index_col=0)
		sample_name = Path(sf).name.split('.')[0]

		cols_new = ['x'.join(cstr.split('_')) for cstr in df_splotch.columns]
		df_splotch.columns = cols_new
		df_splotch = df_splotch.transpose()
		df_splotch.to_csv(os.path.join(args.dest_dir, 'st_cnt_%s.tsv' % sample_name), sep='\t')

if args.cell_file is not None:
	adat = sc.read_h5ad(args.cell_file)
	celldat_name = Path(args.cell_file).name.split('.')[0]

	# Count file
	df_cell = pd.DataFrame(data=adat.X.todense(), 
		columns=adat.var_names, 
		index=pd.Index(adat.obs.index, name='cell'))
	df_cell.to_csv(os.path.join(args.dest_dir, 'sc_cnt_%s.tsv' % celldat_name), sep='\t')

	# Annotation file
	df_annots = pd.DataFrame({args.label: adat.obs[args.label]}, 
		index=pd.Index(adat.obs.index, name='cell'))
	df_annots.to_csv(os.path.join(args.dest_dir, 'sc_lbl_%s.tsv' % celldat_name), sep='\t')