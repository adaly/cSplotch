from __future__ import absolute_import, division, print_function

import sys
import re
import gzip
import logging
import numpy
import h5py
import pandas as pd
import scipy.stats
from scipy.sparse import block_diag, diags, csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import distance_matrix
import sklearn.cluster
try:
  from skimage.segmentation import watershed  # skimage >=0.19
except ImportError:
  from skimage.morphology import watershed  # skimage <0.19
from skimage.feature import peak_local_max

import jax.numpy as np
from jax import grad, jit
try:
  from jax.example_libraries import optimizers
except ImportError:
  from jax.experimental import optimizers
import matplotlib.pyplot as plt

def write_unified_hdf5(outfile, df_sparse, compression='gzip', compression_opts=9):
  hf = h5py.File(outfile, 'w')
  
  # Define the data type for variable-length strings
  dt = h5py.string_dtype(encoding='utf-8')
  # spot names (x_y)
  spots = hf.create_dataset('spots', (len(df_sparse.columns),), dtype=dt)
  spots[:] = df_sparse.columns.values
  # gene names
  genes = hf.create_dataset('genes', (len(df_sparse.index),), dtype=dt)
  genes[:] = df_sparse.index.values
  
  # sparse count matrix 
  cmat = df_sparse.sparse.to_coo()
  hf.create_dataset('rows', data=cmat.row, compression=compression, compression_opts=compression_opts)
  hf.create_dataset('cols', data=cmat.col, compression=compression, compression_opts=compression_opts)
  hf.create_dataset('values', data=cmat.data, compression=compression, compression_opts=compression_opts)
  hf.create_dataset('shape', data=cmat.shape)
  
  hf.close()

def read_unified_hdf5(outfile):
  hf = h5py.File(outfile, 'r')
  genes = [b.decode('utf-8') for b in hf['genes'][:]]
  spots = [b.decode('utf-8') for b in hf['spots'][:]]
  rows = hf['rows'][:]
  cols = hf['cols'][:]
  values = hf['values'][:]
  shape = tuple(hf['shape'][:])
  
  cmat = coo_matrix((values, (rows, cols)), shape=shape)
  df = pd.DataFrame.sparse.from_spmatrix(
      data=cmat,
      index=genes, 
      columns=spots)
  hf.close()
  return df

def to_numeric(num_str):
  num_str = num_str.strip()
  if num_str.startswith('-'):
    return -to_numeric(num_str[1:])
  if num_str.isdigit():
    return int(num_str)
  else:
    return float(num_str)

def read_rdump(filename):
  data_dict = {}
  ndre = re.compile('.*structure\(c\((.*)\), .Dim=c\((.*)\)\)')

  with open(filename, 'r') as f:
    for line in f:
      tokens = line.strip().split('<-')
      key = tokens[0].strip()

      # Single value
      if len(tokens) == 2 and tokens[1] != '':
        data_dict[key] = to_numeric(tokens[1])

      else:
        line = next(f).strip()

        # 1D array
        if line.startswith('c('):
          tokens = line.strip('c()').split(',')
          data_dict[key] = numpy.array([to_numeric(t) for t in tokens if len(t)>0])

        # ND array
        else:
          m = ndre.match(line)
          flat_arr, shape = m.groups()
          flat_arr = numpy.array([to_numeric(t) for t in flat_arr.split(',') if len(t)>0])
          shape = [int(t) for t in shape.split(',')]
          # Inversion of dimensions prior to reshaping necessary to recover intended rows, columns.
          # must be due to differing conventions between R, Python.
          shape.reverse()
          shape = tuple(shape)
          data_dict[key] = flat_arr.reshape(shape).T

  return data_dict

def to_rdump(data,filename):
    with open(filename,'w') as f:
        for key in data:
            tmp = numpy.asarray(data[key])
            if len(tmp.shape) == 0:
                f.write('%s <- %s\n'%(key,str(tmp)))
            elif len(tmp.shape) == 1:
                f.write('%s <-\n c(%s)\n'%(key,
                    ','.join(map(str,tmp))))
            else:
                f.write('%s <-\n structure(c(%s), .Dim=c(%s))\n'%(key,
                    ','.join(map(str,tmp.flatten('F'))),
                    ','.join(map(str,tmp.shape))))

def to_stan_variables(variables,variables_of_interest):
    if not isinstance(variables_of_interest,list):
        return variables.index(variables_of_interest)
    else:
        output = []
        for variable_of_interest in variables_of_interest:
            if not isinstance(variable_of_interest,list):
                output.append(variables.index(variable_of_interest))
            else:
                output.append([variables.index(variable_of_interest_tmp) for variable_of_interest_tmp in variable_of_interest])
        return output

def read_stan_csv(filename,variables,efficient=False):
  # Handle gzipped output, which is far more compact (and easier to store long-term)
  if filename.endswith('.gz'):
    open_fn = gzip.open
  else:
    open_fn = open

  # For small files, pandas is fine:
  if not efficient:
    data = pd.read_csv(open_fn(filename),sep=',',index_col=False,
      comment='#',header=0,na_filter=False,
      usecols=lambda x: x.startswith(tuple([v+'.' for v in variables])) or x in variables)
    N_samples = data.shape[0]
  
  # For large files, (>4GB or so), use custom reading routine:
  else:
    # First count the number of MCMC samples in the file (cheaply):
    N_samples=0
    with open_fn(filename, "r") as fh:
      for i,line in enumerate(fh):
        if isinstance(line, bytes):
          line=line.decode('UTF-8')
        if not line.startswith('#'):
          N_samples+=1
    N_samples-=1 # Number of samples excluding header line
    
    # Construct a (N_samples x N_variables) matrix, then convert to a pandas dataframe.
    # N_variables are the number of variables in the Splotch posterior estimate that start with one of
    #  the names provided in "variables" -- i.e., beta_level_1 yields all beta parameters at level 1.
    keep_inds=None
    i=0
    with open_fn(filename, "r") as fh:
      for line in fh:
        if isinstance(line, bytes):
          line = line.decode('UTF-8')
        if line.startswith('#'):
          continue
        tokens = line.strip().split(",")
        tokens = numpy.array(tokens)

        # header line
        if keep_inds is None:
          keep_inds = numpy.zeros(len(tokens), dtype=bool)
          for v in variables:
            keep_inds += numpy.array([(s.startswith(v+'.') or s==v) for s in tokens])
          cols = tokens[keep_inds]      
          data = numpy.zeros((N_samples, len(cols)))
        else:
          i+=1
          if len(tokens) == len(keep_inds):
            data[i-1,:] = [float(f) for f in tokens[keep_inds]]
          else:
            print("SKIPPING Row %d (length not consistent)" % (i))
            continue
                
    data = pd.DataFrame(data, columns=cols)

  # initialize a dictionary for storing the samples
  samples = {}
  for variable in variables:
    # if the variable does not exist, let's skip it
    if len([foo for foo in list(data.columns) if foo.startswith(variable)]) == 0:
      continue

    if len([foo for foo in list(data.columns) if foo.startswith(variable+'.')]) == 0:
      dimensions = [N_samples,1]
    # non scalar parameter
    else:
      dimensions = numpy.hstack((N_samples,numpy.array([list(map(int,foo.split('.')[1:])) for foo in list(data.columns) if foo.startswith(variable)]).max(0)))

    samples[variable] = numpy.zeros(dimensions)

  for col in data.columns:
    # get the variable name
    variable = col.split('.')[0]
    # scalar parameter
    if len(col.split('.')) == 1:
      samples[variable][:,0] = data[col].values
    # non scalar parameter
    else:
      # get the indices
      indices = map(lambda x: int(x)-1,col.split('.')[1:])
      # collect mean and standard deviation
      samples[variable][tuple([slice(0,N_samples)])+tuple(indices)] = data[col].values

  return samples

# Accepts a path to a Stan-formatted CSV file containing outputs of Splotch run,
# Generates an HDF5 file structured as:
# /
# * [var1]/
#   * mean/
#   * std/
#   * samples/ (only for var == beta_level_[x])
# * [var2]/ ...
def summarize_stan_hdf5(post_file, out_file, variables=None):
  # If variables not specified, summarize all relevant variables in file
  if variables is None:
    variables = ['log_lambda', 'beta_level_1', 'beta_level_2', 'beta_level_3', 'psi', 
                 'sigma', 'sigma_level_2', 'sigma_level_3', 'theta', 'tau', 'a']
    
  post = read_stan_csv(post_file, variables, efficient=True)
  post_means = dict([(key, post[key].mean(axis=0)) for key in post.keys()])
  post_stds = dict([(key, post[key].std(axis=0)) for key in post.keys()])
    
  h = h5py.File(out_file, 'w')
  for key in post.keys():
    h.create_dataset('%s/mean' % key, data=post_means[key])
    h.create_dataset('%s/std' % key, data=post_stds[key])
        
    if key.startswith('beta'):
      h.create_dataset('%s/samples' % key, data=post[key])

  # Create a separate category for lambda (exponentiated):
  if 'log_lambda' in variables:
    h.create_dataset('lambda/mean', data=numpy.exp(post['log_lambda']).mean(axis=0))
    h.create_dataset('lambda/std', data=numpy.exp(post['log_lambda']).std(axis=0))

  h.close()

def savagedickey(samples1,samples2,prior1_mean=0.0,prior1_std=2.0,prior2_mean=0.0,prior2_std=2.0):
  Delta_theta = (numpy.array([samples1]).T - samples2).flatten()
  density = scipy.stats.kde.gaussian_kde(Delta_theta,bw_method='scott')
  
  numerator = scipy.stats.norm.pdf(0,loc=prior1_mean-prior2_mean,scale=numpy.sqrt(prior1_std**2+prior2_std**2))
  denominator = density.evaluate(0)[0]
  
  if denominator==0:
      return numpy.inf
  return numerator/denominator

def get_variable_mappings(count_files,metadata,
                             levels,n_levels):
  conditions_to_variables = {'beta_level_%d'%(idx+1):[] for idx in range(0,n_levels)}

  level_mappings = [[] for _ in range(0,n_levels-1)]
  last_level_identifiers = {}

  if n_levels == 3:
    levels_1 = levels.keys()
  
    level_2_idx = 1
    level_3_idx = 1
    for level_1_idx,level_1 in enumerate(levels_1,start=1):
      logging.info('beta_level_1[%d] := %s'%(level_1_idx,level_1))
      conditions_to_variables['beta_level_1'].append('%s'%(level_1))
  
      for level_2 in levels[level_1]:
        logging.info('beta_level_2[%d] := %s %s'%(level_2_idx,level_1,level_2))
        conditions_to_variables['beta_level_2'].append('%s %s'%(level_1,level_2))
  
        level_mappings[0].append(level_1_idx)
  
        for level_3 in levels[level_1][level_2]:
          logging.info('beta_level_3[%d] := %s %s %s'%(level_3_idx,level_1,level_2,level_3))
          conditions_to_variables['beta_level_3'].append('%s %s %s'%(level_1,level_2,level_3))
          last_level_identifiers[str([level_1,level_2,level_3])] = level_3_idx
          level_mappings[1].append(level_2_idx)
  
          level_3_idx += 1
  
        level_2_idx += 1

  elif n_levels == 2:
    levels_1 = levels.keys()
  
    level_2_idx = 1
    for level_1_idx,level_1 in enumerate(levels_1,start=1):
      logging.info('beta_level_1[%d] := %s'%(level_1_idx,level_1))
      conditions_to_variables['beta_level_1'].append('%s'%(level_1))
  
      for level_2 in levels[level_1]:
        logging.info('beta_level_2[%d] := %s %s'%(level_2_idx,level_1,level_2))
        conditions_to_variables['beta_level_2'].append('%s %s'%(level_1,level_2))
  
        level_mappings[0].append(level_1_idx)
  
        last_level_identifiers[str([level_1,level_2])] = level_2_idx
  
        level_2_idx += 1

  elif n_levels == 1:
    levels_1 = levels
  
    for level_1_idx,level_1 in enumerate(levels_1,start=1):
      logging.info('beta_level_1[%d] := %s'%(level_1_idx,level_1))
      conditions_to_variables['beta_level_1'].append('%s'%(level_1))
  
      last_level_identifiers[str([level_1])] = level_1_idx
  
  return level_mappings,last_level_identifiers,conditions_to_variables

def n_elements_per_level(node,n,tmp=[]):
  tmp[n] += len(node)
  if not isinstance(node,dict):
      return
  for key, item in node.items():
    if isinstance(item,dict):
      n_elements_per_level(item,n+1,tmp)
    else:
      tmp[n+1] += len(item)

def watershed_tissue_sections(unique_label,labels,max_label):
  tmp_labels = 1*labels
  tmp_labels[tmp_labels != unique_label] = 0
  tmp_labels[tmp_labels > 0] = 1

  distance = distance_transform_edt(tmp_labels)
  min_distance = 6

  while True:
    indices = peak_local_max(distance+0.05*numpy.random.rand(tmp_labels.shape[0],
      tmp_labels.shape[1]),min_distance=min_distance,labels=tmp_labels)

    # we are done if we get two distinct classes
    if len(indices) == 2:
      break

    # otherwise reduce the minimum distance
    min_distance = min_distance - 1

    # we failed when min_distance == 0
    if min_distance == 0:
      logging.critical('Watershedding failed!')
      sys.exit(1)

  local_maxi = numpy.zeros(tmp_labels.shape)
  for foo in indices[0:2]:
    local_maxi[foo[0],foo[1]] = 1

  markers = label(local_maxi)[0]
  labels_new = watershed(-distance,markers,mask=tmp_labels)

  if labels_new.max() != 2:
    logging.critical('Watershed gave %d objects instead of 2!'%(labels_new.max()))
    sys.exit(1)
  else:
    logging.info('Watershed gave %d objects!'%(labels_new.max()))

  labels[labels_new == 2] = max_label

  return labels

def print_summary(levels_list,coordinates_list):

  columns = ['Level %d'%(idx+1) for idx in range(0,len(levels_list[0]))]+['Number of spots']

  # create a data frame from data
  summary_df = pd.DataFrame([list(levels)+[len(coordinates)] \
      for levels,coordinates in \
      zip(levels_list,coordinates_list)],columns=columns)

  # more than one level
  if len(columns[0:-2]) > 0:
    logging.info(summary_df.groupby(columns[0:-2]).agg(
      {'Number of spots': ['size','sum'], columns[-2]: 'nunique'}))
  else:
    logging.info(summary_df.groupby(columns[0]).agg(
      {'Number of spots': ['size','sum']}))

def read_aar_matrix(filename, sep='\t'):
  aar_matrix = pd.read_csv(filename,header=0,index_col=0,sep=sep)

  aar_names = list(aar_matrix.index)

  return aar_matrix,aar_names

'''def read_cellcomp_matrix(filename):
  cellcomp_matrix = pd.read_csv(filename,header=0,index_col=0,sep='\t')

  celltype_names = list(cellcomp_matrix.index)

  return cellcomp_matrix, celltype_names'''

def read_cellcomp_matrix(cellcomp_file, array_coordinates_str=None):
  '''
  Read a Splotch-formatted cell composition file
  --------
  Returns
  --------
  (celltypes, spots) matrix of celltype compositions.
    if array_coordinates_str provided, includes only specified spots in specified order.
  (celltyes,) list of celltype names
  '''
  comp_dat = pd.read_csv(cellcomp_file, sep='\t', header=0, index_col=0)
  
  if array_coordinates_str is None:
      return comp_dat, list(comp_dat.index)
  else:
      in_cellcomp = numpy.array([s in comp_dat.columns for s in array_coordinates_str], dtype=bool)
      
      comp_dat_filtered = comp_dat[array_coordinates_str[in_cellcomp]]
      
      return comp_dat_filtered, list(comp_dat_filtered.index)

def read_array(filename):
  # read the count file
  if filename.endswith('.hdf5'):
    count_file = read_unified_hdf5(filename)
  else:
    count_file = pd.read_csv(filename,header=0,index_col=0,sep='\t') 
  # get the gene names
  array_genes = numpy.array(list(count_file.index))
  # get the spot coordinates
  array_coordinates_str = numpy.array(list(count_file.columns))
  array_coordinates_float = numpy.array([list(map(float,coordinate.split('_'))) \
    for coordinate in array_coordinates_str])
  # get the read counts
  array_counts = numpy.array(count_file.values).T
  array_counts_per_spot = array_counts.sum(1)

  return array_genes,array_coordinates_str,array_coordinates_float,array_counts,array_counts_per_spot

def read_array_metadata(metadata,filename,n_levels):
  array_metadata = metadata[metadata['Count file'] == filename]
  array_levels = [array_metadata['Level %d'%(idx+1)].values[0] for idx in range(0,n_levels)]

  return array_levels

def detect_tissue_sections(coordinates,check_overlap=False,threshold=120):
  # let us represent the array as an m x n grid
  # get max row and column value as ints
  max_coordinates = numpy.rint(coordinates.max(axis=0)).astype(int)
  array = numpy.zeros(max_coordinates + 1)

  # let us discretize spot locations and place them on the array
  for coord in coordinates:
    array[int(numpy.round(coord[0])),int(numpy.round(coord[1]))] = 1

  # label the array "image"
  labels,n_labels = label(array,[[0,1,0],[1,1,1],[0,1,0]])

  # get the labels of original spots (before dilation)
  unique_labels,unique_labels_counts = \
    numpy.unique(labels*array,return_counts=True)

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

  unique_labels,unique_labels_counts = \
    numpy.unique(labels*array,return_counts=True)

  # discard tissue sections with less than 10 spots
  for idx in range(0,len(unique_labels_counts)):
    if unique_labels_counts[idx] < 10:
      labels[labels == unique_labels[idx]] = 0
  spots_labeled = labels*array

  # get labels of detected tissue sections
  # and discard skip the background class
  unique_labels = numpy.unique(spots_labeled)
  unique_labels = unique_labels[unique_labels > 0]

  logging.info('Keeping %d tissue sections'%(len(unique_labels)))

  return unique_labels,spots_labeled

def get_spot_adjacency_matrix(coordinates, is_hd=False):
  # Pairwise distance calculation is prohibitively slow for HD data (>50k spots);
  # instead use efficient hashing strategy below
  if is_hd:
    return get_spot_adjacency_matrix_4n(coordinates)
  else:
    # get a matrix containing all pair-wise distances between spots
    dist_matrix = distance_matrix(coordinates,coordinates)

    # generate the spot adjacency matrix by using the 4-neighborhood rule
    # TODO: do something a bit more elegant with the threshold 
    dist_matrix[numpy.logical_or(dist_matrix > 1.2, \
                numpy.isclose(dist_matrix,0))] = numpy.inf
    W = 1.0/dist_matrix
    W[W > 0] = 1

    return W

# Returns an 4-neighbor adjacency matrix for integer-coded coordinates
def get_spot_adjacency_matrix_4n(int_coordinates):
  def _get_4n(c):
    offsets = numpy.array([(0,1), (0,-1), (1,0), (-1,0)])
    nbrs = [c+o for o in offsets if numpy.all((c+o)>=0)]
    return nbrs
  str_coordinates = ['%d_%d' % tuple(c) for c in int_coordinates]
  
  # create hash tables mapping each point to all possible neighbors, as well as index in int_coordinates
  nbr_lookup, idx_lookup = {},{}
  for i,(c,cstr) in enumerate(zip(int_coordinates, str_coordinates)):
    nbr_lookup[cstr] = _get_4n(c)
    idx_lookup[cstr] = i

  n = len(int_coordinates)
  W = scipy.sparse.lil_matrix((n,n))
  # O(4n): populate adjacency matrix
  for i,(c,cstr) in enumerate(zip(int_coordinates, str_coordinates)):
    for nbr in nbr_lookup[cstr]:
      nbr_cstr = '%d_%d' % tuple(nbr)
      if nbr_cstr in idx_lookup.keys():
        j = idx_lookup[nbr_cstr]
        W[i,j] = 1
  return W.tocsr()

def get_counts(gene_idx,N_tissues,counts_list):
  concatenated_counts = []
  for tissue_idx in range(0,N_tissues):
    concatenated_counts = concatenated_counts + list(counts_list[tissue_idx][:,gene_idx])

  return concatenated_counts

# Given an NxN dense representation of an upper-diagonal (adjacency matrix),
# return a (W_n, 2) sparse representation for use with Stan (NOTE: 1-indexed)
def generate_W_sparse(N,W_n,W):

  # Tarmo's manual method: prohibitively slow for large datasets (>200k spots)
  if N < 1000:
    counter = 0
    W_sparse = numpy.zeros((W_n,2))
    for i in range(0,N-1):
      for j in range(i+1,N):
        if W[i,j] == 1:
          W_sparse[counter,0] = i+1
          W_sparse[counter,1] = j+1
          counter += 1
    return W_sparse.astype(int)

  # Faster scipy-based method; should be used in nearly any situation
  else:
    spr, spc = csr_matrix(W).nonzero()
    W_sparse = numpy.vstack((spr+1, spc+1)).T
    # ensure that we only count each pair once!
    W_sparse = numpy.array([x for x in W_sparse if x[0] < x[1]])

    if not W_sparse.shape[0] == W_n:
      logging.critical('Adjacency matrix W does not contain expected number of adjacent pairs W_n!')
      sys.exit(1)

    return W_sparse

# Takes an R-formatted list of adjacent pairs (1-indexed) and converts to (indices, indtr) representation
# (for faster sparse matrix multiplication within CAR component of Stan Splotch model).
def sparse_to_csr_indptr(W_sparse, N_spots):
    '''
    Parameters:
    ----------
    W_sparse: (k, 2) ndarray
        sparse representation of upper-triangular adjacency matrix (row_inds, col_inds)
    N_spots: int
        number of spots (width of dense adjacency matrix)
    Returns:
    -------
    V: (2*k,) ndarray
        column indices for CSR representation of SYMMETRIC adjacency matrix
    U: (n+1,) ndarray
        pointer vector denoting rows of nonzero elements of adjacency matrix
        (see: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr)
    '''
    # The CSR-speedup version expects a symmetric sparse encoding of the adjacency matrix
    r_inds = numpy.concatenate((W_sparse[:,0]-1, W_sparse[:,1]-1))  
    c_inds = numpy.concatenate((W_sparse[:,1]-1, W_sparse[:,0]-1))
    
    data = numpy.ones_like(r_inds, dtype=np.int64)
    cmat = csr_matrix((data, (r_inds, c_inds)), shape=(N_spots, N_spots))
    return cmat.indices+1, cmat.indptr+1

def generate_column_labels(files_list,coordinates_list):
  filenames = [[files_list[r]]*len(coordinates_list[r]) \
    for r in range(0,len(coordinates_list))]
  filenames = [foo for bar in filenames for foo in bar]
  coordinates = numpy.hstack((coordinates_list[:]))
  filenames_coordinates =  list(zip(*[filenames,coordinates]))

  return filenames_coordinates

# Full eigendecomposition of large square matrices (N>10k) for CAR prior calculation is prohibitively expensive
# APPROXIMATE solution: 
# - break tissue into regularly spaced square chunks containing K<10k spots
# - compute eigenvalues of spatial precision matrix (D^(-.5)*W*D^(.5)) in each chunk and concatenate
def eigs_square_chunking(coords_str, chunksize=10000):
    '''
    Parameters:
    ----------
    coords_str: (N,) array of str
        string representation of integer spot coordinates in "x_y" form
    chunksize: int
        maximum number of spots per chunk
    Returns:
    -------
    eigs: (N,) array of float
        sorted list of concatenated eigenvalues from all chunks
    '''
    coords_int = numpy.array([tuple(map(int, c.split('_'))) for c in coords_str])
    xdim, ydim = coords_int[:,0].max(), coords_int[:,1].max()
    eigs_all = []
    
    wc = int(numpy.rint(chunksize**0.5))
    for i in range(0, xdim, wc):
        xinds = numpy.logical_and(coords_int[:,0] >= i, coords_int[:,0] < i+wc)
        for j in range(0, ydim, wc):
            yinds = numpy.logical_and(coords_int[:,1] >= j, coords_int[:,1] < j+wc)
            
            coords_sub = coords_int[numpy.logical_and(xinds, yinds)]
            if len(coords_sub) > 0:
                if len(coords_sub) > chunksize:
                  logging.warning('Improper use of chunking eigenvalue solution -- neighboring spots should be unit-distance apart')

                # Assumes 4-adjacency for now
                W_sub = get_spot_adjacency_matrix_4n(coords_sub)
    
                # From adjacency matrix W, compute eigenvalues of D^(-.5)*W*D^(.5)
                D_sub = W_sub.sum(1).A1.astype(int)
                D_v = diags(1.0/numpy.sqrt(D_sub),0,format='csr')
                R = D_v.dot(W_sub).dot(D_v)
                eigs = numpy.linalg.eigvalsh(R.toarray())
                eigs_all.append(eigs)

    eigs_all = numpy.concatenate(eigs_all)
    eigs_all.sort()
    assert len(eigs_all) == len(coords_str)
    return eigs_all

def scale_and_round(mat):
  x = numpy.maximum(mat, 0)
  x = x / x.sum(axis=1)[:,None]
  x = numpy.around(x, decimals=5)
  #x[:,-1] = 1-numpy.sum(x[:,:-1], axis=1)
  for i in range(x.shape[0]):
    ind_max = numpy.argmax(x[i,:])
    ind_else = numpy.delete(numpy.arange(x.shape[1]), ind_max)
    x[i, ind_max] = 1.0 - numpy.sum(x[i, ind_else])
  assert x.min() >= 0.0
  return x

def generate_dictionary(N_spots_list,N_tissues,N_covariates,
                        N_levels,coordinates_list,
                        size_factors_list,aar_matrix_list, 
                        level_mappings,tissue_mapping_list,
                        W_list,W_n_list,car,zi,nb,
                        compositional,cellcomp_matrix_list):

  data = {'N_spots': N_spots_list,
          'N_tissues': N_tissues,
          'N_covariates': N_covariates,
          'tissue_mapping': tissue_mapping_list,
          'N_levels': len(N_levels),
          'zi': 1*zi,
          'nb': 1*nb,
          'car': 1*car}

  for idx in range(0,len(N_levels)):
      data['N_level_%d'%(idx+1)]  = N_levels[idx]
  for idx in range(len(N_levels),3):
      data['N_level_%d'%(idx+1)]  = 0

  for idx in range(0,len(N_levels)-1):
      data['level_%d_mapping'%(idx+2)] = level_mappings[idx]

  for idx in range(len(N_levels)-1+2,3+1):
      data['level_%d_mapping'%(idx)] = []

  concatenated_size_factors = []
  for tissue_idx in range(0,N_tissues):
    concatenated_size_factors = concatenated_size_factors + \
      list(size_factors_list[tissue_idx])
  data['size_factors'] = concatenated_size_factors

  concatenated_D = []
  for tissue_idx in range(0,N_tissues):
    concatenated_D = concatenated_D + [numpy.where(tissue_section_aar_matrix)[0][0]+1 \
      for tissue_section_aar_matrix in aar_matrix_list[tissue_idx].T]
  data['D'] = concatenated_D

  if compositional:
    concat_E = []
    for tissue_idx in range(0,N_tissues):
      concat_E.append(cellcomp_matrix_list[tissue_idx].T)
    data['E'] = numpy.vstack(concat_E)
    # Ensure that there are no precision errors on the composition simplexes, as Stan is very picky
    data['E'] = numpy.round(data['E'], 8)  # round to 8 decimals (precision of str(float))
    data['E'][:,-1] = 1 - data['E'][:,:-1].sum(axis=1)
    data['E'] = scale_and_round(data['E'])

    data['N_celltypes'] = data['E'].shape[1]

  if car:
    data['W_n']  = [sum(W_n_list)]
    W = block_diag(W_list,format='csr')
    data['W_sparse'] = generate_W_sparse(sum(N_spots_list),data['W_n'][0],W)
    data['D_sparse'] = W.sum(1).A1.astype(int)

    # Computing the eigenvalues for all spots at once scales poorly; prohibitive for >80k spots
    #data['eig_values'] = numpy.linalg.eigvalsh(diags(1.0/numpy.sqrt(data['D_sparse']),
    #  0,format='csr').dot(W).dot(diags(1.0/numpy.sqrt(data['D_sparse']),
    #  0,format='csr')).toarray())

    # Instead, compute eigenvalues for each tissue separately, then concatenate & sort:
    eigval_list = []

    for t, (W_tissue, coords_tissue) in enumerate(zip(W_list, coordinates_list)):
      if W_tissue.shape[0] < 10000:
        W = block_diag([W_tissue],format='csr')
        D_sparse_tissue = W.sum(1).A1.astype(int)
        D_v = diags(1.0/numpy.sqrt(D_sparse_tissue),0,format='csr')
        evs = numpy.linalg.eigvalsh(D_v.dot(W).dot(D_v).toarray())
      else:
        logging.info('Tissue %d exceeds 10k connected points -- CAR prior eigenvalues calculated by chunking' % t)
        evs = eigs_square_chunking(coords_tissue, 10000)
      eigval_list.append(evs)
    data['eig_values'] = numpy.sort(numpy.concatenate(eigval_list))

    # Decomposition for speedup of CAR calculation in newer version of Splotch/cSplotch model:
    V, U = sparse_to_csr_indptr(data['W_sparse'], numpy.sum(data['N_spots']))
    data['U'] = U 
    data['V'] = V

  else:
    data['W_n'] = []
    data['W_sparse'] = numpy.zeros((0,0))
    data['D_sparse'] = []
    data['eig_values'] = []
    data['U'] = []
    data['V'] = []

  return data

def get_tissue_section_spots(tissue_idx,array_coordinates_float,spots_tissue_section_labeled):
  tissue_section_spots = numpy.zeros(array_coordinates_float.shape[0],dtype=bool)
  for n,coord in enumerate(array_coordinates_float):
    if spots_tissue_section_labeled[int(numpy.round(coord[0])), \
      int(numpy.round(coord[1]))] == tissue_idx:
      tissue_section_spots[n] = True

  return tissue_section_spots

def filter_arrays(indices,coordinates_str=None,coordinates_float=None,
                  counts=None,counts_per_spot=None,size_factors=None,
                  aar_matrix=None,W=None,
                  cellcomp_matrix=None):
  output = []

  if coordinates_str is not None:
    output.append(coordinates_str[indices])
  if coordinates_float is not None:
    output.append(coordinates_float[indices,:])
  if counts is not None:
    output.append(counts[indices,:])
  if counts_per_spot is not None:
     output.append(counts_per_spot[indices])
  if size_factors is not None:
     output.append(size_factors[indices])
  if aar_matrix is not None:
    output.append(aar_matrix[:,indices])
  if W is not None:
    output.append(W[indices,:][:,indices])
  if cellcomp_matrix is not None:
    output.append(cellcomp_matrix[:,indices])

  return output

def get_tissue_sections(count_files,metadata,minimum_sequencing_depth=100.0,maximum_number_of_spots_per_tissue_section=2000.0):
  data = {}

  annotation_filename = metadata[metadata['Count file'] == count_files[0]]['Annotation file'].values[0]
  _,aar_names = read_aar_matrix(annotation_filename)
  
  for filename in count_files: 
    # read the count file
    array_genes,array_coordinates_str,array_coordinates_float,array_counts,array_counts_per_spot = \
      read_array(filename)

    # read the spot annotations
    annotation_filename = metadata[metadata['Count file'] == filename]['Annotation file'].values[0]
    array_aar_matrix,array_aar_names = read_aar_matrix(annotation_filename)

    if not numpy.array_equal(array_aar_names,aar_names):
      logging.critical('Mismatch with AAR names! Order of the AARs must match!')
      sys.exit(1)

    # find spots with enough UMIs
    good_spots = array_counts_per_spot >= minimum_sequencing_depth

    # .. additionally find the spots without annotations
    for n,coord in enumerate(array_coordinates_str):
      # discard the spot if it is not present in annotation file
      # or it does not have annotations
      if (coord not in array_aar_matrix.columns) or \
        (array_aar_matrix[coord].sum() == 0):
        good_spots[n] = False

    # let us skip the current array if we have less than 10 spots left
    # useful for troubleshooting
    if good_spots.sum() < 10:
      logging.warning('The array %s will be skipped because it has less than 10 annotated spots!'%(filename))
      continue
  
    # let us focus on the spots with annotations and sufficient sequencing depth
    array_coordinates_str,array_coordinates_float,array_counts,array_counts_per_spot = \
      filter_arrays(good_spots,coordinates_str=array_coordinates_str,
        coordinates_float=array_coordinates_float,counts=array_counts,
        counts_per_spot=array_counts_per_spot)

    # detect distinct tissue sections on the slide and
    # try to separate overlapping tissue sections
    tissue_section_labels,spots_tissue_section_labeled = \
      detect_tissue_sections(array_coordinates_float,True,maximum_number_of_spots_per_tissue_section)

    # loop over the detected tissue sections on the slide
    for tissue_idx in tissue_section_labels:
      # get the indices of the spots assigned to the current
      # tissue section
      tissue_section_spots = get_tissue_section_spots(tissue_idx,array_coordinates_float,
        spots_tissue_section_labeled)

      # get the coordinates of the spots assigned to the current tissue section
      # get the counts of the spots assigned to the current tissue section
      tissue_section_coordinates_str,tissue_section_coordinates_float,tissue_section_counts = \
        filter_arrays(tissue_section_spots,coordinates_str=array_coordinates_str,
          coordinates_float=array_coordinates_float,counts=array_counts)
  
      # get aar matrix for the spots on the current tissue section
      tissue_section_aar_matrix = array_aar_matrix[tissue_section_coordinates_str].values

      if filename not in data:
        data[filename] = []
      data[filename].append({'coordinates':tissue_section_coordinates_str,
                             'coordinates_num':numpy.asarray([list(map(float,spot.split('_'))) for spot in tissue_section_coordinates_str]),
                             'annotations':numpy.asarray([numpy.where(spot)[0][0] for spot in tissue_section_aar_matrix.T])})
            
  return data,aar_names

def registration_individuals(x,y,aar_names,max_iter=10000,aars=None):

  if aars == None:
      aars = range(0,len(aar_names))
    
  aar_indices = [y == aar for aar in aars]
  uti_indices = [np.triu_indices(sum(y == aar),k=1) for aar in aars]
    
  def cost_function(x,y):
    def foo(x,uti): 
      dr = (x[:,uti[0]]-x[:,uti[1]])
      return np.sqrt(np.sum(dr*dr,axis=0)).sum()
    return sum([foo(x[:,aar_indices[aar]],uti_indices[aar]) for aar in range(0,len(aars))])

  def transform(param,x):
    thetas = param[0:len(x)]
    delta_ps = np.reshape(param[len(x):],(2,len(x)))
    return np.hstack([np.dot(np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]),x_s)+np.expand_dims(delta_p,1) for theta,delta_p,x_s in zip(thetas,delta_ps.T,x)])

  def func(param,x,y):
    value = cost_function(transform(param,x),y)
    return value

  loss = lambda param: func(param,x,y)
    
  opt_init, opt_update, get_params = optimizers.adagrad(step_size=1,momentum=0.9)

  @jit
  def step(i, opt_state):
    params = get_params(opt_state)
    g = grad(loss)(params)
    return opt_update(i, g, opt_state)

  net_params = numpy.hstack((numpy.random.uniform(-numpy.pi,numpy.pi,len(x)),numpy.zeros(2*len(x))))
  previous_value = loss(net_params)
  logging.info('Iteration 0: loss = %f'%(previous_value))
  opt_state = opt_init(net_params)
  for i in range(max_iter):
    opt_state = step(i, opt_state)
    if i > 0 and i % 10 == 0:
      net_params = get_params(opt_state)
      current_value = loss(net_params)
      logging.info('Iteration %d: loss = %f'%(i+1,current_value))

      if numpy.isclose(previous_value/current_value,1):
          logging.info('Converged after %d iterations'%(i+1))
          net_params = get_params(opt_state)
          return transform(net_params,x)

      previous_value = current_value

  logging.warning('Not converged after %d iterations'%(i+1))
  net_params = get_params(opt_state)
  return transform(net_params,x)

def registration_consensus(x,y,aar_names):
    
  def transform(theta,x):
    return np.dot(np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]),x)

  values = numpy.zeros((len(aar_names),2))
  for aar in range(0,len(aar_names)):
    Sigma = numpy.cov(x[:,y == aar])
    u,v = numpy.linalg.eigh(Sigma)
    
    # store the greatest eigenvalue and the angle of the corresponding eigenvector
    values[aar,:] = [u[-1],numpy.pi+numpy.arctan(v[0,-1]/v[1,-1])]
    
  # take the weighted average
  theta = (values[:,0]*values[:,1]).sum()/values[:,0].sum()

  x_registered = transform(theta,x)
  return x_registered-x_registered.mean(1,keepdims=True)

def registration(count_files,metadata,max_iter=10000,aars=None):
  logging.info('Reading data')
  data,aar_names = get_tissue_sections(count_files,metadata)

  coordinates_float = []
  annotations = []
  array_filenames = []
  coordinates_str = []
  for key in data:
    for tissue_section in data[key]:
      coordinates_float.append(tissue_section['coordinates_num'].T)
      annotations.append(tissue_section['annotations'])
      array_filenames += [key]*len(tissue_section['annotations'])
      coordinates_str += list(tissue_section['coordinates'])    
  annotations = numpy.hstack(annotations)
    
  # registration
  coordinates_float = [coordinates-coordinates.mean(1,keepdims=True) for coordinates in coordinates_float]
  logging.info('Aligning the tissue sections')   
  coordinates_float = registration_individuals(coordinates_float,annotations,aar_names,max_iter,aars)
  logging.info('Rotating the consensus spot cloud')   
  coordinates_float = registration_consensus(coordinates_float,annotations,aar_names)
    
  # construct the dictionary holding the registration information
  logging.info('Generating the output data')   
  registered_coordinates = {}
  for idx in range(0,coordinates_float.shape[1]):
    if array_filenames[idx] not in registered_coordinates:
      registered_coordinates[array_filenames[idx]] = {}
    registered_coordinates[array_filenames[idx]][coordinates_str[idx]] = '_'.join(map(str,coordinates_float[:,idx]))

  logging.info('Finished')   
  return registered_coordinates,coordinates_float,annotations
