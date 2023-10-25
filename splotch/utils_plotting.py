import os
import h5py
import numpy as np
import pandas as pd
import tqdm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from splotch.utils import to_stan_variables
from sklearn.preprocessing import minmax_scale
from functools import cached_property

def oddr_to_pseudo_hex(col, row):
	y_vis = row
	x_vis = col * 2
	if row % 2 == 1:
		x_vis += 1
	return  x_vis, y_vis

def pseudo_hex_to_actual(x, y):
	return x * 0.5, y * np.sqrt(3) / 2

def gene_facet_kdes(gene_summary, sinfo, conditions=None, aars=None, cell_types=None, hue="Condition", row=None, col=None, condition_level=1, **kwargs):
    """
    Plots the distributions of the priors of a given gene, facetted by AAR, Condition, and Cell type

    Parameters
    ----------
    gene_summary : (hdf5) File object
        Gene summary file object.
    sinfo : Obj
        Unpickled information.p.
    conditions : list[str] (default None)
        The conditions to plot, default of None will plot all conditions.
    aars : list[str] (default None)
        The AARs to plot, default of None will plot all AARs.
    cell_types : list[str] (default None)
        The cell types to plot (default of None plots all cell types, only for compositional spot data).
    hue : str
        One of 'Condition', 'AAR', or 'Cell type'.
        Specifies the variable to split into seperate kde distributions with different colors.
    row : str (default None)
        'Condition', 'AAR', or 'Cell type' to facet the rows.
    col : str (default None)
        'Condition', 'AAR', or 'Cell type' to facet the columns.
    condition_level : int (default 1)
        Beta level of the hierarchical model that the specified conditions come from.
    kwargs
        Keyword arguments that are passed to seaborn.displot().
        See https://seaborn.pydata.org/generated/seaborn.displot.html.

    Returns
    -------
    seaborn.FacetGrid (which includes matplotlib figure and axes)
    """

    groups = ['Condition', 'AAR', 'Cell type']
    assert hue in groups, f"'hue' must be set to {groups}"
    assert row is None or row in groups, f"If specified, 'row' must be set to {groups}"
    assert col is None or col in groups, f"If specified, 'col' must be set to {groups}"

    assert condition_level > 0 and condition_level <= sinfo['n_levels'], f"'condition_level' must be set to a value between 1,...,{sinfo['n_levels']}"
    
    beta_level_str = f"beta_level_{condition_level}"
    
    compositional = 'celltype_mapping' in sinfo.keys()

    #set default aars and conditions
    all_aars = sinfo['annotation_mapping']
    all_conditions = sinfo['beta_mapping'][beta_level_str]
    if compositional:
        all_cell_types = sinfo['celltype_mapping']

    #default to all
    if aars == None:
        aars = all_aars
    if conditions == None:
        conditions = all_conditions
    if compositional and cell_types == None:
        cell_types = all_cell_types

    assert set(conditions).issubset(set(all_conditions)), \
        f"The conditions must be a list of elements from the conditions at level {condition_level}: {all_conditions}"

    assert set(aars).issubset(set(all_aars)), \
        f"The aars must be a list of elements from the full list of possible AARs: {all_aars}" 

    if compositional:
        assert set(cell_types).issubset(set(all_cell_types)), \
            f"The cell_types must be a list of elements from the full list of possible cell types: {all_cell_types}" 

    kwargs = {} if kwargs is None else kwargs

    cond_idxs = [to_stan_variables(sinfo['beta_mapping'][beta_level_str], cond) for cond in conditions]
    aar_idxs = [to_stan_variables(sinfo['annotation_mapping'], aar) for aar in aars]
    
    if compositional:
        cell_type_idxs = [to_stan_variables(sinfo['celltype_mapping'], c) for c in cell_types]

    data = pd.DataFrame()
    for cond_idx, cond in zip(cond_idxs, conditions):
        for aar_idx, aar in zip(aar_idxs, aars):
            if compositional:
                for cell_type_idx, cell in zip(cell_type_idxs, cell_types):
                    samples = gene_summary[beta_level_str]['samples'][:, cond_idx, aar_idx, cell_type_idx].flatten()
                    group_df = pd.DataFrame(samples, columns=['Beta'])
                    group_df['Condition'] = cond
                    group_df['AAR'] = aar
                    group_df['Cell type'] = cell
                    data = pd.concat([data, group_df])
            else:
                samples = gene_summary[beta_level_str]['samples'][:, cond_idx, aar_idx].flatten()
                group_df = pd.DataFrame(samples, columns=['Beta'])
                group_df['Condition'] = cond
                group_df['AAR'] = aar
                data = pd.concat([data, group_df])
                

 
    g = sns.displot(data, row=row, col=col, x='Beta', hue=hue, kind='kde', **kwargs)

    g.set_axis_labels(x_var=r'$\beta (log counts)$')

    return g


def lambda_on_sample(gene_summary, gene_r, gene_name, sinfo, library_sample_id, pseudo_to_actual=True, circle_size=10, fig_kw=None, aar_kw=None, raw_count_kw=None, lambda_kw=None):
    """
    Plots the distributions of the priors of a given gene, grouped by conditions or AARs

    Parameters
    ----------
    gene_summary : (hdf5) File object
        Gene summary file object.
    gene_r : dict
        dict producted by read_rdump in splotch.utils
    gene_name : str
        Name of gene to display.
    sinfo : Obj
        Unpickled information.p.
    library_sample_id : str
        Sample to plot.
    pseudo_to_actual : boolean (default True)
        Transform the coordinates from pseudo hex (as used by Visium), to the actual hex coordinates.
    circle_size : float (default 10)
        Passed to parameter 's' in seaborn.scatterplot().
    fig_kw : dict
        Keyword arguments that are passed to pyplot.subplots().
    aar_kw : dict
        Keyword arguments that are passed to seaborn.scatterplot() for the AAR plot.
    raw_count_kw : dict
        Keyword arguments that are passed to seaborn.scatterplot() for the raw count plot.
    lambda_kw : dict
        Keyword arguments that are passed to seaborn.scatterplot() for the lambda plot.
    
    Returns
    -------
    fig, axs
    """
    if 'library_sample_id' in sinfo['metadata'].columns:
        sample_col_key = 'library_sample_id'
    else:
        sample_col_key = 'Name'

    assert library_sample_id in sinfo['metadata'][sample_col_key].values

    #assign defaults for kwargs
    fig_kw = {'constrained_layout': True, 'figsize': (16, 4)} if fig_kw is None else fig_kw
    aar_kw = {} if aar_kw is None else aar_kw
    raw_count_kw = {'palette':'flare'} if raw_count_kw is None else raw_count_kw
    lambda_kw = {'palette':'flare'} if lambda_kw is None else lambda_kw


    f_and_c = np.array(sinfo['filenames_and_coordinates'])

    filenames = f_and_c[:, 0]
    x_y = np.array([xy.split("_") for xy in f_and_c[:, 1]]).astype(float)

    all_spots = pd.DataFrame(x_y, columns=['x', 'y'])
    all_spots['Raw count'] = gene_r['counts']
    all_spots['Lambda'] = gene_summary['lambda']['mean']
    all_spots['aar_idx'] = gene_r['D'] - 1
    all_spots['AAR'] = all_spots['aar_idx'].apply(lambda x: sinfo['annotation_mapping'][x])

    target_filename = sinfo['metadata'][sinfo['metadata'][sample_col_key] == library_sample_id]['Count file'].tolist()[0]
    
    sample_spots = all_spots.iloc[np.where(filenames == target_filename)[0]].copy()
    
    assert sample_spots.shape[0] > 0, f"The array {library_sample_id} did not have spots that were used in the Splotch run"

    if pseudo_to_actual:
        transformed_coords = np.array(sample_spots[['x', 'y']].apply(lambda row: pseudo_hex_to_actual(row[0], row[1]), axis=1).to_list())
    else:
        transformed_coords = np.array(sample_spots[['x','y']])
    sample_spots[['x', 'y']] = transformed_coords

    n_plots = 3
    fig, axs = plt.subplots(ncols=n_plots, **fig_kw)

    sns.scatterplot(sample_spots, x='x', y='y', hue='AAR', ax=axs[0], s=circle_size, edgecolors='face', **aar_kw)
    sns.scatterplot(sample_spots, x='x', y='y', hue='Raw count', ax=axs[1], s=circle_size, edgecolors='face', **raw_count_kw)
    sns.scatterplot(sample_spots, x='x', y='y', hue='Lambda', ax=axs[2], s=circle_size, edgecolors='face', **lambda_kw)
    
    for ax in axs:
        ax.set_aspect('equal')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    
    fig.suptitle(f"Expression of {gene_name} in {library_sample_id}")
    return fig, axs


class CoexpressionModule:
    """
    Stores lambda values and biclustering linkage matrix to help visualize coexpression modules
    """
    def __init__(self, sinfo, gene_summaries_path, conditions, condition_level=1, linkage_method='complete', linkage_metric='correlation', calculate_properties=False):
        """
        Parameters
        ----------
        sinfo : Obj
            Unpickled information.p.
        gene_summaries_path : str
            Path to directoring containing gene summary files.
        conditions : list[str]
            List of beta level conditions to filter by.
        condition_level : int
            Level of hierarchical beta variables to filter by.
        linkage_method : str (default 'complete')
            The linkage algorithm to use for the biclustering.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        linkage_metric : str (default 'correlation')
            The metric to use in the linkage algorithm for the biclustering.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        calculate_properties : boolean (default False)
            Whether to calculate 'lambda_arr' and 'linkage_Z' on initialization.
        """
        all_conditions = sinfo['beta_mapping'][f"beta_level_{condition_level}"]
        assert set(conditions).issubset(set(all_conditions)), \
            f"The conditions must be a list of elements from the conditions at level {condition_level}: {all_conditions}"
        self.sinfo = sinfo
        self.gene_summaries_path = gene_summaries_path
        self.conditions = conditions
        self.condition_level = condition_level
        self.linkage_method = linkage_method
        self.linkage_metric = linkage_metric

        if calculate_properties:
            self.lambda_arr
            self.linkage_Z

    @cached_property
    def lambda_arr(self):
        metadata = self.sinfo['metadata']
        filtered_metadata = metadata[metadata[f'Level {self.condition_level}'].isin(self.conditions)]

        filtered_filenames = filtered_metadata['Count file'].tolist()
        all_filenames = np.array(self.sinfo['filenames_and_coordinates'])[:, 0]

        spot_idxs = np.where(np.isin(all_filenames, filtered_filenames))[0]

        num_genes = len(self.sinfo['genes'])

        lambda_arr = np.zeros((len(spot_idxs), num_genes))

        for gene_idx in tqdm.trange(1, num_genes + 1, desc="Reading gene summaries for 'lambda_arr'"):
            gene_summary = h5py.File(os.path.join(self.gene_summaries_path, str(gene_idx // 100), f'combined_{gene_idx}.hdf5'))
            lambda_arr[:, gene_idx - 1] = gene_summary['lambda']['mean'][spot_idxs]

        return lambda_arr
    
    @cached_property
    def linkage_Z(self):
        Z = linkage(self.lambda_arr.T, method=self.linkage_method, metric=self.linkage_metric)
        return Z
    
    def get_gene_modules(self, threshold=None):
        """
        Finds the co-expression module for each gene using linkage_Z

        Parameters
        ----------
        threshold : float (default None)
            Value to pass to scipy.cluster.hierarchy.fcluster with a criterion of 'distance'.
            None defaults to 0.7*max(linkage_Z[:,2]) (corresponds with MATLAB and scipy default).
        
        Returns
        -------
        fcluster : ndarray
            Array of size (# of genes, 1), where each element is the module the genes belongs to.
        """
        if threshold is None:
            threshold = 0.7*max(self.linkage_Z[:,2]) 
        return fcluster(self.linkage_Z, threshold, criterion="distance")
    

def dendrogram_correlation(coexpression_module: CoexpressionModule, cmap='Spectral', threshold=None):
    """
    Plots a dendrogram (tree) of the coexpression biclustering above a gene-gene correlation heatmap.

    Parameters
    ----------
    coexpression_module : CoexpressionModule
        Coexpression module to use for lambda values and biclustering modules.
    cmap : str (default 'Spectral')
        Matplotlib cmap to use for the gene-gene correlation heatmap
    threshold : float (default None)
        Value to pass to scipy.cluster.hierarchy.fcluster with a criterion of 'distance'.
        None defaults to 0.7*max(linkage_Z[:,2]) (corresponds with MATLAB and scipy default).
    
    Returns
    -------
    (fig, (left_dendrogram_ax, top_dendrogram_ax, heatmap_ax, cbar_ax))
        Figure and axes.
    """

    lambda_arr = coexpression_module.lambda_arr
    Z = coexpression_module.linkage_Z

    if threshold is None:
        threshold = 0.7*max(Z[:,2]) #default threshold according to scipy (and Matlab)
        
    tdp_dict = dendrogram(Z, no_plot=True)
    leaves = tdp_dict['leaves']

    fig = plt.figure(figsize=(5,5))
    # Add an axes at position rect [left, bottom, width, height]
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    dendrogram(Z, show_leaf_counts=False, no_labels=True, ax=ax1, color_threshold=threshold, orientation='left')
    ax1.axvline(threshold, linestyle="--")
    ax1.invert_yaxis()
    ax1.axis('off')

    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    dendrogram(Z, show_leaf_counts=False, no_labels=True, ax=ax2, color_threshold=threshold, orientation='top')
    ax2.axhline(threshold, linestyle="--")
    ax2.axis('off')

    corr = np.corrcoef(lambda_arr, rowvar=False)

    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    axcolor = fig.add_axes([0.94, 0.1, 0.02, 0.6])
    
    sns.heatmap(corr[leaves][:, leaves], yticklabels=False, xticklabels=False, cmap=cmap, vmin=-1, vmax=1, ax=axmatrix, cbar_ax=axcolor, cbar_kws={'label': "Pearson's Correlation Coefficient"})
    fig.suptitle("Co-expression modules and gene-gene correlation")
    return fig, (ax1, ax2, axmatrix, axcolor)


def dendrogram_aars(coexpression_module: CoexpressionModule, sample_gene_r, fig_kw=None, heatmap_kw=None):
    """
    Plots a dendrogram (tree) of the coexpression biclustering above an AAR vs gene expression heatmap.

    Parameters
    ----------
    coexpression_module : CoexpressionModule
        Coexpression module to use for lambda values and biclustering modules.
    sample_gene_r : dict
        dict producted by read_rdump in splotch.utils - used to obtain spots' AAR labels.
    fig_kw : dict (default None)
        Keyword arguments that are passed to pyplot.figure().
    heatmap_kw : dict (default None)
        Keyword arguments passed to the heatmap of AARs vs genes.

    Returns
    -------
    (fig, (dendrogram_ax, heatmap_ax))
        Figure and axes.
    """

    lambda_arr = coexpression_module.lambda_arr
    Z = coexpression_module.linkage_Z
    sinfo = coexpression_module.sinfo
    lambda_conditions = coexpression_module.conditions
    condition_level = coexpression_module.condition_level

    fig_kw = {} if fig_kw is None else fig_kw
    heatmap_kw = {} if heatmap_kw is None else heatmap_kw

    genes = sinfo['genes']
    metadata = sinfo['metadata']
    filtered_metadata = metadata[metadata[f'Level {condition_level}'].isin(lambda_conditions)]

    filtered_filenames = filtered_metadata['Count file'].tolist()
    all_filenames = np.array(sinfo['filenames_and_coordinates'])[:, 0]

    spot_idxs = np.where(np.isin(all_filenames, filtered_filenames))[0]
    
    aar_idx = sample_gene_r['D'] - 1
    aars = np.array([sinfo['annotation_mapping'][idx] for idx in aar_idx])
    spot_aars = aars[spot_idxs]

    all_aars = sinfo['annotation_mapping']
    aars_by_genes = np.zeros((len(all_aars), len(genes)))

    for i, aar in enumerate(all_aars):
        aar_idxs = np.where(spot_aars == aar)[0]
        aars_by_genes[i, :] = np.mean(lambda_arr[aar_idxs], axis=0)

    aars_by_genes = minmax_scale(aars_by_genes, feature_range=(0,1), axis=0)

    
    fig = plt.figure(constrained_layout=True, **fig_kw)

    gs = gridspec.GridSpec(nrows=2, ncols=1,
                       height_ratios=[1, 3],
                       figure=fig)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    tdp_dict = dendrogram(Z, show_leaf_counts=False, no_labels=True, ax=ax1)
    leaves = tdp_dict['leaves']
    
    t = 0.7*max(Z[:,2])
    ax1.axhline(t, linestyle="--")
    ax1.axis('off')

    sns.heatmap(aars_by_genes[:, leaves], xticklabels=False, yticklabels=all_aars, ax=ax2, **heatmap_kw)

    return fig, (ax1, ax2)


def modules_on_sample(coexpression_module: CoexpressionModule, library_sample_id, pseudo_to_actual=True, circle_size=10, ncols=4, fig_kw=None, module_kw=None):
    """
    Plots the average [0, 1] normalized gene expression of each module's genes on top of a given sample.

    Parameters
    ----------
    coexpression_module : CoexpressionModule
        Coexpression module to use for lambda values and biclustering modules.
    library_sample_id : str
        Sample ID to plot.
    pseudo_to_actual : boolean (default True)
        Convert spot coordinates from pseudo hex (Spaceranger default) to actual hex coordinates.
    circle_size : float (default 10)
        The point size on the plots corresponding to spots.
    ncols : int (default 4)
        Number of columns to plot in the array of modules.
    fig_kw : dict (default None)
        Keyword arguments that are passed to pyplot.subplots().
    module_kw : dict (default None)
        Keyword arguments passed to the pyplot.scatter() plot of each coexpression module.

    Returns
    -------
    (fig, axs)
        Figure and axes.
    """
    
    lambda_arr = coexpression_module.lambda_arr
    Z = coexpression_module.linkage_Z
    sinfo = coexpression_module.sinfo
    metadata = sinfo['metadata']
    lambda_conditions = coexpression_module.conditions
    condition_level = coexpression_module.condition_level
    
    if 'library_sample_id' in sinfo['metadata'].columns:
        sample_col_key = 'library_sample_id'
    else:
        sample_col_key = 'Name'
    
    assert library_sample_id in metadata[sample_col_key].tolist(), f"Sample {library_sample_id} was not found in the sinfo metadata"
    assert metadata[metadata[sample_col_key] == library_sample_id][f'Level {condition_level}'].tolist()[0] in lambda_conditions, \
        f"The Level {condition_level} condition of sample {library_sample_id} was not a part of the conditions originally used to construct 'coexpression_module' - {lambda_conditions}"

    fig_kw = {} if fig_kw is None else fig_kw
    module_kw = {} if module_kw is None else module_kw
    
    f_and_c = np.array(sinfo['filenames_and_coordinates'])

    filenames = f_and_c[:, 0]

    x_y = np.array([xy.split("_") for xy in f_and_c[:, 1]]).astype(float)

    all_spots = pd.DataFrame(x_y, columns=['x', 'y'])

    target_filename = metadata[metadata[sample_col_key] == library_sample_id]['Count file'].tolist()[0]

    target_idxs = np.where(filenames == target_filename)[0]
    sample_spots = all_spots.iloc[target_idxs].copy()
    
    assert sample_spots.shape[0] > 0, f"The array {library_sample_id} did not have spots that were used in the Splotch run"

    if pseudo_to_actual:
        transformed_coords = np.array(sample_spots[['x', 'y']].apply(lambda row: pseudo_hex_to_actual(row[0], row[1]), axis=1).to_list())
    else:
        transformed_coords = np.array(sample_spots[['x','y']])
    sample_spots[['x', 'y']] = transformed_coords


    #have to match lambda_arr filter so the spot idxs line up
    filtered_metadata = metadata[metadata[f'Level {condition_level}'].isin(lambda_conditions)]
    filtered_filenames = filtered_metadata['Count file'].tolist()
    long_filtered_filenames = filenames[np.isin(filenames, filtered_filenames)]
    filtered_idxs = np.where(long_filtered_filenames == target_filename)[0]
    target_lambda_arr = lambda_arr[filtered_idxs]

    scaled_lambdas = minmax_scale(target_lambda_arr, (0,1))

    gene_modules = coexpression_module.get_gene_modules()

    module_list = np.unique(gene_modules)

    x = sample_spots['x'].tolist()
    y = sample_spots['y'].tolist()

    n_plots = len(module_list)
    fig, axs = plt.subplots(ncols=ncols, nrows=np.ceil(n_plots/ ncols).astype(int), figsize=(30,30), **fig_kw)
    axs = np.ravel(axs)

    for i, module in enumerate(module_list):
        genes_in_m = np.where(gene_modules == module)[0]
        scaled_expr = minmax_scale(np.mean(scaled_lambdas[:, genes_in_m], axis=1), (0,1)).tolist()
        im = axs[i].scatter(x=x, y=y, c=scaled_expr, s=circle_size, edgecolors='face', **module_kw)
        axs[i].set_aspect('equal')
        axs[i].set_title(f"Module {module}")
        fig.colorbar(im, ax=axs[i])
    
    #iterate through the last plots to make them blank
    num_remaining = ncols - (n_plots % ncols)
    for idx in range(-1, -(num_remaining+1), -1):
        axs[idx].set_visible(False)

    return fig, axs