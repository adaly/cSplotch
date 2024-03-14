import os
import h5py
import numpy as np
import pandas as pd
import numpy as np
import tqdm
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.stats
from splotch.utils import to_stan_variables
from splotch.utils_sc import grouped_obs_mean_std
from sklearn.preprocessing import minmax_scale
from functools import cached_property

# Compute mean expression within a cell group
def grouped_obs_mean(adata, group_key, layer=None, gene_symbols=None):
    if layer is not None:
        getX = lambda x: x.layers[layer]
    else:
        getX = lambda x: x.X
    if gene_symbols is not None:
        new_idx = adata.var[gene_symbols].values
    else:
        new_idx = adata.var_names

    grouped = adata.obs.groupby(group_key, observed=False)
    
    out = {}
    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        out[group] = np.ravel(X.mean(axis=0, dtype=np.float64))    
    out = pd.DataFrame(out, index=new_idx)
    return out

def idxs_to_genes(splotch_gene_idxs, gene_idxs_csv_path, gene_csv_col='gene'):
    gene_idxs = pd.read_csv(gene_idxs_csv_path, index_col=0)
    
    return gene_idxs[np.isin(gene_idxs.index, splotch_gene_idxs)][gene_csv_col].tolist()

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
    Plots the distributions of the priors of a given gene, facetted by AAR, Condition, and Cell type.

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
        Keyword arguments that are passed to `seaborn.displot()`.
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
        dict producted by `read_rdump` in splotch.utils
    gene_name : str
        Name of gene to display.
    sinfo : Obj
        Unpickled information.p.
    library_sample_id : str
        Sample to plot.
    pseudo_to_actual : boolean (default True)
        Transform the coordinates from pseudo hex (as used by Visium), to the actual hex coordinates.
    circle_size : float (default 10)
        Passed to parameter 's' in `seaborn.scatterplot()`.
    fig_kw : dict
        Keyword arguments that are passed to `pyplot.subplots()`.
    aar_kw : dict
        Keyword arguments that are passed to `seaborn.scatterplot()` for the AAR plot.
    raw_count_kw : dict
        Keyword arguments that are passed to `seaborn.scatterplot()` for the raw count plot.
    lambda_kw : dict
        Keyword arguments that are passed to `seaborn.scatterplot()` for the lambda plot.
    
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
    def __init__(self, sinfo, gene_summaries_path, conditions, condition_level=1, linkage_method='average', linkage_metric='cityblock', calculate_properties=False, threshold=None):
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
        linkage_method : str (default 'average')
            The linkage algorithm to use for the biclustering.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        linkage_metric : str (default 'cityblock')
            The metric to use in the linkage algorithm for the biclustering.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        calculate_properties : boolean (default False)
            Whether to calculate `lambda_arr` and `linkage_Z` on initialization.
        threshold : float (default None)
            Value to pass to `scipy.cluster.hierarchy.fcluster` with a criterion of 'distance'.
            None defaults to `0.54*max(linkage_Z[:,2])` (same value used by https://www.science.org/doi/10.1126/science.aav9776)
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

        self._threshold = threshold

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
    
    @property
    def threshold(self):
        if self._threshold is not None:
            return self._threshold
        
        return 0.54*max(self.linkage_Z[:,2]) 

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    @cached_property
    def linkage_Z(self):
        Z = linkage(self.lambda_arr.T, method=self.linkage_method, metric=self.linkage_metric)
        return Z
    
    def save_lambda_arr(self, path):
        np.save(path, self.lambda_arr)

    def save_linkage_Z(self, path):
        np.save(path, self.linkage_Z)

    def load_lambda_arr(self, path):
        self.lambda_arr = np.load(path)
    
    def load_linkage_Z(self, path):
        self.linkage_Z = np.load(path)

    def get_gene_modules(self):
        """
        Finds the co-expression module for each gene using linkage_Z

        Returns
        -------
        fcluster : ndarray
            Array of size (# of genes, 1), where each element is the module the genes belongs to.
        """
        return fcluster(self.linkage_Z, self.threshold, criterion="distance")
    
    def genes_in_module(self, module_num, gene_idxs_csv_path):
        """
        Finds the names of the genes in the given module number.

        Parameters
        ----------
        module_num : int
            Number of the module to obtain genes from.
        gene_indxs_csv_path : str
            Path of a csv where the first column is the cSplotch index of a gene,
            and there is another column titled 'gene' which lists the name of the gene.

        Returns
        -------
        genes : list[str]
            List of names of genes in module `module_num`.
        """
        modules = self.get_gene_modules()
        unique_modules = np.unique(modules)

        assert module_num <= len(unique_modules) and module_num >= 1
        idxs = np.where(modules == module_num)[0]
        
        genes = idxs_to_genes(idxs, gene_idxs_csv_path)
        return genes

#from splotch.utils_plotting import dendrogram_correlation
def dendrogram_correlation(coexpression_module: CoexpressionModule, cmap='Spectral'):
    """
    Plots a dendrogram (tree) of the coexpression biclustering above a gene-gene correlation heatmap.

    Parameters
    ----------
    coexpression_module : CoexpressionModule
        Coexpression module to use for lambda values and biclustering modules.
    cmap : str (default 'Spectral')
        Matplotlib cmap to use for the gene-gene correlation heatmap
    
    Returns
    -------
    `seaborn.ClusterGrid`
        Figure and axes.
    """

    clusters = coexpression_module.get_gene_modules()
    k = len(np.unique(clusters))
    
    #use tab20 as colors for clusters
    cluster_cmap = plt.get_cmap('tab20')
    cNorm  = colors.Normalize(vmin=0, vmax=20)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cluster_cmap)
    k2col = [scalarMap.to_rgba(np.remainder(i, 20)) for i in range(k)]

    corr = np.corrcoef(coexpression_module.lambda_arr, rowvar=False)
    cg = sns.clustermap(corr, row_linkage=coexpression_module.linkage_Z, col_linkage=coexpression_module.linkage_Z,
               row_colors=[k2col[i-1] for i in clusters], col_colors=[k2col[i-1] for i in clusters],
               cmap=cmap, 
               cbar_kws={'label':'Pearson correlation', 'ticks':[-1, 0, 1]},
               xticklabels=False, 
               yticklabels=False,
              )
    cg.ax_row_dendrogram.set_visible(False)
    
    #set module labels
    labelled_clusters = [m for m in np.unique(clusters)]
    sorted_clusters = np.sort(clusters)
    tick_positions = [0.5 * (np.where(sorted_clusters == m)[0][0] + np.where(sorted_clusters == m)[0][-1]) for m in labelled_clusters]
    cg.ax_col_colors.set_xticks(tick_positions)
    cg.ax_col_colors.set_xticklabels(labelled_clusters)
    cg.ax_col_colors.xaxis.set_tick_params(size=0, pad=-15) # make tick marks invisible

    #add purple line for cutoff
    t = coexpression_module.threshold
    cg.ax_col_dendrogram.axhline(t, color='purple', linestyle='--')

    return cg


def violin_modules(coexpression_module: CoexpressionModule, sample_gene_r, module_num, x, hue=None, **kwargs):
    """
    Creates violin plot of the z-score scaled expression levels of the given module, split by the x and hue values.

    Parameters
    ----------
    coexpression_module : CoexpressionModule
        Coexpression module to use for lambda values and biclustering modules.
    sample_gene_r : dict
        dict producted by `read_rdump` in `splotch.utils` - used to obtain spots' AAR labels.
    module_num : int
        The module to plot.
    x : str
        One of "AAR", "Level 1", "Level 2" or "Level 3". Dictates the variable to plot on the x axis.
    hue : str (default None)
        Can be one of "AAR", "Level 1", "Level 2" or "Level 3". Optionally specifies the colors of the violinplots.
    kwargs
        Keyword arguments that are passed to `seaborn.violinplot()`.
        See https://seaborn.pydata.org/generated/seaborn.violinplot.html.

    Returns
    -------
    violinplot axis
    """
    groups = ['AAR', 'Level 1', 'Level 2', 'Level 3']
    assert x in groups, f"'x' must be one of {groups}"
    assert hue is None or hue in groups, f"'hue' must be None or one of {groups}"

    lambda_arr = coexpression_module.lambda_arr
    Z = coexpression_module.linkage_Z
    sinfo = coexpression_module.sinfo
    lambda_conditions = coexpression_module.conditions
    condition_level = coexpression_module.condition_level

    #get avg zscores for each spotr
    clusters = coexpression_module.get_gene_modules()

    cluster_idxs = np.where(clusters == module_num)[0]

    lambda_subset = lambda_arr[:, cluster_idxs]

    zscores = scipy.stats.zscore(lambda_subset, axis=0)
    avg_zscores = np.mean(zscores, axis=1)

    #obtain metadata information about each spot
    metadata = sinfo['metadata']
    filtered_metadata = metadata[metadata[f'Level {condition_level}'].isin(lambda_conditions)]

    filtered_filenames = filtered_metadata['Count file'].tolist()
    all_filenames = np.array(sinfo['filenames_and_coordinates'])[:, 0]

    spot_idxs = np.where(np.isin(all_filenames, filtered_filenames))[0]
    
    aar_idx = sample_gene_r['D'] - 1
    aars = np.array([sinfo['annotation_mapping'][idx] for idx in aar_idx])
    spot_aars = aars[spot_idxs]

    data = pd.DataFrame()
    data['AAR'] = spot_aars
    data['Count file'] = all_filenames[spot_idxs]
    data = data.merge(metadata, on='Count file')
    data.index.name = 'spot_idx'
    data.index = spot_idxs
    data['z-score'] = avg_zscores

    kwargs = {} if kwargs is None else kwargs
    return sns.violinplot(data=data, x=x, hue=hue, y='Average z-score expression level', **kwargs)

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
        Keyword arguments that are passed to `pyplot.subplots()`.
    module_kw : dict (default None)
        Keyword arguments passed to the `pyplot.scatter()` plot of each coexpression module.

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

    fig_kw = {"figsize": (30,30)} if fig_kw is None else fig_kw
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
    fig, axs = plt.subplots(ncols=ncols, nrows=np.ceil(n_plots/ ncols).astype(int), **fig_kw)
    axs = np.ravel(axs)

    for i, module in enumerate(module_list):
        genes_in_m = np.where(gene_modules == module)[0]
        scaled_expr = minmax_scale(np.mean(scaled_lambdas[:, genes_in_m], axis=1), (0,1)).tolist()
        im = axs[i].scatter(x=x, y=y, c=scaled_expr, s=circle_size, edgecolors='face', **module_kw)
        axs[i].set_aspect('equal')
        axs[i].set_title(f"Module {module}")
        fig.colorbar(im, ax=axs[i])
    
    #iterate through the last plots to make them blank
    if n_plots % ncols != 0:
        num_remaining = ncols - (n_plots % ncols)
        for idx in range(-1, -(num_remaining+1), -1):
            axs[idx].set_visible(False)

    return fig, axs


def plot_submodules(adata, gene_list, obs_celltype, ytick_genes=None, submodule_cutoff=10, threshold=None):
    """
    Uses reference scRNA-seq data to find submodules of genes within a set of genes.

    Parameters
    ----------
    adata : AnnData object
        Anndata object with `obs_celltype` as the name of an `obs` column.
    gene_list : list[str]
        List of genes to create submodules from.
    obs_celltype : str
        Name of the `obs` column in `adata` which indicates the cell type of the observation.
    ytick_genes : list[str] (default None)
        Name of specific genes to label on the y axis. `None` will default to evenly spaced gene names.
    submodule_cutoff : int (default 10)
        The minimum number of genes a submodule must have in order to color and label it.
    threshold : float (default None)
            Value to pass to scipy.cluster.hierarchy.fcluster with a criterion of 'distance'.
            `None` defaults to `0.54*max(Z[:,2])` (same value used by https://www.science.org/doi/10.1126/science.aav9776)

    Returns
    -------
    (g, df_label)
        seaborn `ClusterGrid` and a `DataFrame` with the submodules assigned to each gene.
    """
    
    gene_intersect = np.intersect1d(adata.var.index, gene_list)

    if len(gene_intersect) != len(gene_list):
        print(f"{len(gene_list) - len(gene_intersect)} genes from the gene_set not present in the reference adata")

    df_means = grouped_obs_mean(adata[:, gene_intersect], obs_celltype)
    
    # Normalize gene expression between 0 and 1 across cell types
    df_means = df_means / df_means.max(axis=1).values[:, None] 

    # Determine cluster membership by distance cutoff
    Z = linkage(df_means.values, method='average', metric='cosine')
    if threshold is None:
        max_d = 0.54 * max(Z[:,2])
    else:
        max_d = threshold
    clusters = fcluster(Z, max_d, criterion='distance')

    # Adjust figure/font size to account for increased number of cell types assigned by Azimuth
    if obs_celltype == 'predicted_ids':
        figsize = (22,10)
        fontsize = 8
        bottom = 0.2
    else:
        figsize = (10,10)
        fontsize = 12
        bottom = 0.25

    # Map all clusters with >= submodule_cutoff genes to a Tab20 color; all others to white.
    k = len(np.unique(clusters))
    cmap = plt.get_cmap('tab20')
    cNorm = colors.Normalize(vmin=0, vmax=20)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    k2col = [scalarMap.to_rgba(np.remainder(i, 20)) if np.sum(clusters==i+1) >= submodule_cutoff else (1,1,1,1) for i in range(k)]
    color_labels = [str(m) if np.sum(clusters == m) >= submodule_cutoff else '' for m in range(1, k+1)]

    # Render mean expression as a heatmap with rows grouped (and colored) by subcluster
    vmax = df_means.values.max()
    g = sns.clustermap(df_means,
                   row_cluster=True, row_linkage=Z, row_colors=[k2col[i-1] for i in clusters],
                   col_cluster=False,
                   vmin=0, vmax=vmax,
                   cmap='Greys',
                   figsize=figsize,
                   dendrogram_ratio=0.1,
                   cbar_kws={'label':'',  'orientation':'horizontal'})
    
    #set module labels
    labelled_clusters = [m for m in np.unique(clusters) if np.sum(clusters == m) >= submodule_cutoff]
    sorted_clusters = np.sort(clusters)
    tick_positions = [0.5 * (np.where(sorted_clusters == m)[0][0] + np.where(sorted_clusters == m)[0][-1]) for m in labelled_clusters]
    g.ax_row_colors.set_yticks(tick_positions)
    g.ax_row_colors.set_yticklabels(labelled_clusters)
    g.ax_row_colors.yaxis.set_tick_params(size=0, pad=-30) # make tick marks invisible

    #label the specific genes as yticks
    if ytick_genes is not None:
        reordered_labels = df_means.index[g.dendrogram_row.reordered_ind].tolist()
        use_labels = ytick_genes
        use_ticks = [reordered_labels.index(label) + .5 for label in use_labels]
        g.ax_heatmap.set(yticks=use_ticks, yticklabels=use_labels)

    #threshold dotted line
    g.row_color_labels = color_labels
    g.ax_row_dendrogram.axvline(max_d, color='purple', linestyle='--')

    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = fontsize)

    #reposition
    g.figure.subplots_adjust(bottom=bottom)
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([x0, 1, g.ax_row_dendrogram.get_position().width, 0.02])
    g.ax_cbar.set_title('Scaled mean expression')
    g.ax_cbar.tick_params(axis='x', length=10)

    df_label = pd.DataFrame({'submodule': clusters}, index=df_means.index)

    return g, df_label