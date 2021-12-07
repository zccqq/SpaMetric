# -*- coding: utf-8 -*-

from typing import Optional
from anndata import AnnData

import numpy as np
import scanpy as sc

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix


def spatial_reconstruction(
    adata: AnnData,
    alpha: float = 1,
    n_neighbors: int = 10,
    n_pcs: int = 15,
    use_highly_variable: Optional[bool] = None,
    normalize_total: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    '''
    Spatial reconstruction of gene expression matrix.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    alpha
        Relative weight for the aggregated expression to the original expression.
    n_neighbors
        Number of spatial neighbors for aggregating expression.
    n_pcs
        Number of principal components to use for weighting the spatial neighbors.
    use_highly_variable
        Whether to use highly variable genes only, stored in `adata.var['highly_variable']`.
        By default uses them if they have been determined beforehand.
    normalize_total
        Whether to normalize each cell by total counts over all genes.
    copy
        Return a copy instead of writing to ``adata``.
    
    Returns
    -------
    Depending on ``copy``, returns or updates ``adata`` with the following fields.
    
    .X
        The reconstructed log transformed gene expression matrix.
    .layers['counts']
        The gene expression count matrix.
    .layers['log1p']
        The log transformed gene expression matrix.
    '''
    
    adata = adata.copy() if copy else adata
    
    adata.layers['counts'] = adata.X
    
    sc.pp.normalize_total(adata) if normalize_total else None
    sc.pp.log1p(adata)
    
    adata.layers['log1p'] = adata.X
    
    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)
    
    coord = adata.obsm['spatial']
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coord)
    nbrs = neigh.kneighbors_graph(coord)
    
    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
    conns = nbrs.T.toarray() * dists
    
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X
    
    adata.X = csr_matrix(X_rec)
    
    del adata.obsm['X_pca']
    
    
    adata.uns['spatial_reconstruction'] = {}
    
    rec_dict = adata.uns['spatial_reconstruction']
    
    rec_dict['params'] = {}
    rec_dict['params']['alpha'] = alpha
    rec_dict['params']['n_neighbors'] = n_neighbors
    rec_dict['params']['n_pcs'] = n_pcs
    rec_dict['params']['use_highly_variable'] = use_highly_variable
    rec_dict['params']['normalize_total'] = normalize_total
    
    return adata if copy else None



















