# -*- coding: utf-8 -*-

from typing import Optional
from anndata import AnnData
from ._compat import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances

from scanpy.tools._utils import _choose_representation

_Method = Literal['MiniBatchKMeans', 'KMeans', 'Random']


def reference_centers(
    adata: AnnData,
    n_centers: int,
    method: _Method = 'MiniBatchKMeans',
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    random_state: int = 0,
    copy: bool = False,
) -> Optional[AnnData]:
    '''
    Select reference centers for mini-batch metric learning.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_centers
        Number of reference centers to be selected.
    method
        Method to use for reference center selection.
        
        * ``'MiniBatchKMeans'``
            Use `scikit-learn` :class:`~sklearn.cluster.MiniBatchKMeans`
            to select reference centers.
        * ``'KMeans'``
            Use `scikit-learn` :class:`~sklearn.cluster.KMeans`
            to select reference centers.
        * ``'Random'``
            Randomly choose reference centers.
        
    n_pcs
        Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
    use_rep
        Use the indicated representation. `'X'` or any key for `.obsm` is valid.
        If `None`, the representation is chosen automatically:
        For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
        If 'X_pca' is not present, it’s computed with default parameters.
    random_state
        Change to use different initial states for the optimization.
    copy
        Return a copy instead of writing to ``adata``.
    
    Returns
    -------
    Depending on ``copy``, returns or updates ``adata`` with the following fields.
    
    .obs['reference_centers']
        Boolean indicator of reference centers.
    '''
    
    if method not in ['MiniBatchKMeans', 'KMeans', 'Random']:
        raise ValueError('method needs to be \'MiniBatchKMeans\', \'KMeans\' or \'Random\'')
    
    if n_centers > adata.shape[0]:
        raise ValueError(f'Expected n_centers <= n_obs, but n_centers = {n_centers}, n_obs = {adata.shape[0]}')
    
    adata = adata.copy() if copy else adata
    
    
    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    
    reference_centers = pd.Series(False, index=adata.obs_names)
    
    if method == 'MiniBatchKMeans':
        kmeans = MiniBatchKMeans(n_clusters=n_centers, random_state=random_state).fit(X)
        reference_centers.iloc[np.unique(np.argmin(euclidean_distances(X, kmeans.cluster_centers_), axis=0))] = True
    elif method == 'KMeans':
        kmeans = KMeans(n_clusters=n_centers, random_state=random_state).fit(X)
        reference_centers.iloc[np.unique(np.argmin(euclidean_distances(X, kmeans.cluster_centers_), axis=0))] = True
    elif method == 'Random':
        rng = np.random.RandomState(seed=random_state)
        reference_centers.iloc[rng.choice(np.arange(adata.shape[0]), size=n_centers, replace=False)] = True
    
    
    adata.uns['reference_centers'] = {}
    
    ref_dict = adata.uns['reference_centers']
    
    ref_dict['params'] = {}
    ref_dict['params']['n_centers'] = np.count_nonzero(reference_centers)
    ref_dict['params']['method'] = method
    ref_dict['params']['n_pcs'] = n_pcs
    ref_dict['params']['use_rep'] = use_rep
    ref_dict['params']['random_state'] = random_state
    
    adata.obs['reference_centers'] = reference_centers
    
    return adata if copy else None



















