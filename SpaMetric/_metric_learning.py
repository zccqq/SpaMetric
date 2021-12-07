# -*- coding: utf-8 -*-

from typing import Optional, Tuple
from anndata import AnnData

import torch
import numpy as np
from scipy.sparse import issparse, csr_matrix

from ._solve import solve_Z


@torch.no_grad()
def metric_learning_func(
    X: np.ndarray,
    beta: float,
    tol_err: float,
    n_iters: int,
    random_state: int,
    device: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    rng = torch.Generator()
    rng.manual_seed(random_state)
    
    X = torch.tensor(X).type(dtype=torch.float32).to(device)
    m, n = X.shape
    W = torch.rand(m, m, generator=rng).to(device)
    Z = torch.rand(n, n, generator=rng).to(device)
    
    W, Z = solve_Z(
        X=X,
        W=W,
        Z=Z,
        beta=beta,
        tol_err=tol_err,
        n_iters=n_iters,
        SS_matrix=None,
        device=device,
        tqdm_params={},
    )
    
    return W.cpu().numpy(), Z.cpu().numpy()


def metric_learning(
    adata: AnnData,
    beta: float = 1e-2,
    tol_err: float = 1e-5,
    n_iters: int = 1000,
    use_highly_variable: Optional[bool] = None,
    random_state: int = 0,
    device: Optional[str] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    '''
    Metric learning for spatial transcriptomics.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    beta
        Parameter to balance the main equation and the constraints.
    tol_err
        Relative error tolerance (convergence criteria).
    n_iters
        Number of iterations for the optimization.
    use_highly_variable
        Whether to use highly variable genes only, stored in `adata.var['highly_variable']`.
        By default uses them if they have been determined beforehand.
    random_state
        Change to use different initial states for the optimization.
    device
        The desired device for `PyTorch` computation. By default uses cuda if cuda is avaliable
        cpu otherwise.
    key_added
        If not specified, the metric learning data is stored in `adata.uns['metric']` and
        the metric matrix is stored in `adata.obsp['metric']`.
        If specified, the metric learning data is added to `adata.uns[key_added]` and
        the metric matrix is stored in `adata.obsp[key_added+'_metric']`.
    copy
        Return a copy instead of writing to ``adata``.
    
    Returns
    -------
    Depending on ``copy``, returns or updates ``adata`` with the following fields.
    
    See ``key_added`` parameter description for the storage path of the metric matrix.
    
    metric : :class:`~scipy.sparse.csr_matrix` (.obsp)
        The sample-by-sample metric matrix.
    '''
    
    adata = adata.copy() if copy else adata
    
    if use_highly_variable is True and 'highly_variable' not in adata.var.keys():
        raise ValueError(
            'Did not find adata.var[\'highly_variable\']. '
            'Either your data already only consists of highly-variable genes '
            'or consider running `pp.highly_variable_genes` first.'
        )
    
    if use_highly_variable is None:
        use_highly_variable = True if 'highly_variable' in adata.var.keys() else False
    
    adata_use = (
        adata[:, adata.var['highly_variable']] if use_highly_variable else adata
    )
    
    
    _, Z = metric_learning_func(
        X=adata_use.X.toarray().T if issparse(adata_use.X) else adata_use.X.T,
        beta=beta,
        tol_err=tol_err,
        n_iters=n_iters,
        random_state=random_state,
        device=device,
    )
    
    
    if key_added is None:
        key_added = 'metric'
        conns_key = 'metric'
        dists_key = 'metric'
    else:
        conns_key = key_added + '_metric'
        dists_key = key_added + '_metric'
    
    adata.uns[key_added] = {}
    
    neighbors_dict = adata.uns[key_added]
    
    neighbors_dict['connectivities_key'] = conns_key
    neighbors_dict['distances_key'] = dists_key
    
    neighbors_dict['params'] = {}
    neighbors_dict['params']['n_neighbors'] = np.count_nonzero(Z) // Z.shape[0]
    neighbors_dict['params']['beta'] = beta
    neighbors_dict['params']['tol_err'] = tol_err
    neighbors_dict['params']['n_iters'] = n_iters
    neighbors_dict['params']['use_highly_variable'] = use_highly_variable
    neighbors_dict['params']['random_state'] = random_state
    
    adata.obsp[conns_key] = csr_matrix(Z)
    
    return adata if copy else None



















