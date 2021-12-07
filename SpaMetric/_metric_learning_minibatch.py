# -*- coding: utf-8 -*-

from typing import Optional, Tuple
from anndata import AnnData

import sys
import torch
import numpy as np
import time
from scipy.sparse import issparse

from ._solve import solve_Z
from ._utils import format_interval


@torch.no_grad()
def metric_learning_minibatch_func(
    X: np.ndarray,
    center_idx: np.ndarray,
    beta: float,
    tol_err: float,
    n_iters: int,
    n_epochs: int,
    random_state: int,
    device: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    rng = torch.Generator()
    rng.manual_seed(random_state)
    rng_numpy = np.random.RandomState(seed=random_state)
    
    X = torch.tensor(X).type(dtype=torch.float32).to(device)
    
    m, n = X.shape
    k = center_idx.shape[0]
    
    batch_size = k
    batch_size_extra = (n - k) % batch_size > 0
    num_batches = (n - k) // batch_size + batch_size_extra
    
    W = torch.rand(m, m, generator=rng).to(device)
    
    V = torch.rand(k, n, generator=rng).to(device)
    
    X_ref = X[:, center_idx]
    
    data_idx = np.setdiff1d(np.arange(n), center_idx)
    
    time_total = 0
    
    for epoch in range(n_epochs):
        
        time_start = time.time()
        
        print(f'epoch {epoch+1}/{n_epochs}:', flush=True)
        
        n_perm = rng_numpy.permutation(data_idx)
        
        for batch_i in range(num_batches+1):
            
            if batch_i == 0:
                
                W, V[:, center_idx] = solve_Z(
                    X_ref,
                    W,
                    V[:, center_idx],
                    beta=beta,
                    tol_err=tol_err,
                    n_iters=n_iters,
                    SS_matrix=None,
                    device=device,
                    tqdm_params={
                        'leave': False,
                        'desc': f'batch {batch_i+1}/{num_batches+1}',
                    },
                )
                
                continue
            
            if batch_i < num_batches+1:
                batch_idx = n_perm[((batch_i-1) * batch_size):(batch_i * batch_size)]
            elif batch_size_extra:
                batch_idx = n_perm[-batch_size:]
            else:
                break
            
            n_batch = batch_idx.shape[0]
            
            SS_matrix_batch = np.ones((k+n_batch, k+n_batch))
            SS_matrix_batch[np.ix_(np.arange(k), np.arange(k))] = 0
            SS_matrix_batch[np.ix_(np.arange(k, k+n_batch), np.arange(k, k+n_batch))] = 0
            
            X_batch = torch.cat((X_ref, X[:, batch_idx]), axis=1)
            
            Z_batch = torch.zeros((k+n_batch, k+n_batch)).to(device)
            Z_batch[np.ix_(np.arange(k), np.arange(k, k+n_batch))] = V[:, batch_idx]
            Z_batch[np.ix_(np.arange(k, k+n_batch), np.arange(k))] = V[:, batch_idx].T
            
            W, Z_batch = solve_Z(
                X_batch,
                W,
                Z_batch,
                beta=beta,
                tol_err=tol_err,
                n_iters=n_iters,
                SS_matrix=SS_matrix_batch,
                device=device,
                tqdm_params={
                    'leave': batch_i==num_batches,
                    'desc': f'batch {batch_i+1}/{num_batches+1}',
                },
            )
            
            V[:, batch_idx] = Z_batch[np.ix_(np.arange(k), np.arange(k, k+n_batch))]
            
        sys.stderr.flush()
        
        time_epoch = time.time() - time_start
        time_total += time_epoch
        
        print(f'epoch time {format_interval(time_epoch)},',
              f'total time {format_interval(time_total)}')
        
    return W.cpu().numpy(), V.cpu().numpy()


def metric_learning_minibatch(
    adata: AnnData,
    beta: float = 1e-2,
    tol_err: float = 1e-5,
    n_iters: int = 1000,
    n_epochs: int = 2,
    use_highly_variable: Optional[bool] = None,
    random_state: int = 0,
    device: Optional[str] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    '''
    Mini-batch metric learning for large-scale spatial transcriptomics.
    
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
    n_epochs
        How many times to traverse all samples.
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
        the metric matrix is stored in `adata.obsm['metric']`.
        If specified, the metric learning data is added to `adata.uns[key_added]` and
        the metric matrix is stored in `adata.obsm[key_added+'_metric']`.
    copy
        Return a copy instead of writing to ``adata``.
    
    Returns
    -------
    Depending on ``copy``, returns or updates ``adata`` with the following fields.
    
    See ``key_added`` parameter description for the storage path of the metric matrix.
    
    metric : :class:`~numpy.ndarray` (.obsm)
        The sample-by-reference metric matrix.
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
    
    if 'reference_centers' not in adata.obs.columns:
        raise ValueError(
            'Could not find adata.obs[\'reference_centers\'].'
            'Please run reference_centers first.'
        )
    center_idx = np.flatnonzero(adata.obs['reference_centers'])
    
    
    _, V = metric_learning_minibatch_func(
        X=adata_use.X.toarray().T if issparse(adata_use.X) else adata_use.X.T,
        center_idx=center_idx,
        beta=beta,
        tol_err=tol_err,
        n_iters=n_iters,
        n_epochs=n_epochs,
        random_state=random_state,
        device=device,
    )
    
    
    if key_added is None:
        key_added = 'metric'
        conns_key = 'metric'
    else:
        conns_key = key_added + '_metric'
    
    adata.uns[key_added] = {}
    
    neighbors_dict = adata.uns[key_added]
    
    neighbors_dict['params'] = {}
    neighbors_dict['params']['beta'] = beta
    neighbors_dict['params']['tol_err'] = tol_err
    neighbors_dict['params']['n_iters'] = n_iters
    neighbors_dict['params']['n_epochs'] = n_epochs
    neighbors_dict['params']['random_state'] = random_state
    neighbors_dict['params']['use_highly_variable'] = use_highly_variable
    
    adata.obsm[conns_key] = V.T
    
    return adata if copy else None



















