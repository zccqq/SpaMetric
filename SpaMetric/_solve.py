# -*- coding: utf-8 -*-

from typing import Tuple, Optional

import torch
from torch import Tensor
import numpy as np

from tqdm import trange


@torch.no_grad()
def optimize_W(
    X: Tensor,
    Z: Tensor,
    Y_w: Tensor,
    z_eye: Tensor,
    ones_w: Tensor,
    lamba_w: Tensor,
    beta: float,
    rho: float,
    ceta: float,
) -> Tuple[Tensor, Tensor]:
    
    optimize_zz = torch.matmul(Z, Z.T) + z_eye
    
    L_w_matrix = torch.matmul(torch.matmul(X, 2*optimize_zz - 2*Z), X.T)
        
    L_w = 2*torch.norm(L_w_matrix, 2)+ beta * lamba_w.shape[1] / ceta
    
    M_w = torch.matmul(2*L_w_matrix + beta/ceta, Y_w) - 2*torch.matmul(torch.matmul(X, optimize_zz), X.T) - beta/ceta + lamba_w
    
    W = (Y_w - M_w/L_w) * ones_w
    W = (torch.abs(W) + W) / 2
    W = (W + W.T) / 2
    
    leq3 = torch.sum(W, dim=0) - 1
    
    lamba_w = lamba_w + beta*rho*leq3
    
    return W, lamba_w


@torch.no_grad()
def optimize_Z(
    X: Tensor,
    W: Tensor,
    Y_z: Tensor,
    w_eye: Tensor,
    ones_z: Tensor,
    lamba_z: Tensor,
    beta: float,
    rho: float,
    ceta: float,
) -> Tuple[Tensor, Tensor]:
    
    optimize_ww = torch.matmul(W, W.T) + w_eye
    
    L_z_matrix = torch.matmul(torch.matmul(X.T, 2*optimize_ww - 2*W), X)
    
    L_z = 2*torch.norm(L_z_matrix, 2) + beta * lamba_z.shape[1] / ceta
    
    M_z = torch.matmul(2*L_z_matrix + beta/ceta, Y_z) - 2*torch.matmul(torch.matmul(X.T, optimize_ww), X) - beta/ceta + lamba_z
    
    Z = (Y_z - M_z/L_z) * ones_z
    Z = (torch.abs(Z) + Z)/2
    Z = (Z + Z.T)/2
    
    leq4 = torch.sum(Z, dim=0) - 1
    
    lamba_z = lamba_z + beta*rho*leq4
    
    return Z, lamba_z


@torch.no_grad()
def solve_Z(
    X: Tensor,
    W: Tensor,
    Z: Tensor,
    beta: float,
    tol_err: float,
    n_iters: int,
    SS_matrix: Optional[np.ndarray],
    device: torch.device,
    tqdm_params: dict,
) -> Tuple[Tensor, Tensor]:
    
    m, n = X.shape
    
    rho = 0.8
    ceta_prev = 1 / rho
    ceta = 1
    
    func_err = float('inf')
    
    W_prev = W
    Z_prev = Z
    
    lamba_w = torch.zeros(1, m).to(device)
    lamba_z = torch.zeros(1, n).to(device)
    
    z_eye = torch.eye(n).to(device)
    w_eye = torch.eye(m).to(device)
    
    if SS_matrix is None:
        ones_z = 1 - z_eye
    else:
        ones_z = torch.tensor(SS_matrix, dtype=torch.float32).to(device)
    ones_w = 1 - w_eye
    
    pbar = trange(n_iters, position=0, **tqdm_params)
    
    for Iter in pbar:
        
        func_err_prev = func_err
        
        Y_iter_value = (ceta * (1 - ceta_prev)) / ceta_prev
        
        Y_w = W + Y_iter_value * (W - W_prev)
        Y_z = Z + Y_iter_value * (Z - Z_prev)
        
        W_prev = W
        Z_prev = Z
        
        W, lamba_w = optimize_W(
            X=X,
            Z=Z,
            Y_w=Y_w,
            z_eye=z_eye,
            ones_w=ones_w,
            lamba_w=lamba_w,
            beta=beta,
            rho=rho,
            ceta=ceta,
        )
        
        Z, lamba_z = optimize_Z(
            X=X,
            W=W,
            Y_z=Y_z,
            w_eye=w_eye,
            ones_z=ones_z,
            lamba_z=lamba_z,
            beta=beta,
            rho=rho,
            ceta=ceta,
        )
        
        ceta_prev = ceta
        ceta = 1 / (1 - rho + 1 / ceta)
        
        func_1_err = torch.norm(torch.matmul(W.T, torch.matmul(X, z_eye-Z)), 'fro')
        func_2_err = torch.norm(torch.matmul(X, z_eye-Z), 'fro')
        func_3_err = torch.norm(torch.matmul(Z.T, torch.matmul(X.T, w_eye-W)), 'fro')
        func_4_err = torch.norm(torch.matmul(X.T, w_eye-W), 'fro')
        
        func_err = func_1_err + func_2_err + func_3_err + func_4_err
        
        func_err_rel = torch.abs(func_err_prev - func_err) / func_err_prev
        
        pbar.set_postfix_str(f'err={func_err_rel.item():.5e}')
        
        if func_err_rel < tol_err:
            pbar.set_postfix_str(f'err={func_err_rel.item():.5e}, converged!')
            break
        
    return W, Z



















