# rpca.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import re

class RPCA:
    def __init__(self,
                 lam: Optional[float] = None,
                 mu: float = 1.0,
                 max_iter: int = 500,
                 tol: float = 1e-6,
                 verbose: bool = False) -> None:
        self.lam = lam
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # fitted
        self.L_: Optional[np.ndarray] = None
        self.S_: Optional[np.ndarray] = None
        self.rel_err_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    # ---------- public API ----------
    def fit(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = np.asarray(M, dtype=np.float64, order="F")  # Fortran-order helps SVD a bit
        m, n = M.shape
        lam = self.lam if self.lam is not None else 1.0 / np.sqrt(max(m, n))
        mu = float(self.mu)
        assert mu > 0.0

        L = np.zeros_like(M)
        S = np.zeros_like(M)
        Y = np.zeros_like(M)

        Mnrm = np.linalg.norm(M, "fro") + 1e-12

        for k in range(self.max_iter):
            L = self._svt(M - S + (1.0 / mu) * Y, tau=1.0 / mu) # L-update: singular value thresholding
            T = M - L + (1.0 / mu) * Y
            S = self._soft_threshold(T, tau=lam / mu)  # S-update: elementwise soft-thresholding
            Y = Y + mu * (M - L - S) # dual update

            rel_err = np.linalg.norm(M - L - S, "fro") / Mnrm
            if self.verbose and (k % 50 == 0 or rel_err < self.tol):
                print(f"[RPCA] iter={k:4d}, rel_err={rel_err:.3e}")
            if rel_err < self.tol:
                self.n_iter_ = k + 1
                self.rel_err_ = float(rel_err)
                break
        else:
            self.n_iter_ = self.max_iter
            self.rel_err_ = float(rel_err)

        self.L_, self.S_ = L, S
        return L, S

    def estimate_noise(self, axis: int = 0) -> np.ndarray:
        """
        Return σ_hat for each column (default) as ||S_:i||_2.
        Call after fit().
        """
        if self.S_ is None:
            raise RuntimeError("Call fit(M) before estimate_noise().")
        return np.linalg.norm(self.S_, axis=axis)

    def weights_from_noise(self, sigma_hat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Convert noise estimates into inverse-proportional normalized weights.
        Larger noise -> smaller weight.
        """
        sigma_hat = np.asarray(sigma_hat, dtype=np.float64)
        if np.allclose(sigma_hat, 0):
            return np.ones_like(sigma_hat) / sigma_hat.size
        inv = 1.0 / (sigma_hat + eps)
        return inv / inv.sum()

    # ---------- static helpers ----------
    @staticmethod
    def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

    @staticmethod
    def _svt(X: np.ndarray, tau: float) -> np.ndarray:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_thr = np.maximum(s - tau, 0.0)
        return (U * s_thr) @ Vt
    
def _nat_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch  
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().contiguous().numpy()
    except Exception:
        pass
    return np.asarray(x)

def build_rpca_M_from_B(client_models: List[Dict[str, Any]],
                        dtype: np.dtype = np.float32) -> np.ndarray:
    if not client_models:
        raise ValueError("client_models is empty.")

    key_sets = []
    for p in client_models:
        key_sets.append({k for k in p.keys() if "lora_B" in k})
    common_keys = set.intersection(*key_sets)
    if not common_keys:
        raise ValueError("No 'lora_B' keys found in client_models.")

    b_key_order = sorted(common_keys, key=_nat_key)

    cols = []
    for idx, params in enumerate(client_models):
        missing = [k for k in b_key_order if k not in params]
        if missing:
            raise KeyError(f"Client {idx} missing keys: {missing}")
        flat_parts = []
        for k in b_key_order:
            B = _to_numpy(params[k])        
            flat_parts.append(B.reshape(-1))
        cols.append(np.concatenate(flat_parts, axis=0))

    M = np.stack(cols, axis=1)
    if dtype is not None:
        M = M.astype(dtype, copy=False)
    return M


def noise_by_residual_pca_B(client_models, b_keys, rank_k=1):
    cols = []
    for mp in client_models:
        flat = np.concatenate([_to_numpy(mp[k]).reshape(-1).astype(np.float32) for k in b_keys], axis=0)
        cols.append(flat)
    X = np.stack(cols, axis=1)                       # d × C

    X = X - X.mean(axis=1, keepdims=True)

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k_eff = max(0, min(rank_k, U.shape[1]-1))             
    if k_eff > 0:
        Uk = U[:, :k_eff]
        R  = X - Uk @ (Uk.T @ X)
        denom = max(X.shape[0] - k_eff, 1)
    else:
        R = X
        denom = max(X.shape[0], 1)

    sigma_hat = np.sqrt((R**2).sum(axis=0) / denom)  # shape=(C,)
    return sigma_hat

def noise_by_residual_pca_B_loo(client_models, b_keys, rank_k=1):
    # 构造 X(d×C)
    cols = []
    for mp in client_models:
        flat = np.concatenate([_to_numpy(mp[k]).reshape(-1).astype(np.float32) for k in b_keys], 0)
        cols.append(flat)
    X = np.stack(cols, 1)                      # d × C
    d, C = X.shape
    sig = np.zeros(C, dtype=np.float64)

    for i in range(C):
        Xm = np.delete(X, i, axis=1)           # d × (C-1)
        Xm = Xm - Xm.mean(axis=1, keepdims=True)   
        U, s, Vt = np.linalg.svd(Xm, full_matrices=False)
        k_eff = max(0, min(rank_k, U.shape[1]-1))
        if k_eff > 0:
            Uk = U[:, :k_eff]
            mu = Xm.mean(axis=1, keepdims=True)
            xi = (X[:, [i]] - mu)
            ri = xi - Uk @ (Uk.T @ xi)
            denom = max(d - k_eff, 1)
        else:
            mu = Xm.mean(axis=1, keepdims=True)
            ri = (X[:, [i]] - mu)
            denom = d
        sig[i] = float(np.linalg.norm(ri)**2 / denom)
    return np.sqrt(sig)  # σ̂_i

def noise_by_residual_pca_B_loo_hetero(client_models, b_keys, rank_k=1):
    max_r_per_key = {}
    out_dim_per_key = {}

    for k in b_keys:
        max_r = 0
        out_dim = None
        for mp in client_models:
            B = _to_numpy(mp[k]).astype(np.float32)
            if B.ndim != 2:
                raise ValueError(f"{k} is not 2D, got shape={B.shape}")
            out_dim = B.shape[0]
            max_r = max(max_r, B.shape[1])
        max_r_per_key[k] = max_r
        out_dim_per_key[k] = out_dim

    cols = []
    for mp in client_models:
        flat_blocks = []
        for k in b_keys:
            B = _to_numpy(mp[k]).astype(np.float32)    # [d_out, r_i]
            d_out, r_i = B.shape
            r_max = max_r_per_key[k]

            if r_i < r_max:
                pad = np.zeros((d_out, r_max - r_i), dtype=np.float32)
                B_pad = np.concatenate([B, pad], axis=1)
            else:
                B_pad = B

            flat_blocks.append(B_pad.reshape(-1))

        flat = np.concatenate(flat_blocks, axis=0)
        cols.append(flat)

    X = np.stack(cols, axis=1)   # [d, C]
    d, C = X.shape
    sig = np.zeros(C, dtype=np.float64)

    for i in range(C):
        Xm_raw = np.delete(X, i, axis=1)                    # [d, C-1]
        mu = Xm_raw.mean(axis=1, keepdims=True)             
        Xm = Xm_raw - mu

        U, s, Vt = np.linalg.svd(Xm, full_matrices=False)
        k_eff = min(rank_k, U.shape[1])
        xi = X[:, [i]]

        if k_eff > 0:
            Uk = U[:, :k_eff]
            ri = xi - Uk @ (Uk.T @ xi)                      # ≈ (I - P_i) x_i
            denom = max(d - k_eff, 1)
        else:
            ri = xi
            denom = d

        sig[i] = float(np.linalg.norm(ri) ** 2 / denom)

    return np.sqrt(sig)


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)




def RPCA_weights(CLIENT_SIGMA, client_models, num_successful_clients):
    M_B = build_rpca_M_from_B(client_models)  # M.shape = (d_B, num_successful_clients)
    rpca_B = RPCA(lam=None, mu=1.0, max_iter=500, tol=1e-6, verbose=False)
    L_rpca_B, S_rpca_B = rpca_B.fit(M_B) 
    sigma_hat_B = rpca_B.estimate_noise(axis=0)                 # ||S_:i||_2
    weights_B = rpca_B.weights_from_noise(sigma_hat_B).tolist()   # 1/(sigma_hat+eps)
    weights_B = [round(float(w), 3) for w in weights_B]
            
    print("\n=== Noise per Client (RPCA estimate vs. true if available) ===")
    sigma_true_std = [float(CLIENT_SIGMA[cid][0]) for cid in range(num_successful_clients)]
    for i in range(num_successful_clients):
        st = "N/A" if (sigma_true_std[i] is None) else f"{sigma_true_std[i]:.3f}"
        print(f"Client {i+1}: sigma_hat={sigma_hat_B[i]:.3f}, sigma_true={st}, weight={weights_B[i]:.4f}")

    print(f"Client Weights based on estimated noise by B: [{', '.join(f'{w:.3f}' for w in weights_B)}]")
            
    weights = weights_B
    print(f"Client Weights based on estimated noise: [{', '.join(f'{w:.3f}' for w in weights)}]")
    return sigma_true_std

def noise_by_residual_pca_B_projector_loo(
    client_models,
    b_keys,
    q_common=None,
    eig_threshold=None,
    eps=1e-12,
):

    C = len(client_models)
    if C < 2:
        raise ValueError("Need at least 2 clients for LOO estimation.")

    numer = np.zeros(C, dtype=np.float64)
    denom = np.zeros(C, dtype=np.float64)

    for k in b_keys:
        B_list = []
        d_out = None
        ranks = []

        for mp in client_models:
            B = _to_numpy(mp[k]).astype(np.float32)
            if B.ndim != 2:
                raise ValueError(f"{k} is not 2D, got shape={B.shape}")
            B_list.append(B)
            d_out = B.shape[0]
            ranks.append(B.shape[1])

        P_list = []
        for B in B_list:
            U, s, Vt = np.linalg.svd(B, full_matrices=False)
            r_i = U.shape[1]
            Ui = U[:, :r_i]
            Pi = Ui @ Ui.T                      # [d_out, d_out]
            P_list.append(Pi.astype(np.float32))

        for i in range(C):
            P_bar = np.zeros((d_out, d_out), dtype=np.float32)
            for j in range(C):
                if j == i:
                    continue
                P_bar += P_list[j]
            P_bar /= max(C - 1, 1)

            evals, evecs = np.linalg.eigh(P_bar)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            if q_common is not None:
                qk = min(q_common, d_out)
            elif eig_threshold is not None:
                qk = int(np.sum(evals > eig_threshold))
                qk = max(qk, 1)
            else:
                other_ranks = [ranks[j] for j in range(C) if j != i]
                qk = int(np.median(other_ranks))
                qk = max(1, min(qk, d_out))

            Uc = evecs[:, :qk]                 # [d_out, qk]
            P_common = Uc @ Uc.T               # [d_out, d_out]

            B_i = B_list[i]                    # [d_out, r_i]
            r_i = B_i.shape[1]

            R_i = B_i - P_common @ B_i         # (I - P_common) B_i
            res_energy = float(np.linalg.norm(R_i, ord='fro') ** 2)

            numer[i] += res_energy
            denom[i] += max((d_out - qk) * r_i, eps)

    sigma_hat = np.sqrt(numer / np.maximum(denom, eps))
    return sigma_hat

def noise_by_residual_pca_B_shared_projector(
    client_models,
    b_keys,
    q_subspace=4,
    q_common=4,
    layer_stride=1,
    eps=1e-12,
):
    N = len(client_models)
    if N == 0:
        raise ValueError("client_models is empty")

    numer = np.zeros(N, dtype=np.float64)
    denom = np.zeros(N, dtype=np.float64)

    sel_keys = b_keys[::layer_stride]

    for k in sel_keys:
        B_list = []
        U_blocks = []
        d_out = None

        for mp in client_models:
            B = _to_numpy(mp[k]).astype(np.float32)   # [d_out, r_i]
            if B.ndim != 2:
                raise ValueError(f"{k} is not 2D, got shape={B.shape}")

            d_out, r_i = B.shape
            B_list.append(B)

            U, s, Vt = np.linalg.svd(B, full_matrices=False)
            q_i = min(q_subspace, U.shape[1])
            Uq = U[:, :q_i]                           # [d_out, q_i]
            U_blocks.append(Uq)

        A = np.concatenate(U_blocks, axis=1)         # [d_out, sum_i q_i]
        Uc, sc, Vct = np.linalg.svd(A, full_matrices=False)
        qk = min(q_common, Uc.shape[1])
        U_common = Uc[:, :qk]                        # [d_out, qk]

        for i, B_i in enumerate(B_list):
            r_i = B_i.shape[1]
            R_i = B_i - U_common @ (U_common.T @ B_i)
            numer[i] += float(np.linalg.norm(R_i, ord="fro") ** 2)
            denom[i] += max((d_out - qk) * r_i, eps)

    sigma_hat = np.sqrt(numer / np.maximum(denom, eps))
    return sigma_hat


def noise_by_residual_pca_B_shared_projector_layer(
    client_models,
    b_keys,
    q_subspace=4,
    q_common=4,
    layer_stride=1,
    eps=1e-12,
    selected_key=None,     
    verbose=True,          
    return_layerwise=True, 
):
    N = len(client_models)
    if N == 0:
        raise ValueError("client_models is empty")

    if selected_key is not None:
        if selected_key not in b_keys:
            raise ValueError(f"selected_key={selected_key} not in b_keys")
        sel_keys = [selected_key]
    else:
        sel_keys = b_keys[::layer_stride]

    numer = np.zeros(N, dtype=np.float64)
    denom = np.zeros(N, dtype=np.float64)

    layer_sigma_dict = {}

    for k in sel_keys:
        B_list = []
        U_blocks = []
        d_out = None

        # -------- 1) For each client, extract the B matrix and local left subspace for this layer --------
        for mp in client_models:
            B = _to_numpy(mp[k]).astype(np.float32)   # [d_out, r_i]
            if B.ndim != 2:
                raise ValueError(f"{k} is not 2D, got shape={B.shape}")

            d_out, r_i = B.shape
            B_list.append(B)

            U, s, Vt = np.linalg.svd(B, full_matrices=False)
            q_i = min(q_subspace, U.shape[1])
            Uq = U[:, :q_i]                           # [d_out, q_i]
            U_blocks.append(Uq)

        # -------- 2) Construct the shared common subspace for this layer --------
        A = np.concatenate(U_blocks, axis=1)         # [d_out, sum_i q_i]
        Uc, sc, Vct = np.linalg.svd(A, full_matrices=False)
        qk = min(q_common, Uc.shape[1])
        U_common = Uc[:, :qk]                        # [d_out, qk]

        # -------- 3) Calculate the residual sigma for each client in this layer --------
        numer_layer = np.zeros(N, dtype=np.float64)
        denom_layer = np.zeros(N, dtype=np.float64)

        for i, B_i in enumerate(B_list):
            r_i = B_i.shape[1]
            R_i = B_i - U_common @ (U_common.T @ B_i)
            numer_layer[i] = float(np.linalg.norm(R_i, ord="fro") ** 2)
            denom_layer[i] = max((d_out - qk) * r_i, eps)

        sigma_layer = np.sqrt(numer_layer / np.maximum(denom_layer, eps))
        layer_sigma_dict[k] = sigma_layer

        if verbose:
            print(f"[Layer] {k}")
            print(f"  sigma_layer = {np.array2string(sigma_layer, precision=6, separator=', ')}")
            print(f"  mean={sigma_layer.mean():.6f}, std={sigma_layer.std():.6f}")

        # -------- 4) Accumulate to the global result --------
        numer += numer_layer
        denom += denom_layer

    sigma_hat = np.sqrt(numer / np.maximum(denom, eps))

    if verbose:
        if selected_key is not None:
            print(f"\n[Final] only use one layer: {selected_key}")
        else:
            print(f"\n[Final] aggregated over {len(sel_keys)} layers")
        print(f"  sigma_hat = {np.array2string(sigma_hat, precision=6, separator=', ')}")

    if return_layerwise:
        return sigma_hat, layer_sigma_dict
    else:
        return sigma_hat