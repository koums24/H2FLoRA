import math
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from collections import Counter

def printargs(args):
    fields = [
        "add_lora_noise",
        "load_artifacts",
        "weight_mode",
        "tau",
        "noise_seed",
        "gamma_seed",
        "sigma_max",
    ]
    
    def _get(a, k, default=None):
        if isinstance(a, dict):
            return a.get(k, default)
        return getattr(a, k, default)
    key_width = max(len(k) for k in fields)

    print("=== Arguments ===")
    for k in fields:
        v = _get(args, k, default=None)
        print(f"{k:<{key_width}} : {repr(v)}")
        

def dirichlet_partition_indices(labels, num_clients, alpha=0.3, seed=42):
        labels = np.asarray(labels)
        classes = np.unique(labels)
        rng = np.random.default_rng(seed)

        while True:
            client_indices = [[] for _ in range(num_clients)]
            for k in classes:
                k_idx = np.where(labels == k)[0]
                rng.shuffle(k_idx)
                p = rng.dirichlet(np.full(num_clients, alpha))       
                counts = rng.multinomial(len(k_idx), p)             
                s = 0
                for cid, c in enumerate(counts):
                    if c > 0:
                        client_indices[cid].extend(k_idx[s:s+c].tolist())
                        s += c
            if min(len(x) for x in client_indices) >= 1:
                break
        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])
        return client_indices

def dirichlet_partition_from_subset(subset_indices, subset_labels, num_subclients, alpha, seed):
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(subset_labels)
    client_indices = [[] for _ in range(num_subclients)]

    for c in unique_classes:
        cls_local_idx = np.where(subset_labels == c)[0]   
        rng.shuffle(cls_local_idx)

        proportions = rng.dirichlet([alpha] * num_subclients)
        split_points = (np.cumsum(proportions)[:-1] * len(cls_local_idx)).astype(int)
        split_local = np.split(cls_local_idx, split_points)

        for cid in range(num_subclients):
            client_indices[cid].extend(subset_indices[split_local[cid]].tolist())

    for cid in range(num_subclients):
        rng.shuffle(client_indices[cid])

    return client_indices

    
def build_client_data_with_task_shift(
    global_train_texts,
    global_train_labels,
    num_clients,
    client_data_size=500,
    dir_alpha=0.3,
    seed=42,
    drift_ratio=0.1,
    enable_task_shift=True,
    main_classes=None,
    drift_classes=None,
):
    rng = np.random.default_rng(seed)
    labels_np = np.asarray(global_train_labels)
    all_classes = sorted(np.unique(labels_np).tolist())

    if not enable_task_shift:
        client_train_indices = dirichlet_partition_from_subset(
            subset_indices=np.arange(len(labels_np)),
            subset_labels=labels_np,
            num_subclients=num_clients,
            alpha=dir_alpha,
            seed=seed
        )

    else:
        if main_classes is None or drift_classes is None:
            mid = len(all_classes) // 2
            main_classes = all_classes[:mid]
            drift_classes = all_classes[mid:]

        main_classes = sorted(main_classes)
        drift_classes = sorted(drift_classes)

        if len(main_classes) == 0 or len(drift_classes) == 0:
            raise ValueError("main_classes and drift_classes must both be non-empty.")

        num_drift_clients = max(1, int(num_clients * drift_ratio))
        num_main_clients = num_clients - num_drift_clients

        all_client_ids = np.arange(num_clients)
        rng.shuffle(all_client_ids)
        drift_client_ids = all_client_ids[:num_drift_clients]
        main_client_ids  = all_client_ids[num_drift_clients:]

        main_pool_indices = np.where(np.isin(labels_np, main_classes))[0]
        drift_pool_indices = np.where(np.isin(labels_np, drift_classes))[0]

        main_pool_labels = labels_np[main_pool_indices]
        drift_pool_labels = labels_np[drift_pool_indices]

        if len(main_pool_indices) == 0:
            raise ValueError(f"No samples found for main_classes={main_classes}")
        if len(drift_pool_indices) == 0:
            raise ValueError(f"No samples found for drift_classes={drift_classes}")

        main_partitions = dirichlet_partition_from_subset(
            subset_indices=main_pool_indices,
            subset_labels=main_pool_labels,
            num_subclients=num_main_clients,
            alpha=dir_alpha,
            seed=seed
        )

        drift_partitions = dirichlet_partition_from_subset(
            subset_indices=drift_pool_indices,
            subset_labels=drift_pool_labels,
            num_subclients=num_drift_clients,
            alpha=dir_alpha,
            seed=seed + 1
        )

        client_train_indices = [None] * num_clients

        for local_i, global_cid in enumerate(main_client_ids):
            client_train_indices[global_cid] = main_partitions[local_i]

        for local_i, global_cid in enumerate(drift_client_ids):
            client_train_indices[global_cid] = drift_partitions[local_i]

    client_data = []
    for idxs in client_train_indices:
        client_texts = [global_train_texts[i] for i in idxs]
        client_labels = [global_train_labels[i] for i in idxs]
        client_data.append((client_texts, client_labels))

    for i, (client_texts, client_labels) in enumerate(client_data):
        if len(client_texts) == 0:
            raise ValueError(f"Client {i} has no samples after partition.")

        if len(client_texts) > client_data_size:
            sel = rng.choice(len(client_texts), size=client_data_size, replace=False)
        else:
            extra = rng.choice(len(client_texts), size=client_data_size - len(client_texts), replace=True)
            sel = np.concatenate([np.arange(len(client_texts)), extra])

        client_data[i] = (
            [client_texts[j] for j in sel],
            [client_labels[j] for j in sel]
        )

    print("\n===== Client label distribution =====")
    for cid, (client_texts, client_labels) in enumerate(client_data):
        label_count = Counter(client_labels)
        label_count = dict(sorted(label_count.items(), key=lambda x: x[0]))
        print(f"Client {cid}: total={len(client_labels)}, labels={label_count}")
        
    splits = []
    for cid, (client_texts, client_labels) in enumerate(client_data):
        n = len(client_texts)
        idx = np.arange(n)
        labels_arr = np.asarray(client_labels)

        strat = labels_arr if (
            len(np.unique(labels_arr)) > 1 and
            all((labels_arr == c).sum() >= 2 for c in np.unique(labels_arr))
        ) else None

        train_idx, val_idx = train_test_split(
            idx, test_size=0.2, shuffle=True, random_state=cid + 1000, stratify=strat
        )
        splits.append((train_idx, val_idx))

    return client_data, splits

def surpassed_percentage(weights):
    w = np.asarray(weights, dtype=float)
    N = w.size
    if N <= 1:
        return np.zeros_like(w, dtype=float)
    vals, inv, counts = np.unique(w, return_inverse=True, return_counts=True)  
    
    counts_less_group = np.cumsum(counts) - counts  # shape=(#groups,)
    less = counts_less_group[inv]
  
    return less / (N - 1)

def compute_utilities(alpha, beta, acc_prev, sigmas, sigma_max):
  
    a  = np.asarray(alpha,  dtype=float).reshape(-1)
    b  = np.asarray(beta,   dtype=float).reshape(-1)
    ap = np.asarray(acc_prev, dtype=float).reshape(-1)
    sg = np.asarray(sigmas, dtype=float).reshape(-1)
   
    utility = a * ap + b * sg/sigma_max
    norm_utility = (a * ap + b * sg/sigma_max) / (a+b)
    return utility, norm_utility

def to_jsonable_metrics(result: dict) -> dict:
    out = {}
    for k, v in result.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = str(v)
    return out

def frobenius_of_lora_AB(lora_params: dict):
    import math, torch
    A_sq = B_sq = 0.0
    for k, t in lora_params.items():
        v = t.detach().float().cpu()
        s = (v * v).sum().item()
        if "lora_A" in k:
            A_sq += s
        elif "lora_B" in k:
            B_sq += s
    return {
        "fro_A": math.sqrt(A_sq),
        "fro_B": math.sqrt(B_sq),
        "fro_all": math.sqrt(A_sq + B_sq),
    }
    
def add_noise_to_lora(lora_params: dict, sigmaA: float, sigmaB: float, seed: int = None):
    g = torch.Generator(device='cpu')
    if seed is not None:
        g.manual_seed(int(seed))
    out = {}
    for k, v in lora_params.items():
        t = v.detach().cpu()
        if "lora_A" in k:
            eps = torch.normal(0.0, sigmaA, size=t.shape, generator=g, dtype=t.dtype)
            out[k] = t + eps
        elif "lora_B" in k:
            eps = torch.normal(0.0, sigmaB, size=t.shape, generator=g, dtype=t.dtype)
            out[k] = t + eps
        else:
            out[k] = t.clone()
    return out

def measure_noise_ratio_for_client(lora_clean: dict, lora_noisy: dict, lora_alpha: float, r: int):
    import math, re
    def nat_key(s: str):
        return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

    A_keys = sorted([k for k in lora_clean if "lora_A" in k], key=nat_key)
    B_keys = sorted([k for k in lora_clean if "lora_B" in k], key=nat_key)
    assert len(A_keys) == len(B_keys)
    scale = float(lora_alpha) / float(r)

    num_sq, den_sq = 0.0, 0.0
    for Ak, Bk in zip(A_keys, B_keys):
        A  = lora_clean[Ak].float()
        B  = lora_clean[Bk].float()
        A_ = lora_noisy[Ak].float()
        B_ = lora_noisy[Bk].float()
        DW  = scale * (B  @ A)
        DW_ = scale * (B_ @ A_)
        E   = DW_ - DW
        nf = torch.linalg.matrix_norm(E,  ord='fro').item()
        sf = torch.linalg.matrix_norm(DW, ord='fro').item()
        num_sq += nf*nf
        den_sq += sf*sf
    return math.sqrt(num_sq) / (math.sqrt(den_sq) + 1e-12)

import math
import torch

def add_noise_to_lora_laplace(clean_lp: dict,
                      sigmaA: float,
                      sigmaB: float,
                      seed: int = None,
                      sigma_is_std: bool = True,
                      tau: float = 6.0,
                      max_abs: float = None):
    noisy = {k: v.clone() for k, v in clean_lp.items()}

    bA = sigmaA / math.sqrt(2.0) if sigma_is_std else sigmaA
    bB = sigmaB / math.sqrt(2.0) if sigma_is_std else sigmaB

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    def truncated_laplace_noise_like(t: torch.Tensor, b: float) -> torch.Tensor:
        T = float(max_abs) if (max_abs is not None and max_abs > 0) else float(max(1e-8, (tau * b)))
        ratio = T / max(b, 1e-12)
        a = 0.5 * math.exp(-ratio)
        a = min(max(a, 0.0), 0.499999999)

        calc_dtype = torch.float32 if t.dtype in (torch.float16, torch.bfloat16) else t.dtype

        u = torch.rand(t.shape, dtype=calc_dtype, generator=gen)
        u = a + (1.0 - 2.0 * a) * u  
        if t.device.type != "cpu":
            u = u.to(t.device)

        eps = torch.finfo(calc_dtype).eps
        u = u.clamp(min=a + eps, max=1.0 - a - eps)

        x = torch.empty_like(u, dtype=calc_dtype)
        mask = (u < 0.5)
        x[mask] = b * torch.log(2.0 * u[mask])
        x[~mask] = -b * torch.log(2.0 * (1.0 - u[~mask]))

        x = x.clamp(min=-T, max=T)
        return x.to(dtype=t.dtype)

    with torch.no_grad():
        for k, v in noisy.items():
            if "lora_A" in k and v.ndim >= 2:
                v.add_(truncated_laplace_noise_like(v, bA))
            elif "lora_B" in k and v.ndim >= 2:
                v.add_(truncated_laplace_noise_like(v, bB))

    return noisy
