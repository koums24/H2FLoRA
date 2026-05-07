import math
import random
from typing import List, Dict, Tuple
import numpy as np
from utils import compute_utilities


class GreedyNoiseSelector:

    def __init__(
        self,
        sigma_max: float,
        alpha: List[float],      
        beta:  List[float],     
        sigma_init: List[float], 
        c: float = 0.05,          
        use_ema: bool = True,   
        rho: float = 0.1        
    ):
        assert sigma_max > 0, "sigma_max must be > 0"


        # self.S_base = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
        # self.S_base = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.S_base = [0.01, 0.1, 0.5, 1.0]

        self.sigma_max = float(sigma_max)
        self.S = [s * self.sigma_max for s in self.S_base]

        alpha = np.asarray(alpha, dtype=float).reshape(-1)
        beta  = np.asarray(beta,  dtype=float).reshape(-1)
        assert len(alpha) == len(beta), "alpha and beta must have the same length"
        self.num_clients = len(alpha)
        self.alpha: List[float] = alpha
        self.beta:  List[float] = beta

        sigma_init = np.asarray(sigma_init, dtype=float).reshape(-1)
        if len(sigma_init) != self.num_clients:
            raise ValueError("sigma_init length must equal number of clients")
        self.sigma = sigma_init.tolist()

        self.mu: List[Dict[float, float]] = [{s: 0.0 for s in self.S} for _ in range(self.num_clients)]
        self.N:  List[Dict[float, int]]   = [{s: 0   for s in self.S} for _ in range(self.num_clients)]

        self.c = float(c)
        self.use_ema = bool(use_ema)
        self.rho = float(rho)
        
    def select_ucb_arm(self, mu, N, t, arms, c=1.0):
            untried = [s for s in arms if N[s] == 0]
            if untried:
                return random.choice(untried)

            t = max(2, int(t))
            best_s, best_idx = None, float("-inf")
            for s in arms:
                idx = mu[s] + c * math.sqrt(math.log(t) / N[s])
                if idx > best_idx:
                    best_s, best_idx = s, idx
            return best_s
        
    def update_and_select_next(
        self,
        t: int,
        acc_t                 
    ) -> List[float]:
        acc_t = np.asarray(acc_t, dtype=float).reshape(-1)
        assert len(acc_t) == self.num_clients, "acc_t length mismatch"

        eps = min(1.0, self.c / max(1.0, math.sqrt(max(1, t))))
        sigma_next = np.zeros(self.num_clients, dtype=float)

        for i in range(self.num_clients):
            s_prev = self._snap_to_grid(float(self.sigma[i]))

            U_obs, norm_utility = compute_utilities(self.alpha[i], self.beta[i], acc_t[i], s_prev, self.sigma_max)

            if self.use_ema:
                self.mu[i][s_prev] = (1.0 - self.rho) * self.mu[i][s_prev] + self.rho * U_obs
                self.N[i][s_prev] += 1
            else:
                self.N[i][s_prev] += 1
                n = self.N[i][s_prev]
                self.mu[i][s_prev] += (U_obs - self.mu[i][s_prev]) / n
            
            t_i = max(1, sum(self.N[i].values()))
            c = self.c
            s_next = self.select_ucb_arm(self.mu[i], self.N[i], t_i, self.S, c=c/5)
            
            
            sigma_next[i] = s_next

        self.sigma = sigma_next.tolist()
        return self.sigma

    def get_alphas_betas(self) -> List[Tuple[float, float]]:
        return list(zip(self.alpha, self.beta))

    def get_sigma(self) -> List[float]:
        return list(self.sigma)

    def set_sigma(self, sigma_new: List[float]) -> None:
        sigma_new = np.asarray(sigma_new, dtype=float).reshape(-1)
        if len(sigma_new) != self.num_clients:
            raise ValueError("sigma_new length must equal number of clients")
        self.sigma = sigma_new.tolist()

    

    def _snap_to_grid(self, sigma: float) -> float:
        best_s, best_d = self.S[0], float("inf")
        for s in self.S:
            d = abs(s - sigma)
            if d < best_d:
                best_s, best_d = s, d
        return best_s

    @staticmethod
    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)