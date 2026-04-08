import numpy as np
import networkx as nx
import numpy as np
import networkx as nx
from .entropy import spectral_entropy_lap

def closed_neighborhood_nodes(G, v):
    neigh = list(G.neighbors(v))
    nodes = neigh + [v]
    return nodes, neigh

def lap_eigs_from_adj(A):
    deg = np.sum(A, axis=1)
    L = np.diag(deg) - A
    return np.linalg.eigvalsh(L).real

def safe_normalize_eigs(eigvals):
    eigvals = np.asarray(eigvals, dtype=float)
    m = np.max(np.abs(eigvals))
    if m <= 1e-12 or np.isnan(m) or np.isinf(m):
        return eigvals
    return eigvals / m

def probabilistic_closure_perturb_adj(A_loc, neigh_pos, omega, rng):
    A_new = A_loc.copy()
    m = len(neigh_pos)
    if m < 2 or omega <= 0:
        return A_new
    for a in range(m):
        i = neigh_pos[a]
        for b in range(a + 1, m):
            j = neigh_pos[b]
            if A_new[i, j] == 0 and rng.random() < omega:
                A_new[i, j] = 1.0
                A_new[j, i] = 1.0
    return A_new

def compute_nppe(G, beta_grid, samples, omega, seed):
    rng = np.random.default_rng(seed)
    nodelist = sorted(G.nodes())
    N = len(nodelist)
    B = len(beta_grid)
    E = np.full((B, N), np.nan, dtype=float)

    for v_id, v in enumerate(nodelist):
        loc_nodes, _ = closed_neighborhood_nodes(G, v)
        if len(loc_nodes) <= 1:
            E[:, v_id] = 0.0
            continue

        A_loc = nx.to_numpy_array(G, nodelist=loc_nodes).astype(float)
        neigh_pos = list(range(len(loc_nodes) - 1))

        eig0 = safe_normalize_eigs(lap_eigs_from_adj(A_loc))
        S0 = np.array([spectral_entropy_lap(eig0, b) for b in beta_grid], dtype=float)
        S0 = np.nan_to_num(S0, nan=0.0, posinf=0.0, neginf=0.0)

        S_acc = np.zeros(B, dtype=float)
        valid_cnt = np.zeros(B, dtype=float)

        for _ in range(samples):
            A_pert = probabilistic_closure_perturb_adj(A_loc, neigh_pos, omega, rng)
            eigp = safe_normalize_eigs(lap_eigs_from_adj(A_pert))
            Sp = np.array([spectral_entropy_lap(eigp, b) for b in beta_grid], dtype=float)
            mask = ~np.isnan(Sp) & ~np.isinf(Sp)
            S_acc[mask] += Sp[mask]
            valid_cnt[mask] += 1.0

        S_mean = np.where(valid_cnt > 0, S_acc / valid_cnt, np.nan)
        E[:, v_id] = S_mean - S0

    nppe = -np.nanmin(E, axis=0)
    return nppe, E, nodelist