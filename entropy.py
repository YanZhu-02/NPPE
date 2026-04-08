import numpy as np

def spectral_entropy_lap(eigvals, beta):
    exp_terms = np.exp(-beta * eigvals)
    Z = np.sum(exp_terms)
    if Z == 0 or np.isinf(Z) or np.isnan(Z):
        return np.nan
    t = -beta * eigvals
    S = -np.sum(exp_terms * (t / np.log(2) - np.log2(Z)) / Z)
    return S