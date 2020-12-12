import numpy as np
from numpy.core.fromnumeric import prod
from scipy.stats import pearsonr
from itertools import product

def genetic_distance(l1: int, l2: int) -> float:
    '''
    Computes the genetic distance in morgans between the loci at positions `l1` and `l2`.
    '''
    # One centimorgan = about 1 million bp
    return np.abs(l1 - l2) / 1e8

def weight(P: np.ndarray, k1: int = 0, k2: int = 1) -> np.ndarray:
    '''
    Computes the rolloff weight function for each locus, for two populations.

    Parameters:
    - P[k, l, j] (float): the frequency of allele `j` at locus `l` in population `k`
    - k1, k2 (int): two populations chosen for this function. Default: 0, 1
    '''
    # Caution: Each P[k, l, :] sums to 1
    # Choose the alleles with the highest frequency in each population at each locus
    max_allele_freq = np.amax(P, axis=2)
    # Choose k1, k2
    a, b = max_allele_freq[k1], max_allele_freq[k2]
    p = (a + b) / 2
    return (a - b) / np.sqrt(p * (1 - p))

def ld_score(X: np.ndarray) -> np.ndarray:
    '''
    Computes the rolloff LD scores between all pairs of loci in the sample.

    Parameters:
    - X[l, i, a] (int): the genotype of allele copy `a` at locus `l` for individual `i`

    Returns:
    - ld[l1, l2] (float): the rolloff LD score `z(l1, l2)`
    '''
    # Get number of loci, number of samples
    L, N = X.shape[:2]
    # Compute Pearson correlation coefficients between all loci
    corr = np.zeros((L, L))
    for l1, l2 in product(range(L), range(L)):
        corr[l1, l2] = pearsonr(X[l1].flatten(), X[l2].flatten())
    # Clip correlation coefficients between [-0.9, 0.9]
    np.clip(corr, -0.9, 0.9, out=corr)
    # Fisher z-transformation
    z_transform = np.arctanh(corr)
    # Multiply by sqrt(N - 3) and return
    return np.sqrt(N - 3) * z_transform
