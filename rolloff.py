import numpy as np
from numpy.core.fromnumeric import prod
from scipy.stats import pearsonr
from itertools import product

def genetic_distance(pos: np.ndarray) -> np.ndarray:
    '''
    Computes the genetic distance in morgans between all pairs of loci in `pos`.

    Parameters:
    - pos[l] (float): the position of locus `l` in bps

    Returns:
    - dist[l1, l2] (float): the genetic distance between `l1` and `l2` in morgans
    '''
    # Compute the physical distances between all pairs of loci
    dist_bps = np.abs(pos[np.newaxis, :] - pos[:, np.newaxis])
    # Convert to genetic distance in morgans (1 centimorgan = about 1 million bp)
    return dist_bps / 1e8

def weight(P: np.ndarray, k1: int = 0, k2: int = 1) -> np.ndarray:
    '''
    Computes the rolloff weight function for each locus, for two populations.

    Parameters:
    - P[k, l, j] (float): the frequency of allele `j` at locus `l` in population `k`
    - k1, k2 (int): two populations chosen for this function. Default: 0, 1

    Returns:
    - w[l] (float): the weight function for locus `l`
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

def bin_by_distance(dist: np.ndarray, w_prod: np.ndarray, z: np.ndarray):
    '''
    Sorts all pairs of loci by genetic distance, then bins them 
    '''
    # Sort all pairs of loci using dist as a sort key
    dist, w_prod, z = dist.flatten(), w_prod.flatten(), z.flatten()
    sort_indices = dist.argsort()
    dist, w_prod, z = dist[sort_indices], w_prod[sort_indices], z[sort_indices]
    # Bin size = 0.1 cM
    bin_size = 0.1e-2


def num_generations(X: np.ndarray, P: np.ndarray, pos: np.ndarray, k1: int = 0, k2: int = 1) -> float:
    '''
    Estimates the number of generates since admixture between populations `k1` and `k2`.

    Parameters:
    - X[l, i, a] (int): the genotype of allele copy `a` at locus `l` for individual `i`
    - P[k, l, j] (float): the frequency of allele `j` at locus `l` in population `k`
    - k1, k2 (int): two populations chosen for this function. Default: 0, 1
    '''
    # Compute weights, LD scores, and distances for all loci
    w = weight(P, k1, k2)
    z = ld_score(X)
    dist = genetic_distance(pos)
    # Compute the outer product of w with itself, i.e. w(l1) * w(l2) for all l1, l2
    w_prod = np.outer(w, w)
    # Sort all cells by genetic distance
    bin_by_distance(dist, w_prod, z)
