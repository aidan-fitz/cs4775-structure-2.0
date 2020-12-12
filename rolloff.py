import numpy as np
from scipy.stats import pearsonr, linregress
from itertools import product

def genetic_distance(pos: np.ndarray) -> np.ndarray:
    '''
    Computes the genetic distance in morgans between all pairs of loci in `pos`.

    Parameters:
    - pos[l] (int): the position of locus `l` in bps

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
    # Ensure N >= 4
    if N < 4:
        raise ValueError('Number of individuals in sample must be at least 4')
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

def bin_by_distance(dist: np.ndarray, min_bin_size: float = 9e-4):
    '''
    Assigns distances (in morgans) to bins without sorting.

    Parameters:
    - dist[l1, l2] (float): the genetic distance between `l1` and `l2` in morgans
    - min_bin_size (float): the minimum size in morgans of the bins that this function
    should create. Default: 0.09 cM. ValueError is raised if this value is less than 0.05 cM.

    Returns:
    - bin_indices[l1, l2] (int): the zero-based index of the bin that `(l1, l2)` should be assigned to.
    - num_bins (int): the number of bins
    '''
    # Ensure min_bin_size is at least 0.05 cM
    if min_bin_size < 5e-4:
        raise ValueError('Bin size must be at least 0.05 centimorgans')
    # Set the max distance as the highest bin value
    # Since dist is now sorted, the highest value must be at the end
    max_dist = dist[-1]
    # Create evenly spaced bins no less than 0.05 cM wide
    # Add a tiny amount to max_dist while computing bin boundaries so that max_dist
    # falls within the last bin instead of just outside it
    num_boundaries = int(np.ceil(max_dist / min_bin_size))
    bin_boundaries = np.linspace(0, max_dist + 1e-6, num_boundaries)
    # Assign elements of the input tensor to bins and return the number of bins
    bin_indices = np.digitize(dist, bin_boundaries) - 1
    num_bins = num_boundaries - 1
    return bin_indices, num_bins


def num_generations(X: np.ndarray, P: np.ndarray, pos: np.ndarray, k1: int = 0, k2: int = 1) -> float:
    '''
    Estimates the number of generations since admixture between populations `k1` and `k2`.

    Parameters:
    - X[l, i, a] (int): the genotype of allele copy `a` at locus `l` for individual `i`
    - P[k, l, j] (float): the frequency of allele `j` at locus `l` in population `k`
    - pos[l] (int): the position of locus `l` in bps
    - k1, k2 (int): two populations chosen for this function. Default: 0, 1

    Returns:
    - n (float): the estimated number of generations since admixture
    '''
    # Compute weights, LD scores, and distances for all loci
    w = weight(P, k1, k2)
    z = ld_score(X)
    dist = genetic_distance(pos)
    # Compute the outer product of w with itself, i.e. w(l1) * w(l2) for all l1, l2
    w_prod = np.outer(w, w)
    # Assign all data points to bins based on genetic distance
    bin_indices, num_bins = bin_by_distance(dist)
    # Compute rolloff statistics for all bins
    coeff, dist_bin = np.zeros(num_bins), np.zeros(num_bins)
    for bin in range(num_bins):
        # Select the data points in this bin
        bin_mask = (bin_indices == bin)
        w_bin, z_bin, data_dists_bin = w_prod[bin_mask], z[bin_mask], dist[bin_mask]
        # Use the average distance to represent this bin
        dist_bin[bin] = np.mean(data_dists_bin)
        # Compute the binned correlation coefficient
        coeff[bin] = pearsonr(w_bin, z_bin)
    # Fit the binned coefficients to an exponential curve given by coeff = exp(-n * dist_bin)
    # via log-linear regression
    slope = linregress(dist, np.log(coeff))[0]
    return -slope
