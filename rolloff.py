from preprocess import hdf5_to_numpy, subst_file_suffix
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from tqdm import trange
import numba

import matplotlib.pyplot as plt
import argparse
import h5py
import cProfile

def genetic_distance(POS: np.ndarray) -> np.ndarray:
    '''
    Computes the genetic distance in morgans between all pairs of loci in `POS`.

    Parameters:
    - POS[l] (int): the position of locus `l` in bps

    Returns:
    - dist[l1, l2] (float): the genetic distance between `l1` and `l2` in morgans
    '''
    # Compute the physical distances between all pairs of loci
    # Results must be signed integers
    dist_bps = np.abs(np.subtract.outer(POS, POS, dtype=np.int32))
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
    # Use the frequency of allele 0 at each k, l
    a, b = P[k1, :, 0], P[k2, :, 0]
    p = (a + b) / 2
    return (a - b) / np.sqrt(p * (1 - p))

@numba.njit
def symmetric_outer(X):
    '''
    Performs a symmetric outer product of X with itself, taking the dot products of the rows.
    Implementation due to https://stackoverflow.com/a/51481295
    '''
    n = X.shape[0]
    res = np.empty((n, n), X.dtype)
    for i in range(n):
        for j in range(i+1):
            res[i, j] = np.dot(X[i], X[j])
            res[j, i] = res[i, j]
    return res

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
    print(f'Computing LD scores ({L} loci, {N} samples). Result will be a {L}*{L} matrix.')
    # Compute Pearson correlation coefficients between all loci (vectorized)
    X_flat = X.reshape((L, -1))
    X_flat = X_flat - np.mean(X_flat, axis=1)[:, None] # this casts X_flat from int to float
    # Numerator: essentially a sum of outer products
    corr_num = symmetric_outer(X_flat)
    # Denominator: square root of outer product of sums of squares
    X_sumsq = np.sum(X_flat**2, axis=1)
    corr_denom = np.sqrt(np.outer(X_sumsq, X_sumsq))
    corr = corr_num / corr_denom
    # Change infinities to NaNs to be consistent with scipy.stats.pearsonr
    corr[~np.isfinite(corr)] = np.nan
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
    - max_dist (float): the maximum genetic distance in morgans
    '''
    # Ensure min_bin_size is at least 0.05 cM
    if min_bin_size < 5e-4:
        raise ValueError('Bin size must be at least 0.05 centimorgans')
    # Set the max distance as the highest bin value
    max_dist = np.max(dist)
    # Create evenly spaced bins no less than 0.05 cM wide
    # Add a tiny amount to max_dist while computing bin boundaries so that max_dist
    # falls within the last bin instead of just outside it
    num_boundaries = int(np.ceil(max_dist / min_bin_size))
    bin_boundaries = np.linspace(0, max_dist + 1e-6, num_boundaries)
    # Assign elements of the input tensor to bins and return the number of bins
    bin_indices = np.digitize(dist, bin_boundaries) - 1
    num_bins = num_boundaries - 1
    return bin_indices, num_bins, max_dist

def rolloff(d: np.ndarray, n: float, a: float):
    return a * np.exp(-n * d)

def num_generations(X: np.ndarray, P: np.ndarray, POS: np.ndarray, k1: int = 0, k2: int = 1) -> float:
    '''
    Estimates the number of generations since admixture between populations `k1` and `k2`.

    Parameters:
    - X[l, i, a] (int): the genotype of allele copy `a` at locus `l` for individual `i`
    - P[k, l, j] (float): the frequency of allele `j` at locus `l` in population `k`
    - POS[l] (int): the position of locus `l` in bps
    - k1, k2 (int): two populations chosen for this function. Default: 0, 1

    Returns:
    - n (float): the estimated number of generations since admixture
    - a (float): a scale parameter for the rolloff curve
    - ncov (float): the variance of n
    - dist_bin (np.ndarray): the mean genetic distance of each bin
    - coeff (np.ndarray): the rolloff coefficient of each bin
    - max_dist (float): the maximum genetic distance in morgans
    '''
    # Compute weights, LD scores, and distances for all loci
    w = weight(P, k1, k2)
    z = ld_score(X)
    dist = genetic_distance(POS)
    # Compute the outer product of w with itself, i.e. w(l1) * w(l2) for all l1, l2
    w_prod = np.outer(w, w)
    # Assign all data points to bins based on genetic distance
    bin_indices, num_bins, max_dist = bin_by_distance(dist, 6e-4)
    # Compute rolloff statistics for all bins
    coeff, dist_bin = np.zeros(num_bins), np.zeros(num_bins)
    for bin in trange(num_bins, desc='Compute rolloff coefficients'):
        # Select the data points in this bin, and filter out infinities and NaNs
        bin_mask = (bin_indices == bin) & np.isfinite(w_prod) & np.isfinite(z)
        w_bin, z_bin, data_dists_bin = w_prod[bin_mask], z[bin_mask], dist[bin_mask]
        # Use the average distance to represent this bin
        dist_bin[bin] = np.mean(data_dists_bin)
        if len(w_bin) >= 2:
            # Compute the binned correlation coefficient
            coeff[bin] = pearsonr(w_bin, z_bin)[0]
        else:
            coeff[bin] = np.nan
    print('Distances:', dist_bin)
    print('Rolloff coefficients:', coeff)
    # Fit the binned coefficients to an exponential curve given by coeff = a * exp(-n * dist_bin)
    est_params, est_covariance = curve_fit(rolloff, dist_bin, coeff)
    return est_params[0], est_params[1], est_covariance[0, 0], dist_bin, coeff, max_dist

def plot_rolloff(n, a, dist_bin, coeff, max_dist, base_filename):
    '''
    Plot the rolloff coefficients and best-fit curve as a function of distance and save the chart
    to a file.

    Parameters:
    - n (float): the estimated number of generations since admixture
    - a (float): a scale parameter for the rolloff curve
    - dist_bin (np.ndarray): the mean genetic distance of each bin
    - coeff (np.ndarray): the rolloff coefficient of each bin
    - max_dist (float): the maximum genetic distance in morgans
    - base_filename (str): the base filename used to derive the filename for the chart
    '''
    # Plot distances and rolloff coefficients
    plt.scatter(dist_bin, coeff)
    # Plot best-fit curve
    xs = np.linspace(0, max_dist, num=len(dist_bin) * 2)
    plt.plot(xs, rolloff(xs, n, a))
    # Add labels
    plt.xlabel('Genetic distance (morgans)')
    plt.ylabel('Rolloff coefficient')
    plt.savefig(subst_file_suffix(base_filename, '_rolloff.png'))

def parse_args():
    parser = argparse.ArgumentParser(description='Infer the number of generations since admixture')
    parser.add_argument('file', metavar='The data file in HDF5 format')
    parser.add_argument('--profile', action='store_true')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    with h5py.File(args.file, 'r') as file:
        X, P, POS = map(hdf5_to_numpy, [file['X'], file['P'], file['POS']])
        if args.profile:
            cProfile.runctx(
                'num_generations(X, P, POS)',
                {'num_generations': num_generations},
                {'X': X, 'P': P, 'POS': POS},
                sort='cumtime')
        else:
            num_gen, a, n_cov, dist_bin, coeff, max_dist = num_generations(X, P, POS)
            n_stdev = np.sqrt(n_cov)
            print(f'Number of generations: {num_gen}')
            print(f'Standard deviation: {n_stdev}')
            plot_rolloff(num_gen, a, dist_bin, coeff, max_dist, args.file)

if __name__ == '__main__':
    main()

