import numpy as np
from scipy.stats import dirichlet, mode
from itertools import product
from copy import deepcopy
from operator import attrgetter
from tqdm import trange

import argparse
import h5py
from preprocess import read_file, subst_file_suffix
import os
import cProfile

class Structure:
    '''
    Implements the STRUCTURE 1.0 population structure model with admixture.

    Inputs:
    - X[l, i, a] (int): the genotype of allele copy `a` at locus `l` for individual `i`
    - J[l] (int): the number of possible alleles at locus `l`
    - POS[l] (int): unused, but required for ROLLOFF
    - K (int): the number of populations

    Want to estimate:
    - Z[l, i, a] (int): the origin of the `a`th allele copy at locus `l` in individual `i`
    - P[k, l, j] (float): the frequency of allele `j` at locus `l` in population `k`
    - Q[i, k] (float): the fraction of individual `i`'s ancestry from population `k`
    - alpha (float): a parameter to the underlying Dirichlet distribution. Indicates the
      degree of admixture in the populations.
    '''
    def __init__(self, X: np.ndarray, J: np.ndarray, POS: np.ndarray, K: int) -> None:
        '''
        Initializes a STRUCTURE data object.
        '''
        # Copy inputs
        self.X = X
        self.J = J
        self.POS = POS
        self.K = K
        # Convenience parameters: number of loci, number of individuals, and number of copies
        # per allele (ploidy)
        self.num_loci, self.sample_size, self.ploidy = self.X.shape
        # Initialize generator
        self.rng = np.random.default_rng()
        # Initialize Z: draw each Z[l, i, a] uniformly from k in {0, ..., K-1}
        self.Z = self.rng.integers(self.K, size=self.X.shape)
        # Initialize P: empty K * num_loci * max(J) array
        self.P = np.zeros((self.K, self.num_loci, np.max(self.J)))
        # Initialize Q: empty K * sample_size array
        self.Q = np.zeros((self.sample_size, self.K))
        # Initialize alpha: draw uniformly from [0, 10)
        self.alpha = self.rng.uniform(10)

    def step_pq(self):
        '''
        Performs the first step in the Gibbs training loop: sampling P, Q from Pr(P, Q | X, Z)
        '''
        # Sample P
        for l in range(self.num_loci):
            # N[k, j] = "the number of copies of allele j at locus l ... assigned (by Z)
            # to population k" (Algorithm A2, Pritchard et al. 2000)
            N = np.zeros((self.K, self.J[l]), dtype=np.uint64)
            for k, j in product(range(self.K), range(self.J[l])):
                N[k, j] = np.count_nonzero((self.X[l] == j) & (self.Z[l] == k))
            # Sample each P[k, l, :] from Dirichlet (N[k] + lambda)
            # Using different values of lambda[j] for each allele allows us to model correlations
            # between allele frequencies. This assumption is better for closely related populations.
            # We use lambda = 1.0, implying that allele frequencies are independent.
            lmbda = 1.0
            for k in range(self.K):
                # Since the sampled vector is variable-length, we assign it only to the first
                # J[l] elements of P[k, l]
                self.P[k, l, 0:self.J[l]] = self.rng.dirichlet(N[k] + lmbda)
        # Sample Q
        M = np.zeros_like(self.Q)
        for i in range(self.sample_size):
            # M[i, k] = "the number of allele copies in individual i that originated (according to Z)
            # in population k" (Algorithm A3, Pritchard et al. 2000)
            for k in range(self.K):
                M[i, k] = np.count_nonzero(self.Z[:, i, :] == k)
            # Sample each Q[i] from Dirichlet (M[i] + alpha)
            self.Q[i] = self.rng.dirichlet(M[i] + self.alpha)

    def step_z(self):
        '''
        Performs the second step in the Gibbs training loop: sampling Z from Pr(Z | X, P, Q)
        '''
        ## This vectorized sampling algorithm converts continuous uniform samples to k-values
        ## by using the cdf of Pr(Z[l, i, a] | X, P, Q). Previously, we used a naive
        ## sampling algorithm based on calling rng.choice() once for each [l, i, a]. This took
        ## a very long time to run, and rng.choice() cannot be vectorized.

        # Compute the pmf of Pr(Z[l, i, a] = k | X, P, Q) using Equation A12
        pmf_k = np.zeros((self.num_loci, self.sample_size, self.ploidy, self.K))
        for l in range(self.num_loci):
            pmf_k[l] = np.einsum('ik,kia->iak', self.Q, self.P[:, l, self.X[l]])
        pmf_k /= np.sum(pmf_k, axis=3, keepdims=True)
        # Convert to cdf; remove the last "hyper-row" (always 1.0)
        cdf = np.cumsum(pmf_k[:, :, :, :-1], axis=3)
        # Draw random samples from continuous uniform U(0, 1) for each [l, i, a]
        uniform_samples = self.rng.random(self.Z.shape + (1,))
        # Convert the uniform samples to k-values by counting the number of k-indices for which
        # the uniform sample is greater than the cdf at that index
        self.Z = np.sum(uniform_samples > cdf, axis=3)
    
    def log_likelihood_alpha(self, alpha):
        '''
        Returns the log likelihood of alpha given the current values of Q.
        The likelihood function of alpha is equal to the product of the probabilities of
        each Q[i] as drawn from a Dirichlet distribution where all concentration parameters
        are set to alpha.
        '''
        log_likelihoods = np.zeros(self.sample_size)
        alpha_vec = np.full(self.K, alpha)
        for i in range(self.sample_size):
            log_likelihoods[i] = dirichlet.logpdf(self.Q[i], alpha_vec)
        return np.sum(log_likelihoods)

    def step_alpha(self):
        '''
        Performs a Metropolis-Hastings step for alpha.
        '''
        # Sample a new alpha from N(alpha, 0.25)
        new_alpha = self.rng.normal(self.alpha, 0.25)
        # alpha must be in (0, 10)
        if 0 < new_alpha and new_alpha < 10:
            # Compute the likelihood ratio of the old and new alphas
            likelihood_ratio = np.exp(self.log_likelihood_alpha(new_alpha) - self.log_likelihood_alpha(self.alpha))
            # Draw a uniform random variable between 0 and 1
            uniform = self.rng.random()
            # With probability `likelihood_ratio`, accept the new alpha
            if uniform <= likelihood_ratio:
                self.alpha = new_alpha
    
    def gibbs_round(self):
        '''
        Performs one round of Gibbs sampling. See Algorithm 2 in Pritchard et al. (2000).
        '''
        self.step_pq()
        self.step_z()
        self.step_alpha()

    def gibbs_sampling(self, m: int = 1000, c: int = 50, num_samples: int = 30):
        '''
        Performs a Gibbs sampling loop.

        Parameters:
        - m (int): the burn-in period
        - c (int): the number of rounds between samples
        - num_samples (int): the number of samples to take
        '''
        # Burn-in: discard the first several rounds
        for _ in trange(m, desc='Burn-in rounds'):
            self.gibbs_round()
        # Take a sample every t rounds
        samples = []
        for sample_num in trange(num_samples, desc='Take samples'):
            for _ in trange(c, desc=f'Rounds for sample {sample_num}'):
                self.gibbs_round()
            # Use deepcopy to take a snapshot of the object state
            samples.append(deepcopy(self))
        # Aggregate all samples
        Z_all = np.stack(list(map(attrgetter('Z'), samples)), axis=0)
        P_all = np.stack(list(map(attrgetter('P'), samples)), axis=0)
        Q_all = np.stack(list(map(attrgetter('Q'), samples)), axis=0)
        alpha_all = np.array(list(map(attrgetter('alpha'), samples)))
        # Take the mean of P, Q, alpha, and the mode of Z; and assign them to this object
        self.Z = mode(Z_all, axis=0)[0].squeeze(axis=0)
        self.P = np.mean(P_all, axis=0)
        self.Q = np.mean(Q_all, axis=0)
        self.alpha = np.mean(alpha_all)
    
    def save(self, hdf5_file):
        '''
        Saves the state of this data object to an HDF5 file.
        '''
        with h5py.File(hdf5_file, 'w') as f:
            attrs_to_store = ['X', 'J', 'POS', 'K', 'Z', 'P', 'Q', 'alpha']
            for attr in attrs_to_store:
                f[attr] = getattr(self, attr)

def parse_args():
    parser = argparse.ArgumentParser(description='Infer the population structure of a genetic sample using the STRUCTURE algorithm')
    parser.add_argument('file', metavar='The data file (*.vcf or .phgeno)')
    parser.add_argument('-k', type=int, default=2, metavar='The number of populations')
    parser.add_argument('-o', '--out', metavar='The file to which the result will be written in HDF5 format')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('-d', '--drop-frac', type=float, default=0.6, metavar='Randomly drop this fraction of loci')
    parser.add_argument('-b', '--burn-in', type=int, default=400, metavar='The burn-in period (rounds)')
    parser.add_argument('-n', '--num-samples', type=int, default=20, metavar='The number of samples to collect')
    parser.add_argument('-p', '--sample-interval', type=int, default=20, metavar='The number of rounds between samples')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    # Read from file and randomly drop loci
    X, J, POS = read_file(args.file, drop_frac=args.drop_frac)
    # Run the STRUCTURE algorithm
    structure = Structure(X, J, POS, args.k)
    if args.profile:
        cProfile.runctx('structure.gibbs_round()', {}, {'structure': structure}, sort='cumtime')
    else:
        structure.gibbs_sampling(args.burn_in, args.sample_interval, args.num_samples)
        # Create output file using command-line argument if provided. If not, construct
        # the filename by replacing the file extension with .hdf5
        if args.out:
            out = args.out
        else:
            out = subst_file_suffix(args.file, '.hdf5')
        # Write to the output file
        structure.save(out)

if __name__ == '__main__':
    main()
