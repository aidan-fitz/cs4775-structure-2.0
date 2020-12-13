import numpy as np
from scipy.stats import dirichlet, mode
from itertools import product
from copy import deepcopy

import argparse
import h5py
from vcf import parser
from preprocess import read_vcf
import cProfile

class Structure:
    '''
    Implements the STRUCTURE 1.0 population structure model with admixture.

    Inputs:
    - X[l, i, a] (int): the genotype of allele copy `a` at locus `l` for individual `i`
    - J[l] (int): the number of possible alleles at locus `l`
    - K (int): the number of populations

    Want to estimate:
    - Z[l, i, a] (int): the origin of the `a`th allele copy at locus `l` in individual `i`
    - P[k, l, j] (float): the frequency of allele `j` at locus `l` in population `k`
    - Q[i, k] (float): the fraction of individual `i`'s ancestry from population `k`
    - alpha (float): a parameter to the underlying Dirichlet distribution. Indicates the
      degree of admixture in the populations.
    '''
    def __init__(self, X: np.ndarray, J: np.ndarray, K: int) -> None:
        '''
        Initializes a STRUCTURE data object.
        '''
        # Copy inputs
        self.X = X
        self.J = J
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
        for l, i, a in product(range(self.num_loci), range(self.sample_size), range(self.ploidy)):
            pmf_k[l, i, a] = self.Q[i] * self.P[:, l, self.X[l, i, a]]
        pmf_k /= np.sum(pmf_k, axis=3, keepdims=True)
        # Convert to cdf; remove the last "hyper-row" (always 1.0)
        cdf = np.cumsum(pmf_k, axis=3)[:, :, :, :-1]
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
        for _ in range(m):
            self.gibbs_round()
        # Take a sample every t rounds
        samples = []
        for _ in range(num_samples):
            for _ in range(c):
                self.gibbs_round()
            # Use deepcopy to take a snapshot of the object state
            samples.append(deepcopy(self))
        # Aggregate all samples
        Z_all = np.stack(map(lambda obj: obj.Z, samples), axis=0)
        P_all = np.stack(map(lambda obj: obj.P, samples), axis=0)
        Q_all = np.stack(map(lambda obj: obj.Q, samples), axis=0)
        alpha_all = np.fromiter(map(lambda obj: obj.alpha, samples))
        # Take the mean of P, Q, alpha, and the mode of Z; and assign them to this object
        self.Z, _ = mode(Z_all, axis=0)
        self.P = np.mean(P_all, axis=0)
        self.Q = np.mean(Q_all, axis=0)
        self.alpha = np.mean(alpha_all)

def parse_args():
    parser = argparse.ArgumentParser(description='Infer the population structure of a genetic sample using the STRUCTURE algorithm')
    parser.add_argument('file', metavar='The data file containing the sample in VCF format')
    parser.add_argument('-k', type=int, default=2, metavar='The number of populations')
    parser.add_argument('-o', '--out', metavar='The file to which the result will be written in HDF5 format')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    # Read from file
    X, J, POS = read_vcf(args.file)
    # Run the STRUCTURE algorithm
    structure = Structure(X, J, args.k)
    # for i in range(5):
    #     structure.gibbs_round()
    #     print(f'Round {i}')
    cProfile.runctx('structure.gibbs_round()', None, {'structure': structure}, sort='cumtime')

if __name__ == '__main__':
    main()
