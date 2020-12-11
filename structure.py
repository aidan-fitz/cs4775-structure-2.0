import numpy as np
from itertools import product

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
        # Initialize alpha: arbitary constant
        self.alpha = 1.0
        # "Niceness": a quantity proportional to the likelihood of this model (log-space)
        self.niceness = -np.inf

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
                self.P[k, l, :] = self.rng.dirichlet(N[k] + lmbda)
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
        for l, i, a in product(range(self.num_loci), range(self.sample_size), range(self.ploidy)):
            # Sample Z[l, i, a] from Pr(Z[l, i, a] = k | X, P, Q) using Equation A12
            prob_k = self.Q[i] * self.P[:, l, self.X[l, i, a]]
            prob_k /= np.sum(prob_k)
            self.Z[l, i, a] = self.rng.choice(self.K, p=prob_k)
    
    def step_alpha(self):
        '''
        Performs a Metropolis-Hastings step for alpha.
        '''
    
    def gibbs_round(self):
        '''
        Performs one round of Gibbs sampling. See Algorithm 2 in Pritchard et al. (2000).
        '''
        self.step_pq()
        self.step_z()
        self.step_alpha()
