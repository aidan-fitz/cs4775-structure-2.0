import numpy as np

class Structure:
    '''
    Implements the STRUCTURE 1.0 population structure model with admixture.

    Inputs:
    - X[l, i, a] (int): the genotype of allele copy `a` at locus `l` for individual `i`
    - K (int): the number of populations

    Want to estimate:
    - Z[l, i, a] (int): the origin of the `a`th allele copy at locus `l` in individual `i`
    - P[k, l, j] (float): the frequency of allele `j` at locus `l` in population `k`
    - Q[k, i] (float): the fraction of individual `i`'s ancestry from population `k`
    - alpha (float): a parameter to the underlying Dirichlet distribution. Indicates the
      degree of admixture in the populations.
    '''
    def __init__(self, X: np.ndarray, K: int) -> None:
        '''
        Initializes a STRUCTURE data object.
        '''
        # Copy inputs
        self.X = X
        self.K = K
        # Convenience parameters: number of loci, number of individuals, and number of copies
        # per allele (ploidy)
        self.num_loci, self.sample_size, self.ploidy = self.X.shape
        # Initialize generator
        self.rng = np.random.default_rng()
        # Initialize Z: draw each Z[l, i, a] uniformly from k in {0, ..., K-1}
        self.Z = self.rng.integers(self.K, size=self.X.shape)
        # "Niceness": a quantity proportional to the likelihood of this model (log-space)
        self.niceness = -np.inf

    def step_pq(self):
        '''
        Performs the first step in the Gibbs training loop: sampling P, Q from Pr(P, Q | X, Z)
        '''

    def step_z(self):
        '''
        Performs the second step in the Gibbs training loop: sampling Z from Pr(Z | X, P, Q)
        '''
    
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
