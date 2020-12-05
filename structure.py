import numpy as np

class StructureData:
    '''
    An object that stores the variables used in STRUCTURE 2.0.

    Inputs:
    - X[l, i, a] (int): the genotype of allele copy `a` at locus `l` for individual `i`
    - K (int): the number of ancestral populations

    Want to estimate:
    - Z[l, i, a] (int): the origin of the `a`th allele copy at locus `l` in individual `i`
    - P[k, l, j] (float): the frequency of allele `j` at locus `l` in population `k`
    - Q[k, i] (float): the fraction of individual `i`'s ancestry from population `k`
    - r (float): the number of generations since admixture
    - F[k] (float): TODO figure out what these variables mean
    - alpha
    - lamda
    '''
    def __init__(self, X, K) -> None:
        '''
        Initializes a STRUCTURE 2.0 data object.
        '''
        self.X = X
        self.K = K
    
    def gibbs_round(self):
        '''
        Performs a round of Gibbs sampling.
        TODO In-place or out-of-place?
        '''
