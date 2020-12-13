import numpy as np
import vcf

def read_vcf(filename):
    '''
    Reads a VCF file and converts it to the internal NumPy representation.
    
    :param filename: the path to the VCF file
    :return:
        X: genotype at each locus for each individual
        J: number of alleles for each locus
        POS: position of each locus
    :rtype: (ndarray, ndarray, ndarray)
    '''
    with open(filename, 'r') as f:
        vcf_reader = vcf.Reader(f)
        vcf_records = list(vcf_reader)
    # Get number of samples, number of loci, names of samples
    samples = vcf_reader.samples
    num_samples = len(samples)
    num_loci = len(vcf_records)
    # X = genotypes, J = number of alleles for each locus
    X = np.zeros((num_loci, num_samples, 2), dtype=np.uint8)
    J = np.zeros(num_loci, dtype=np.uint8)
    # POS = position of each locus on the chromosome
    # The longest human chromosome is 248,956,422 bp long, which is shorter than 2^32
    POS = np.zeros(num_loci, dtype=np.uint32)
    for l, locus in enumerate(vcf_records):
        # The number of alleles at locus l
        J[l] = len(locus.alleles)
        # The position of locus l
        POS[l] = locus.POS
        # Genotypes for each individual (assumed diploid and phased)
        for i, sample in enumerate(samples):
            call = locus.genotype(sample)
            # Parse call data into numpy array
            alleles = np.fromiter(map(int, call.data.GT.split('|')), dtype=np.uint8)
            # Then, assign it to the appropriate row of X
            X[l, i, :] = alleles
    return X, J, POS

def drop_loci(X, J, POS, frac=0.3):
    '''
    Randomly drops loci.

    Parameters:
    - frac (float): the fraction of loci to drop
    '''
    num_loci = X.shape[0]
    # Generate random boolean mask
    unif = np.random.random(num_loci)
    mask = (unif >= frac)
    return X[mask], J[mask], POS[mask]

def test_read_vcf(filename):
    X, J, POS = read_vcf(filename)
    print(X.shape, J.shape, POS.shape)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read a VCF file and parse it into the internal NumPy representation')
    parser.add_argument('path', metavar='The path to the VCF file to be read')
    args = parser.parse_args()
    test_read_vcf(args.path)

