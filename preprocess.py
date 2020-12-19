import numpy as np
import vcf
from tqdm import tqdm, trange
import os
import h5py

import argparse

def vcf_drop_loci(vcf_reader: vcf.Reader, drop_frac=0.6):
    '''
    Reads in the VCF file, randomly dropping a given fraction of records.

    Parameters:
    - drop_frac (float): the fraction of records to drop
    '''
    rng = np.random.default_rng()
    for record in vcf_reader:
        unif = rng.random()
        if unif >= drop_frac:
            yield record

def read_vcf(filename, drop_frac=0.6):
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
        vcf_records = list(tqdm(vcf_drop_loci(vcf_reader, drop_frac), desc=f'Reading data file (drop {drop_frac:.0%} of loci)'))
    # Get number of samples, number of loci, names of samples
    samples = vcf_reader.samples
    num_samples = len(samples)
    num_loci = len(vcf_records)
    # X = genotypes, J = number of alleles for each locus
    X = np.zeros((num_loci, num_samples, 2), dtype=np.uint8)
    J = np.zeros(num_loci, dtype=np.uint8)
    # POS = position of each locus on the chromosome
    # The longest human chromosome is 248,956,422 bp long, which is shorter than 2^31
    POS = np.zeros(num_loci, dtype=np.int32)
    for l in trange(num_loci, desc='Converting data to NumPy'):
        locus = vcf_records[l]
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

def read_phgeno(filename):
    '''
    Reads an EIGENSTRAT-formatted (.phgeno) file
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_loci = len(lines)
        num_samples = len(lines[0])
        # If any of the lines contain '2', this sample is diploid; otherwise, assume it's haploid
        ploidy = 2 if any('2' in line for line in lines) else 1
        X = np.zeros((num_loci, num_samples, ploidy), dtype=np.uint8)
        J = np.full(num_loci, 2, dtype=np.uint8)
        POS = np.arange(0, num_loci * 10000, 10000, dtype=np.int32)
        # Generate random alleles for missing data
        rng = np.random.default_rng()
        for l, row in enumerate(lines):
            for i, x in enumerate(map(int, row.strip())):
                if x == 9:
                    # If missing data (9), make it up
                    X[l, i] = rng.integers(2, size=ploidy)
                else:
                    # x == 0, 1, 2 means that 0, 1, or 2 alleles are positive
                    X[l, i, :x] = 1
        return X, J, POS

def read_file(filename, drop_frac=0.6):
    '''
    Determine the file format and read it in
    '''
    _, ext = os.path.splitext(filename)
    if ext == '.vcf':
        return read_vcf(filename, drop_frac)
    elif ext == '.phgeno':
        return read_phgeno(filename)
    else:
        raise ValueError('File extension must be .vcf or .phgeno')

def test_read_file(filename):
    X, J, POS = read_file(filename)
    print(X.shape, J.shape, POS.shape)

def hdf5_to_numpy(hdf5_dset: h5py.Dataset):
    array = np.empty(hdf5_dset.shape, hdf5_dset.dtype)
    hdf5_dset.read_direct(array)
    return array

def subst_file_suffix(filename, new_ext):
    root, _ = os.path.splitext(filename)
    return os.path.basename(root) + new_ext

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read a VCF file and parse it into the internal NumPy representation')
    parser.add_argument('path', metavar='The path to the VCF file to be read')
    args = parser.parse_args()
    test_read_file(args.path)

