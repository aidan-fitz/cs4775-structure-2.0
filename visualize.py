import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from preprocess import hdf5_to_numpy, subst_file_suffix
import os

import argparse

def plot_Q(h5file: h5py.File, filename: str):
    '''
    Create histograms of the population ancestry proportions (Q) similar to Figure 1
    from Pritchard et al. (2000).
    '''
    Q = hdf5_to_numpy(h5file['Q'])
    K = hdf5_to_numpy(h5file['K']).item()
    fig, axs = plt.subplots(K, 1, constrained_layout=True)
    fig.suptitle('Inferred proportion of ancestry from each population')
    for k in range(K):
        axs[k].hist(Q[:, k], bins=100)
        axs[k].set_xlabel(f'Proportion from population {k}')
        axs[k].set_ylabel('Number of individuals')
    out = subst_file_suffix(filename, '_Q.png')
    print(f'Writing plot to {out}')
    plt.savefig(out)

def plot_QZ(h5file: h5py.File, filename: str):
    '''
    Create scatter plots showing the proportions of ancestry & alleles in each population
    similar to Figure 2 from Pritchard et al. (2000).
    '''
    Q = hdf5_to_numpy(h5file['Q'])
    K = hdf5_to_numpy(h5file['K']).item()
    Z = hdf5_to_numpy(h5file['Z'])
    num_loci, _, ploidy = Z.shape
    fig, axs = plt.subplots(1, K, constrained_layout=True)
    for k in range(K):
        frac_ancestry = Q[:, k]
        frac_alleles = np.count_nonzero(Z == k, axis=(0, 2)) / (num_loci * ploidy)
        axs[k].scatter(frac_ancestry, frac_alleles)
        axs[k].set_xlabel(f'Inferred proportion of ancestry from population {k}')
        axs[k].set_ylabel('Proportion of alleles from population {k}')
    out = subst_file_suffix(filename, '_QZ.png')
    print(f'Writing plot to {out}')
    plt.savefig(out)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize program output')
    parser.add_argument('file', metavar='The data file (*.hdf5)')
    return parser.parse_args()

def main():
    args = parse_args()
    with h5py.File(args.file, 'r') as h5file:
        plot_Q(h5file, args.file)
        plot_QZ(h5file, args.file)

if __name__ == '__main__':
    main()
