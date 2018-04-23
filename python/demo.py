#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np, scipy as sp
import orthAE as ae
import scipy.io
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import CCA

import argparse, sys


__author__ = "TengQi Ye"
__copyright__ = "Copyright 2017"
__credits__ = ["TengQi Ye"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "TengQi Ye"
__email__ = "yetengqi@gmail.com"
__status__ = "Research"



def toy_data(fig_list=[1, 2], save=False):
    """
    Generate toy data for multi-view learning based on the paper <Learning Multiple Views with Orthogonal
Denoising Autoencoders>.

    Params:
        fig_list: the list of index of figures to be plotted.

    Returns:
        t: timestamp
        x: shared latent space
        x_noise: correlated noise
    """

    # Base signals.
    t = np.arange(-1, 1, 0.02)
    x = np.sin(2*np.pi*t) # shared latent space
    x_noise = 0.02*np.sin(3.6*np.pi*t) # correlated noise


    # private latent spaces
    z1 = np.cos(np.pi*np.pi*t)
    z2 = np.cos(5*np.pi*t)

    # Plot Fig.2.(a).
    if 1 in fig_list:

        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, z1, 'g')
        plt.plot(t, x, 'b')
        plt.plot(t, x_noise, 'r')

        plt.subplot(212)
        plt.plot(t, z2, 'g')
        plt.plot(t, x, 'b')
        plt.plot(t, x_noise, 'r')
        plt.show()



    # shared private spaces
    m1 = np.vstack((x, z1))
    m2 = np.vstack((x, z2))

    m1 = np.matmul(np.random.rand(20, 2), m1)
    m2 = np.matmul(np.random.rand(20, 2), m2)


    # add gaussian noise with mean=0, standard deviation=0.01
    m1 = m1 + np.multiply(np.random.randn(*m1.shape), 0.01)
    m2 = m2 + np.multiply(np.random.randn(*m2.shape), 0.01)

    # add correlated noise
    m1 = np.vstack((m1, x_noise))
    m2 = np.vstack((m2, x_noise))


    # Plot Fig.2.(b).
    if 2 in fig_list:

        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, m1.T)

        plt.subplot(212)
        plt.plot(t, m2.T)
        plt.show()


    if save:
        sp.io.savemat('exp1.mat', mdict={'t': t, 'x': x, 'x_noise': x_noise, 'm1': m1, 'm2': m2})


    return m1, m2



def cca(m1, m2, preprocessing=None):
    """
    Use CCA to decompose two views and plot result.

    Params:
        m1, m2: Every column is a example with every row as a feature.
        preprocessing: If None, we don't do pre-processing; if 'orth', we adjust center to 0 and perform PCA.
    """
    # Adjust means to be 0 and perform PCA.
    if preprocessing == "orth":
        # Zero means.
        m1 -= np.mean(m1, axis=0)
        m2 -= np.mean(m2, axis=0)

        # PCA.

    cca = CCA(n_components=3)
    cca.fit(m1.T, m2.T)

    X_c = cca.transform(m1.T)

    fig, ax = plt.subplots()
    ax.set_title('Fig.2.(c)')
    # ax.set_color_cycle(['blue', 'green', 'red'])
    ax.set_prop_cycle('color', ['blue', 'red', 'green'])
    ax.plot(X_c)
    # ax.plot(Y_c)
    plt.show()



def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_frequency', type=int,
                        default=10, help='How often to log results to the console.')




def main(args):

    m1, m2 = toy_data()

    cca(m1, m2, "orth")

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
