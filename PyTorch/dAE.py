#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np, scipy as sp
import torch
import scipy.io
import matplotlib.pyplot as plt

# from visualdl import LogWriter

import argparse, sys

sys.path.append("../python/")
from demo import toy_data


__author__ = "TengQi Ye"
__copyright__ = "Copyright 2017"
__credits__ = ["TengQi Ye"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "TengQi Ye"
__email__ = "yetengqi@gmail.com"
__status__ = "Research"


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()



class AE(torch.nn.Module):
    """The class implements the basic auto-encoder."""

    def __init__(self, n_input, n_hidden, transfer_function=torch.nn.functional.softplus):
        """The constructor of an auto-encoder.

        Args:
            n_input: `int`, the dimension of input layer.
            n_hidden: `int`, the dimension of hidden layer.
            transfer_function: `Operation`, activation function.
            optimizer: `Operation`, optimizer to train the model.
        """

        super(AE, self).__init__()

        # Hyper-parameter.
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function


        # Weights.
        self.encoder = torch.nn.Linear(self.n_input, self.n_hidden)
        self.decoder = torch.nn.Linear(self.n_hidden, self.n_input)

        # Initialization.
        self.input = None
        self.output = None



    def forward(self, input):
        """Feed-forward.

        Args:
            input: with shape (N, n_input)

        Returns:
            output: with shape (N, n_input)
        """
        self.input = torch.tensor(input).float()

        self.hidden = self.encoder(self.input)
        self.output = self.decoder(self.hidden)

        return self.output




    def loss(self, criterion = mse_loss):

        """Auto-encoder is unsupervised.

        Args:
            criterion: loss function.
        """
        if self.input is None or self.output is None:
            print("The Auto-Encoder is not initialized yet.")
            return

        return criterion(self.input, self.output)



    def tune(self, inputs, max_of_epoch = 10000, verbal = True):
        """Train the anto-encoder.

        Args:
            inputs: with shape (N, n_input). We don't need target label because of unsupervised.
            max_of_epoch: the maximum number of epoch (iterations).
            verbal: print training process.
        """

        # Create optimizer.
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(max_of_epoch):

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Foward, backward, and optimize
            outputs = self(inputs)
            loss = self.loss()
            loss.backward()
            optimizer.step()


            # Print running statistics.
            print("Learning loss is ", loss.item())

        print("Finish training.")


    def show_hidden(self):
        """Plot the points of hidden layer."""
        plt.plot(self.hidden.detach().numpy())
        plt.show()




class dAE(AE):
    """Denoising auto-encoder. Inherit every methods except tuning."""

    def tune(self, inputs, corrupt_type = "trivial", max_of_epoch = 10000, verbal = True):
        """Train the anto-encoder.

        Args:
            inputs: with shape (N, n_input). We don't need target label because of unsupervised.
            corrupt_type: in which way data is pre-processing.
            max_of_epoch: the maximum number of epoch (iterations).
            verbal: print training process.
        """
        if corrupt_type == "trivial":
            inputs = self.trivial_denoising(inputs)

        super(dAE, self).tune(inputs, max_of_epoch, verbal)


    def trivial_denoising(self, inputs, repeat=None):
        """Trivial denoising method. A random feature is whitening.

        Args:
        inputs: original data. Its shape is [n_samples, n_features].
        repeat: how many times to repeat inputs.

        Returns:
        corruption: denoised data. Its shape is [n_samples, n_features].
        """
        n, m = np.shape(inputs)

        if repeat is not None:
            m = repeat

        corruption = np.tile(inputs, [m, 1])

        for i in range(m):
            corruption[i*n : i*n+n, i] = 0

        return corruption


def main():

    m1, m2 = toy_data([])

    ae = AE(42, 3)
    ae.tune(np.concatenate((m1, m2), axis=0).T, 100000)
    ae.show_hidden()
    # print(ae.hidden)



if __name__ == "__main__":
    main()
