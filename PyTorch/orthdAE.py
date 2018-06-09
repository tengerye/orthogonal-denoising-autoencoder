#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np, scipy as sp
import torch
from torch.nn.parameter import Parameter
import scipy.io
import matplotlib.pyplot as plt

# from visualdl import LogWriter

import argparse, sys

sys.path.append("../python/")
from demo import toy_data
from dAE import dAE


__author__ = "TengQi Ye"
__copyright__ = "Copyright 2017"
__credits__ = ["TengQi Ye"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "TengQi Ye"
__email__ = "yetengqi@gmail.com"
__status__ = "Research"


class orthdAE(dAE):
    """The class implements orthogonal denoising auto-encoder."""


    def __init__(self, array_n_input, array_n_hidden, transfer_function=torch.nn.functional.softplus):
        """Initialization.

        Args:
        array_n_input: an array or list indicating the number of dimensions of each view.
        array_n_hidden: an array or list indicating the number of dimensions of spaces,
        the last one corresponds to private space.
        transfer_function: activation function.
        """

        super(dAE, self).__init__(np.sum(array_n_input), np.sum(array_n_hidden), transfer_function=torch.nn.functional.softplus)

        self.array_n_input = array_n_input
        self.array_n_hidden = array_n_hidden
        self.transfer = transfer_function

        # Weights. They are auto initialized.
        self.encoder = Parameter(torch.Tensor(np.sum(self.array_n_hidden), np.sum(self.array_n_input) + 1))
        self.decoder = Parameter(torch.Tensor(np.sum(self.array_n_input), np.sum(self.array_n_hidden) + 1))

        # Masks. Note we have to manually set bias.
        # For the encoder.
        encoder_mask = np.zeros((np.sum(self.array_n_hidden), np.sum(self.array_n_input) + 1))

        array_idx_hidden = np.cumsum(self.array_n_hidden)
        array_idx_input = np.cumsum(self.array_n_input)

        np.insert(array_idx_hidden, 0, 0)
        np.insert(array_idx_input, 0, 0)

        encoder_mask[0 : array_idx_hidden[1], :] = 1  # Shared space connects all spaces from input layer.
        for idx in range(1, array_idx_input):
            encoder_mask[array_idx_hidden[idx-1]:array_idx_hidden[idx], array_idx_input[idx]:array_idx_input[idx+1]] = 1 # Private space connects itself.

        self.encoder = self.encoder * torch.from_numpy(np.transpose(encoder_mask))


        # For the decoder.
        decoder_mask = np.zeros((np.sum(self.array_n_input), np.sum(self.array_n_hidden) + 1))

        decoder_mask[:, -1] = 1  # Shared space connects all spaces from input layer.
        for idx in range(1, array_idx_input):
            decoder_mask[array_idx_input[idx]:array_idx_input[idx+1], array_idx_hidden[idx-1]:array_idx_hidden[idx]] = 1 # Private space connects itself.

        self.encoder = self.encoder * torch.from_numpy(np.transpose(decoder_mask))



    def forward(self, x):
        self.hidden = self.transfer(self.encoder())
        return x