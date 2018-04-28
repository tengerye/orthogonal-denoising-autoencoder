#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np, scipy as sp
import torch
import scipy.io
import matplotlib.pyplot as plt


import argparse, sys


__author__ = "TengQi Ye"
__copyright__ = "Copyright 2017"
__credits__ = ["TengQi Ye"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "TengQi Ye"
__email__ = "yetengqi@gmail.com"
__status__ = "Research"


class ToyDataset(Dataset):
    """The toy dataset from our paper."""

    def __init__(self, view1, view2, ):