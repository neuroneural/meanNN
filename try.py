import torch
import numpy as np
import cv2
import ipdb
import scipy

from torch import nn
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from numpy import dot
from numpy.linalg import matrix_rank, inv
from numpy.random import permutation
from scipy.linalg import eigh
from scipy.linalg import norm as mnorm

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


def entropyLoss(X):
    entropy = torch.tensor([0]).type(torch.float32)
    for batch in X:
        d = batch.shape[0]
        n = batch.shape[1]
        squared_norms = (batch**2).sum(0).repeat(n, 1)
        squared_norms_T = squared_norms.T
        batch_T = batch.T
        arg = squared_norms + squared_norms_T - 2 * torch.mm(batch_T, batch)
        expression = torch.sum(torch.log(torch.abs(arg)+ torch.eye(n, n) + 1e-12))/2
        entropy += d/(n*(n-1)) * expression
    return entropy/X.size()[0]


a = torch.rand(2, 2, 10000, dtype=torch.float32)
b = torch.randn(2, 2, 10000)

print(entropyLoss(a), entropyLoss(b))