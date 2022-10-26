import numpy as np
from numpy import dot
from numpy.linalg import matrix_rank, inv
from numpy.random import permutation
from scipy.linalg import eigh #use this when you know your matrix is symmetric
from scipy.linalg import norm as mnorm

def diagsqrts(w):
    """
    Returns direct and inverse square root normalization matrices
    """
    Di = np.diag(1. / (np.sqrt(w) + np.finfo(float).eps))
    D = np.diag(np.sqrt(w))
    return D, Di

def pca_whiten(x2d, n_comp, verbose=True):
    """ data Whitening
    *Input
    x2d : 2d data matrix of observations by variables
    n_comp: Number of components to retain
    *Output
    Xwhite : Whitened X
    white : whitening matrix (Xwhite = np.dot(white,X))
    dewhite : dewhitening matrix (X = np.dot(dewhite,Xwhite))
    """
    x2d_demean = x2d - x2d.mean(axis=1).reshape((-1, 1))
    NSUB, NVOX = x2d_demean.shape
    if NSUB > NVOX:
        cov = np.dot(x2d_demean.T, x2d_demean) / (NSUB - 1)
        w, v = eigh(cov, eigvals=(NVOX - n_comp, NVOX - 1))
        D, Di = diagsqrts(w)
        u = dot(dot(x2d_demean, v), Di)
        x_white = v.T
        white = dot(Di, u.T)
        dewhite = dot(u, D)
    else:
        cov = np.dot(x2d_demean, x2d_demean.T) / (NVOX - 1)
        w, u = eigh(cov, eigvals=(NSUB - n_comp, NSUB - 1))
        D, Di = diagsqrts(w)
        white = dot(Di, u.T)
        x_white = dot(white, x2d_demean)
        dewhite = dot(u, D)
    return (x_white, white, dewhite)

# class whitening:
#     def diagsqrts(self, w):
#         """
#         Returns direct and inverse square root normalization matrices
#         """
#         Di = np.diag(1. / (np.sqrt(w) + np.finfo(float).eps))
#         D = np.diag(np.sqrt(w))
#         return D, Di

#     def pca_whiten(self, x2d, n_comp, verbose=True):
#         """ data Whitening
#         *Input
#         x2d : 2d data matrix of observations by variables
#         n_comp: Number of components to retain
#         *Output
#         Xwhite : Whitened X
#         white : whitening matrix (Xwhite = np.dot(white,X))
#         dewhite : dewhitening matrix (X = np.dot(dewhite,Xwhite))
#         """
#         x2d_demean = x2d - x2d.mean(axis=1).reshape((-1, 1))
#         NSUB, NVOX = x2d_demean.shape
#         if NSUB > NVOX:
#             cov = np.dot(x2d_demean.T, x2d_demean) / (NSUB - 1)
#             w, v = eigh(cov, eigvals=(NVOX - n_comp, NVOX - 1))
#             D, Di = self.diagsqrts(w)
#             u = dot(dot(x2d_demean, v), Di)
#             x_white = v.T
#             white = dot(Di, u.T)
#             dewhite = dot(u, D)
#         else:
#             cov = np.dot(x2d_demean, x2d_demean.T) / (NVOX - 1)
#             w, u = eigh(cov, eigvals=(NSUB - n_comp, NSUB - 1))
#             D, Di = self.diagsqrts(w)
#             white = dot(Di, u.T)
#             x_white = dot(white, x2d_demean)
#             dewhite = dot(u, D)
#         return (x_white, white, dewhite)