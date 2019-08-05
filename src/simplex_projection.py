"""
NoRELAX: Nonconvex Relaxation for Markov Random Fields Optimization

Copyright D. Khue Le-Huu
https://khue.fr

If you use any part of this code, please cite the following paper.
 
D. Khuê Lê-Huu and Nikos Paragios. Continuous Relaxation of MAP Inference: A Nonconvex Perspective.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

BibTeX:

@inproceedings{lehuu2018norelax,
  title={Continuous Relaxation of MAP Inference: A Nonconvex Perspective},
  author={L{\^e}-Huu, D. Khu{\^e} and Paragios, Nikos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}

This file is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY, without even the implied 
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
import numpy as np

def unit_simplex_projection(c):
    """
    return a solution to: min ||x - c||_2^2 s.t. dot(1, x) = 1 and x >= 0
    """
    n = len(c)
    a = -np.sort(-c)
    lambdas = (np.cumsum(a) - 1)/np.arange(1, n+1)
    for k in range(n-1, -1, -1):
        if a[k] > lambdas[k]:
            return np.maximum(c - lambdas[k], 0)

def unit_simplex_projection_rowwise(C):
    """
    Doing simplex projection for each row of C
    """
    X = np.zeros(C.shape)
    for i in range(C.shape[0]):
        X[i] = unit_simplex_projection(C[i])
    return X