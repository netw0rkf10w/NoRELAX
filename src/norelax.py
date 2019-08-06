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
from scipy.sparse import csr_matrix, lil_matrix

from simplex_projection import unit_simplex_projection_rowwise

class MRF():
    """
    Graph class
    """
    def __init__(self, num_nodes=0, num_labels=0, directed=False, sparse=True):
        self.num_nodes = num_nodes
        self.num_labels = num_labels
        self.is_directed = directed
        self.is_sparse = sparse
        # self.num_edges = 0
        self.edges = []
        # neighbors[i] is the set of neighbors of i
        self.neighbors = [set() for i in range(num_nodes)]
        self.dim = int(num_nodes*num_labels)
        self.nodePot = np.zeros(self.dim)
        if sparse:
            # self.edgePot = csr_matrix((self.dim, self.dim))
            self.edgePot = lil_matrix((self.dim, self.dim))
        else:
            self.edgePot = np.zeros((self.dim, self.dim))
    
    def set_node_potentials(self, pvector):
        """
        Set the node potentials for the MRF
        pvector: vector to dimension num_nodes*num_labels
        """
        assert len(pvector) == self.dim
        # for ia in range(len(pvector)):
        #     if pvector[ia] != 0:
        #         self.pairwisePot[ia, ia] = pvector[ia]
        self.nodePot = pvector

    def add_edge(self, e, pmatrix):
        """
        e: a tuple of nodes (i, j)
        pmatrix: potential matrix of dimension num_labels*num_labels
            pmatrix[a, b] is the cost of assigning i to label a and j to label b
        """
        (i, j) = e
        assert i <= (self.num_nodes - 1) and i >= 0
        assert j <= (self.num_nodes - 1) and j >= 0
        L = self.num_labels
        assert pmatrix.shape == (L, L)
        if e in self.edges or (not self.is_directed and (j, i) in self.edges):
            print('The edge', e, 'already exists.')
        else:
            self.edges.append((i, j))
            self.neighbors[i].add(j)
            self.neighbors[j].add(i)
            for a in range(L):
                for b in range(L):
                    if pmatrix[a, b] != 0:
                        ia = i*L + a
                        jb = j*L + b
                        self.edgePot[ia, jb] = pmatrix[a, b]

    def energy(self, labels):
        """
        labels: given labeling
        """
        E = 0
        for i in range(self.num_nodes):
            ia = i*self.num_labels + labels[i]
            E += self.nodePot[ia]
        for e in self.edges:
            (i, j) = e
            ia = i*self.num_labels + labels[i]
            jb = j*self.num_labels + labels[j]
            E += self.edgePot[ia, jb]
        return E

    def energy_continuous(self, X):
        """
        X: assignment matrix of dimensions VxL
        """
        x = X.flatten()
        E = np.dot(self.nodePot, x) + np.dot(x, self.edgePot.dot(x))
        return E

    def optimize(self, method='admm', X0=None, **kwargs):
        """
        MRF optimization, or MAP inference
        """
        # If the graph is sparse then convert to csr matrix for better performance
        if self.is_sparse:
            self.edgePot = self.edgePot.tocsr()
        if method == 'admm':
            labels = self.ADMM(X0, **kwargs)
        elif method == 'bcd':
            labels = self.BCD(X0, **kwargs)
        else:
            raise "Method %s not supported!"%method

        return labels

    def BCD(self, X0=None, **kwargs):
        """
        (Continuous-valued) Block Coordinate Descent
        """
        # TODO: implement BCD
        X = X0
        labels = np.argmax(X, axis=1)
        return labels

    def ADMM(self, X0=None, **kwargs):
        """
        MAP inference
        """
        rho_min = float(kwargs.get('rho_min', 0.001))
        rho_max = float(kwargs.get('rho_max', 1e5))
        step = float(kwargs.get('step', 1.2))
        precision = float(kwargs.get('precision', 1e-5))
        decrease_delta = float(kwargs.get('decrease_delta', 1e-3))
        iter1 = int(kwargs.get('iter1', 10))
        iter2 = int(kwargs.get('iter2', 50))
        max_iter = int(kwargs.get('max_iter', 1000))
        verbose = bool(kwargs.get('verbose', True))

        absMax = abs(self.edgePot).max()
        rho_min = rho_min*absMax

        n = self.dim
        L = self.num_labels
        V = self.num_nodes
        P = self.edgePot
        u = self.nodePot

        # Initialization
        if X0 is not None:
            x1 = X0.flatten().copy()
        else:
            x1 = np.zeros(n) + 1.0/L
        x2 = x1.copy()
        y = np.zeros(n)

        rho = rho_min
        iter1_cumulated = iter1
        res_best_so_far = 1e10
        counter = 0
        residual = 1e10
        residuals = []

        for k in range(max_iter):
            # Step 1: update x1 = argmin 0.5||x||^2 - c1^T*x where
            # c1 = x2 - (d + M*x2 + y)/rho
            x1_old = x1
            c1 = x2 - (u + P.dot(x2) + y)/rho
            C1 = c1.reshape((V, L))
            # Simplex projection for each row of C1
            X1 = unit_simplex_projection_rowwise(C1)
            x1 = X1.flatten()

            # Step 2: update x2 = argmin 0.5||x||^2 - c2^T*x where
            # c2 = x1 - (M^T*x1 - y)/rho
            x2_old = x2
            c2 = x1 + (y - P.dot(x1))/rho
            # C2 = c2.reshape((V, L))
            # X2 = SimplexProjection(C2)
            # x2 = X2.flatten()
            x2 = np.maximum(c2, 0)

            # Step 3: update y
            y += rho*(x1 - x2)

            # Step 4: compute the residuals and update rho
            r = np.linalg.norm(x1 - x2)
            s = np.linalg.norm(x1 - x1_old) + np.linalg.norm(x2 - x2_old)

            residual_old = residual
            residual = r + s

            # energy = computeEnergy(x1, d, M)
            # energies.push_back(energy)
            residuals.append(residual)

    #        if(energy < energy_best){
    #            energy_best = energy
    #            x = x1
    #        }

            if verbose:
                print("%d\t residual = %f"%(k+1, residual))

            # If convergence
            if residual <= precision and residual_old <= precision:
                break

            # Only after iter1 iterations that we start to track the best residuals
            if k >= iter1_cumulated:
                if residual < res_best_so_far - decrease_delta:
                    counter = 0
                else:
                    counter += 1
                if residual < res_best_so_far:
                    res_best_so_far = residual
                
                # If the best_so_far residual has not changed during iter2 iterations, then update rho
                if counter >= iter2:
                    if rho < rho_max:
                        rho = min(rho*step, rho_max)
                        if verbose:
                            print('\t UPDATE RHO = %f at iteration %d'%(rho/absMax, k+1))
                        counter = 0
                        iter1_cumulated = k + iter1
                    else:
                        break
        # Continuous energy
        if verbose:
            e_continuous = self.energy_continuous(X1)
            print('Continuous energy = %f'%e_continuous)
            # print('Assignment matrix =', X1)

        labels = self.BCD(X1)
        return labels