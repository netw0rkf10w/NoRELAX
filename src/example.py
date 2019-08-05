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
from norelax import MRF

def main():
    # Initialize the MRF with the number of nodes/labels
    # By default, sparse=True.
    V = 4
    L = 3
    mrf = MRF(num_nodes=V, num_labels=L, sparse=True)

    # Set the node potentials, which is the flatten vector
    # of a VxL matrix (each row corresponds to a node)
    node_potentials = np.array([2.0, 0.5, 1.5,  # node 0
                                1.2, 0.0, 8.5,  # node 1
                                0.2, 5.0, 3.5,  # node 2
                                1.1, 1.0, 2.5]) # node 3
    mrf.set_node_potentials(pvector=node_potentials)

    # Add edges to the graph, with corresponding potentials

    # Potts model: f(a, a) = 0 and f(a, b) = 0.5 if a != b
    pmatrix = np.zeros((L, L)) + 0.5
    np.fill_diagonal(pmatrix, 0)

    # inverse Potts model: f(a, a) = 0.5 and f(a, b) = 0 if a != b
    # pmatrix = np.diag(np.zeros(L) + 0.5)

    mrf.add_edge((0, 1), pmatrix)
    mrf.add_edge((2, 3), pmatrix)
    mrf.add_edge((0, 3), pmatrix)
    mrf.add_edge((1, 2), pmatrix)

    # Run MAP inference with ADMM
    # These parameters are optional, but they can affect greatly the
    # running time as well as the solution quality (c.f. paper)
    kwargs = {"rho_min": 0.01,
              "step": 1.2,
              "precision": 1e-5,
              "iter1": 10,
              "iter2": 50,
              "max_iter": 1e4,
              "verbose": True
             }
    labels = mrf.optimize(method='admm', **kwargs)
    print('labels =', labels)

if __name__ == "__main__":
    main()
