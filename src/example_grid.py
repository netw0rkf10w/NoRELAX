import numpy as np
import time

from norelax import MRF

np.random.seed(12)

def main():
    # Initialize the MRF with the number of nodes/labels
    # By default, sparse=True.
    w = 50
    h = 50
    V = w*h # number of nodes
    L = 5 # number of labels per node
    mrf = MRF(num_nodes=V, num_labels=L, sparse=True)

    # Set the node potentials, which is the flatten vector
    # of a VxL matrix (each row corresponds to a node)
    node_potentials = np.random.rand(V, L).flatten()
    mrf.set_node_potentials(pvector=node_potentials)

    # Add edges to the graph, with corresponding potentials

    # Potts model: f(a, a) = 0 and f(a, b) = 0.5 if a != b
    # pmatrix = np.zeros((L, L)) + 0.1
    # np.fill_diagonal(pmatrix, 0)

    # inverse Potts model: f(a, a) = 0.5 and f(a, b) = 0 if a != b
    pmatrix = np.diag(np.zeros(L) + 0.5)
    print('Adding edge potentials')
    start = time.time()
    for y in range(h):
        for x in range(w):
            i = y*w + x
            j_right = None
            j_bottom = None
            if x + 1 < w:
                j_right = i + 1
            if y + 1 < h:
                j_bottom = (y + 1)*w + x
            if j_right is not None:
                mrf.add_edge((i, j_right), pmatrix)
            if j_bottom is not None:
                mrf.add_edge((i, j_bottom), pmatrix)
    print('Took', time.time() - start, '(s)')

    # print('Compiling the model')
    # start = time.time()
    mrf.compile()
    # print('Took', time.time() - start, '(s)')

    # print('node potentials =', mrf.nodePot)
    # print('edge potentials =', mrf.edgePot)
    # return

    # Run MAP inference with ADMM
    # These parameters are optional, but they can affect greatly the
    # running time as well as the solution quality (c.f. paper)
    kwargs = {"rho_min": 0.1,
              "step": 1.2,
              "precision": 1e-5,
              "iter1": 5,
              "iter2": 10,
              "max_iter": 1e4,
              "verbose": True
             }
    labels = mrf.optimize(method='admm', **kwargs)
    print('labels =', labels)

    E = mrf.energy(labels)
    print('Discrete energy = %f'%E)


if __name__ == "__main__":
    main()