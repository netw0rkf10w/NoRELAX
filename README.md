# NoRELAX: Nonconvex Relaxation Methods for MAP Inference in Discrete Markov Random Fields

This repo aims at re-implementing in Python the algorithms presented in the following paper:

*D. Khuê Lê-Huu and Nikos Paragios. **Continuous Relaxation of MAP Inference: A Nonconvex Perspective**.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.*

If you use any part of this code, please cite the above paper.

BibTeX:
```
@inproceedings{lehuu2018norelax,
  title={Continuous Relaxation of MAP Inference: A Nonconvex Perspective},
  author={L{\^e}-Huu, D. Khu{\^e} and Paragios, Nikos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## Usage
### Initialization
An MRF can be initialized with a pre-defined number of nodes and number of labels per node (different numbers of labels are currently not supported):
```
from norelax import MRF
V = 4 # number of nodes
L = 3 # number of labels per node
mrf = MRF(num_nodes=V, num_labels=L, sparse=True)
```
If one knows in advance the sparseness of the graph then it is beneficial to set the `sparse` argument (default to `True`).

### Adding node (or *unary*) potentials
This can be done simply as
```
mrf.set_node_potentials(pvector=node_potentials)
```
where `node_potentials` is a 1D vector of length `V*L`, which can be seen as the *flatten* version of a 2D potential matrix of dimensions `V x L` where each row corresponds to a node. I used a vector instead of a matrix because this would allow an easier extension to the case where the nodes have different numbers of labels.

### Adding edge (or *pairwise*) potentials
An edge `(i, j)` with its potential matrix `edge_potentials_ij` (of dimensions `L x L`) can be added to the model using
```
mrf.add_edge((i, j), pmatrix=edge_potentials_ij)
```

### Perform MAP inference
Finally, MAP inference can be performed using the `optimize()` member function that returns the optimal labeling:
```
labels = mrf.optimize(method='admm', **kwargs)
```
The variable `**kwargs` contains optional parameters for `method`. For example, the default parameters for ADMM are:
```
kwargs = {"rho_min": 0.01,
          "rho_max": 1000,
          "step": 1.2,
          "precision": 1e-5,
          "iter1": 10,
          "iter2": 50,
          "max_iter": 10000,
          "verbose": True
          }
```
**Notes:** these parameters affects greatly the convergence as well as the solution quality of ADMM.
* For faster convergence: increase `rho_min` or (preferably) decrease `iter2`.
* For better (lower) energy: decrease `rho_min` and/or increase `iter2`.

See paper (Section 4.2).

The final discrete energy can be computed using:
```
E = mrf.energy(labels)
print('Discrete energy = %f'%E)
```

An complete example of usage is given in `example.py`. Run `python example.py` to see the results.

## Notes
In the current version:
* Only pairwise (i.e. first-order) MRFs are supported.
* Only ADMM with `argmax` rounding is implemented (which is suboptimal). The other methods and Block Coordinate Descent rounding will be added later.
* No performance optimization.
