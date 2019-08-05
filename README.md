# NoRELAX: Nonconvex Relaxation Methods for MAP Inference in Discrete Markov Random Fields

This repo aims at re-implementing in Python the algorithms presented in the following paper:

*D. Khuê Lê-Huu and Nikos Paragios. **Continuous Relaxation of MAP Inference: A Nonconvex Perspective**. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.*

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

## Example
Example of usage is given in `example.py`.

## Notes
In the current version:
* Only pairwise (i.e. first-order) MRFs are supported.
* Only ADMM with `argmax` rounding is implemented (which is suboptimal).
* No performance optimization.
