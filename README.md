# Physics-Informed Vessel Trajectory Prediction via Finite Difference Kinematic Losses

This repository implements physics-informed neural networks (PINNs) for vessel trajectory prediction. We augment sequence models with finite-difference kinematic loss terms that enforce kinematic constraints by penalizing discrepancies in estimated velocity and acceleration computed from predicted positions. The approach encourages smoother, physically plausible vessel tracks, reduces unrealistic jumps, and improves long-horizon stability compared to purely data-driven baselines.

Citation

```bibtex
@inproceedings{alam2025physics,
  title={Physics-informed neural networks for vessel trajectory prediction: Learning time-discretized kinematic dynamics via finite differences},
  author={Alam, Md Mahbub and Soares, Amilcar and Rodrigues-Jr, Jos{\'e} Fernando and Spadon, Gabriel},
  booktitle={Proceedings of the 19th International Symposium on Spatial and Temporal Data},
  pages={55--65},
  year={2025}
}
```

```bibtex
@article{10.21203/rs.3.rs-8291452/v1,
  author = {Alam, M. M. and Soares, A. and Rodrigues, J. F. and Spadon, G.},
  title = {Physics-Informed Vessel Trajectory Prediction via Finite Difference Kinematic Losses},
  year = {2025},
  doi = {10.21203/rs.3.rs-8291452/v1}
}
```
