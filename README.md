# Boundary-Aware Graph Neural Networks for Compound Flood Forecasting

This repository accompanies the manuscript **"Boundary-Aware Graph Neural Networks for Compound Flood Forecasting"**, prepared for initial submission to *Water Resources Research*.

## Scientific Description

Compound flooding arises from nonlinear interactions among rainfall-driven runoff, coastal water-level forcing, riverine response, and the hydraulic pathways that transfer these forcings through low-lying coastal floodplains. GNN4CF is a boundary-aware graph neural network surrogate designed to predict compound-flood water-depth evolution on unstructured hydraulic meshes while preserving the physical distinction between external boundary forcing and interior floodplain propagation.

The framework represents the hydraulic mesh as a typed graph with separate coastal boundary and interior computational nodes. A dedicated boundary-interior coupling processor transfers coastal-stage information into the floodplain, while a separate interior processor propagates the hydraulic response through the computational domain. Virtual boundary-to-interior edges support long-range coastal signal transfer under finite message-passing depth, and repeated rainfall and coastal-stage conditioning preserves time-varying forcing information during autoregressive rollout.

Across held-out events, GNN4CF reduces RMSE, relative L2 error, and false alarm ratio while improving inundation-detection skill relative to ablated graph models. The model also achieves rapid GPU inference and generalizes from single-active-boundary training events to unseen simultaneous multi-boundary forcing configurations, indicating that learned boundary-response pathways can be recombined under more complex coastal-connectivity states.

## Highlights

- GNN4CF enables rapid, spatially distributed compound-flood forecasting on unstructured hydraulic meshes.
- Explicit boundary-interior coupling improves graph-based surrogate modeling of coastal-inland exchange.
- Separate boundary and interior processors outperform shared message passing for boundary-driven hydraulic response.
- Virtual boundary-to-interior edges enhance long-range coastal propagation under finite message-passing depth.
- Dynamic rainfall and coastal-stage conditioning preserves time-varying forcings during autoregressive rollout.
- The trained model generalizes to unseen simultaneous multi-boundary forcing without additional fine-tuning.

## WRR Key Points

- Boundary-aware graph learning improves compound-flood forecasting by separating coastal exchange from interior propagation.
- Virtual boundary-to-interior edges and dynamic forcing conditioning improve long-range, time-varying flood response.
- GNN4CF generalizes from single-boundary training to unseen multi-boundary coastal-forcing configurations.

## Proposed Framework

![Proposed GNN4CF framework](figures/GNN4CF_framework.png)

## Reviewer Supplementary Animations

Representative rollout GIFs are prepared separately for reviewer inspection because the animation files are large. The set includes both wet-depth thresholds used for sensitivity checking and covers both single-boundary and multi-boundary examples:

| Animation | Configuration | Wet-depth threshold | External link |
| --- | --- | --- | --- |
| `single_boundary_all_models_BC1_S402_R402_WDge10cm.gif` | Single active boundary, all model variants | 0.10 m | To be added after Google Drive upload |
| `single_boundary_all_models_BC1_S402_R402_WDge20cm.gif` | Single active boundary, all model variants | 0.20 m | To be added after Google Drive upload |
| `multiboundary_M4_S476_R476_WDge10cm.gif` | Simultaneous multi-boundary forcing, M4 | 0.10 m | To be added after Google Drive upload |
| `multiboundary_M4_S476_R476_WDge20cm.gif` | Simultaneous multi-boundary forcing, M4 | 0.20 m | To be added after Google Drive upload |

A local copy of these files is organized at:

`C:\CF_GNN\rollout_transfer_bundle_4_2\paper_final_results\reviewer_supplementary_gifs`

See [`supplementary_animations/README.md`](supplementary_animations/README.md) for the reviewer animation manifest and upload notes.

## Citation

If you use this repository, code, figures, or concepts from GNN4CF in your research, please cite the manuscript:

**Zandsalimi, Z.**, Taghizadeh, M., Lee Lynn, S., Goodall, J. L., Shafiee-Jood, M., and Alemazkoor, N. (2026). **Boundary-Aware Graph Neural Networks for Compound Flood Forecasting.** *Water Resources Research*. Under review.

```bibtex
@article{zandsalimi2026gnn4cf,
  title = {Boundary-Aware Graph Neural Networks for Compound Flood Forecasting},
  author = {Zandsalimi, Zanko and Taghizadeh, Mehdi and Lee Lynn, S. and Goodall, Jonathan L. and Shafiee-Jood, Majid and Alemazkoor, Negin},
  journal = {Water Resources Research},
  year = {2026},
  note = {Under review}
}
```

## Repository Status

This repository is being prepared for manuscript review. Additional code, model-configuration files, and reproducibility materials will be added as the submission package is finalized.
