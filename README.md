# GNN4CF

GNN4CF is a boundary-aware graph neural network surrogate for autoregressive
compound-flood forecasting under rainfall and coastal water-level forcing on
an unstructured hydraulic mesh.

The model uses typed computational and coastal boundary ghost nodes, separate
boundary-interior coupling and interior processors, physical and virtual
boundary connections, boundary conditioning, rainfall conditioning, and
autoregressive water-depth prediction.

![GNN4CF framework](figures/GNN4CF_framework.png)

## Workflow

```text
HEC-RAS simulations + DEM + scalar mesh attributes
        |
compute_terrain_features.py
        |
gnn4cf_hdf_graph_dataset.py
        |
train_gnn4cf.py
        |
prepare_gnn4cf_test_graphs.py
        |
run_gnn4cf_rollout_test.py
```

| Script in `src/` | Role |
| --- | --- |
| `compute_terrain_features.py` | Computes GIS/terrain attributes sampled at computational cells. |
| `gnn4cf_hdf_graph_dataset.py` | Converts event outputs into graph-ready HDF5 datasets; selects interior representatives and adds virtual boundary-interior edges. |
| `train_gnn4cf.py` | Trains from grouped event splits with teacher-forced and pushforward stability branches. |
| `prepare_gnn4cf_test_graphs.py` | Caches initial test graphs and future forcing sequences. |
| `run_gnn4cf_rollout_test.py` | Loads a checkpoint and evaluates autoregressive test-event rollouts. |
| `gnn4cf_graph_builder.py` | Assembles mesh connectivity, boundary nodes, features, normalization, and temporal snapshots. |
| `gnn4cf_model.py` | Defines typed encoders, coupling/interior processors, conditioning, and the water-depth decoder. |
| `gnn4cf_training_utils.py` | Provides losses, pushforward training, validation metrics, and rollout utilities. |

The graph builder and HDF dataset module are complementary libraries, not
alternative dataset versions.

## Configuration And Execution

Update all placeholder paths in `configs/config_gnn4cf_final.yml`, the HEC-RAS
area name, and the grouped event-split manifest before running. Supply existing
event HDF files, a DEM in the mesh coordinate system, scalar cell attributes
(`cell_id,zmin,zmax,relief`), and face geometry/connectivity attributes.
Terrain extraction supplies the GIS table, not these scalar mesh tables.
The manifest must contain the configured group and split columns.
GIS feature selectors preserve source column order; keep it consistent when
reusing normalization statistics and checkpoints.

`config_name` organizes checkpoints, logs, rollout outputs, and optional W&B
artifacts under `paths.runs_root`. It is not a model or scientific parameter.
Use a distinct name per experiment and retain it when resuming the same run.

Dependencies include PyTorch, PyTorch Geometric, NumPy, pandas, h5py, PyYAML,
matplotlib, tqdm, rasterio, WhiteboxTools/`whitebox`, and
`wandb` (imported by the trainer even when tracking is disabled).
The training launcher currently requires POSIX signal handling (`SIGUSR1`).

Run from the repository root:
```bash
python src/compute_terrain_features.py --config configs/config_gnn4cf_final.yml
python src/gnn4cf_hdf_graph_dataset.py --config configs/config_gnn4cf_final.yml
python src/train_gnn4cf.py --config configs/config_gnn4cf_final.yml
python src/prepare_gnn4cf_test_graphs.py --config configs/config_gnn4cf_final.yml --no-interactive
python src/run_gnn4cf_rollout_test.py --config configs/config_gnn4cf_final.yml
```

## Manuscript

Zandsalimi, Z., Taghizadeh, M., Shafiee-Jood, M., and Alemazkoor, N. (2026).
*Boundary-Aware Graph Neural Networks for Compound Flood Forecasting.*
Water Resources Research, under review.

[Reviewer supplementary animations](https://drive.google.com/drive/folders/1HcZCrWNIrm8h0ekQw5B9Z_MB6RL6o7XS?usp=sharing)
