# -*- coding: utf-8 -*-

"""
Create HDF5 graph datasets for GNN4CF training and rollout inference.

Store each event's shared static graph tensors and indexed dynamic windows
for lazy PyTorch Geometric reconstruction. Graph preprocessing includes
boundary-line assignment, representative-node selection, coverage checks,
and virtual boundary-interior edges. Feature assembly is provided by
gnn4cf_graph_builder.py.
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

import gnn4cf_graph_builder as base


def _wd_norm_to_dir_label(wd_norm):
    """Convert config normalization names into folder-friendly labels."""
    if wd_norm == "log1p_zscore":
        return "logp1_zscore"
    return str(wd_norm)


def resolve_hdf_graph_paths(cfg):
    """
    Build the graph-configuration folder and its standard subpaths from config.

    Expected layout:
      <graphs_hdf_root>/<predictor_step>Step_predictor_<wd_norm>/
        graph_snapshots_hdf/
        test_snapshots/
        train_val_snapshots/
        initial_test_graph/
        normalization_stats.json
    """
    paths_cfg = cfg.get("paths", {})
    window_cfg = cfg.get("window", {})
    norm_cfg = cfg.get("normalization", {})

    graphs_root = paths_cfg.get("graphs_hdf_root", os.path.join(paths_cfg.get("output_dir", "output_data"), "Graphs_HDF"))
    predictor_step = window_cfg.get("predictor_step")
    if predictor_step is None:
        raise KeyError("window.predictor_step must be defined in config.yml")

    wd_norm = (
        norm_cfg.get("feature_methods", {})
        .get("dynamic", {})
        .get("wd", norm_cfg.get("default_methods", {}).get("dynamic", "zscore"))
    )
    wd_norm_label = _wd_norm_to_dir_label(wd_norm)

    folder_template = paths_cfg.get("graph_config_dir_template", "{predictor_step}Step_predictor_{wd_norm}")
    graph_dir_name = folder_template.format(
        predictor_step=int(predictor_step),
        wd_norm=wd_norm_label,
    )
    graph_root = os.path.join(graphs_root, graph_dir_name)

    resolved = {
        "graph_root": graph_root,
        "graph_dir_name": graph_dir_name,
        "wd_norm_label": wd_norm_label,
        "snapshot_hdf_dir": os.path.join(graph_root, paths_cfg.get("snapshot_hdf_dir_name", "graph_snapshots_hdf")),
        "test_snapshot_dir": os.path.join(graph_root, paths_cfg.get("test_snapshot_dir_name", "test_snapshots")),
        "train_val_snapshot_dir": os.path.join(graph_root, paths_cfg.get("train_val_snapshot_dir_name", "train_val_snapshots")),
        "initial_test_graph_dir": os.path.join(graph_root, paths_cfg.get("initial_test_graph_dir_name", "initial_test_graph")),
        "normalization_stats_path": os.path.join(
            graph_root,
            paths_cfg.get("normalization_stats_filename", "normalization_stats.json"),
        ),
    }
    resolved["snapshot_dir"] = resolved["train_val_snapshot_dir"]
    resolved["train_val_cache_dir"] = resolved["train_val_snapshot_dir"]
    return resolved


def apply_hdf_graph_paths_to_config(cfg):
    """Resolve graph paths and store them back into `cfg['paths']` for downstream consumers."""
    resolved = resolve_hdf_graph_paths(cfg)
    cfg.setdefault("paths", {})
    cfg["paths"]["graph_root"] = resolved["graph_root"]
    cfg["paths"]["snapshot_hdf_dir"] = resolved["snapshot_hdf_dir"]
    cfg["paths"]["test_snapshot_dir"] = resolved["test_snapshot_dir"]
    cfg["paths"]["train_val_snapshot_dir"] = resolved["train_val_snapshot_dir"]
    cfg["paths"]["train_val_snapshots_dir"] = resolved["train_val_snapshot_dir"]
    cfg["paths"]["snapshot_dir"] = resolved["snapshot_dir"]
    cfg["paths"]["train_val_cache_dir"] = resolved["train_val_cache_dir"]
    cfg["paths"]["initial_test_graph_dir"] = resolved["initial_test_graph_dir"]
    cfg["paths"]["normalization_stats_path"] = resolved["normalization_stats_path"]
    return resolved


def ensure_hdf_graph_dirs(graph_paths):
    """Create the graph root and all standard subdirectories."""
    dirs_to_create = [
        graph_paths["graph_root"],
        graph_paths["snapshot_hdf_dir"],
        graph_paths["test_snapshot_dir"],
        graph_paths["train_val_snapshot_dir"],
        graph_paths["initial_test_graph_dir"],
    ]
    for path in dirs_to_create:
        os.makedirs(path, exist_ok=True)


def save_runtime_config_copy(cfg, graph_paths):
    """Save the effective config used for graph generation into the graph root."""
    config_copy_path = os.path.join(graph_paths["graph_root"], "config_used.yml")
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return config_copy_path


def _infer_time_and_nodes(raw_ts, candidate_keys):
    """Infer `(T, N)` from the first available time-series variable."""
    first_key = next((k for k in candidate_keys if k in raw_ts), next(iter(raw_ts)) if raw_ts else None)
    if first_key is None:
        raise ValueError("raw_ts is empty or no valid keys were found.")

    time_steps, num_nodes = raw_ts[first_key].shape
    return time_steps, num_nodes


def _stack_dynamic_vars(raw_ts, keys, time_steps, num_nodes):
    """Stack selected time-varying variables as `[T, N, F]` float32 arrays."""
    available_keys = [key for key in keys if key in raw_ts]
    if not available_keys:
        return np.zeros((time_steps, num_nodes, 0), dtype=np.float32), []

    stacked = np.stack([raw_ts[key] for key in available_keys], axis=-1).astype(np.float32, copy=False)
    return stacked, available_keys


def _build_window_tensor(sequence_array, start_indices, window_length):
    """
    Build a stacked window tensor from `[T, N, F]` to `[S, N, W, F]`.

    `S` is the number of valid snapshots/windows and `W` is the window length.
    """
    if sequence_array.ndim != 3:
        raise ValueError(f"Expected a 3D array [T, N, F], got shape {sequence_array.shape}")

    num_snapshots = len(start_indices)
    _, num_nodes, num_features = sequence_array.shape
    windows = np.empty((num_snapshots, num_nodes, window_length, num_features), dtype=np.float32)

    for out_idx, start_idx in enumerate(tqdm(start_indices, desc="  Building HDF windows", leave=False)):
        window = sequence_array[start_idx:start_idx + window_length]
        windows[out_idx] = np.transpose(window, (1, 0, 2))

    return windows


def create_graph_snapshot_payload_hdf(
    nodes_static_df,
    edges_df,
    edges_static_df,
    raw_ts,
    cfg,
    node_type_tensor,
    edge_type_tensor,
):
    """
    Create an event payload suitable for HDF5 storage.

    The payload stores the shared static graph once and the dynamic windows in
    indexed arrays. A loader can reconstruct a single `Data` sample on demand.
    """
    print("Creating HDF5 graph snapshot payload...")
    features_cfg = cfg.get("features", {})
    window_cfg = cfg.get("window", {})
    past_steps = window_cfg["past_steps"]
    future_steps = window_cfg["future_steps"]

    label_vars = window_cfg.get("label_vars", [])
    aux_vars = window_cfg.get("aux_vars", [])
    state_vars = features_cfg.get("dynamic_input_state_variables", [])
    driver_vars = features_cfg.get("dynamic_input_drivers", [])
    input_vars = state_vars + driver_vars

    time_steps, num_nodes = _infer_time_and_nodes(raw_ts, label_vars + input_vars + aux_vars)
    num_snapshots = time_steps - past_steps - future_steps + 1
    if num_snapshots <= 0:
        print("WARNING: Not enough time steps to create snapshots. Returning empty payload.")
        return None

    x_static = nodes_static_df.drop(columns=["node_id"]).to_numpy(dtype=np.float32, copy=True)
    edge_attr_static = edges_static_df.drop(columns=["src", "dst", "face_id"]).to_numpy(dtype=np.float32, copy=True)
    edge_index = edges_df[["src", "dst"]].to_numpy(dtype=np.int64, copy=True).T

    dynamic_inputs, available_input_vars = _stack_dynamic_vars(raw_ts, input_vars, time_steps, num_nodes)
    dynamic_labels, available_label_vars = _stack_dynamic_vars(raw_ts, label_vars, time_steps, num_nodes)
    future_drivers_stack, available_driver_vars = _stack_dynamic_vars(raw_ts, driver_vars, time_steps, num_nodes)
    dynamic_aux, available_aux_vars = _stack_dynamic_vars(raw_ts, aux_vars, time_steps, num_nodes)

    start_indices = list(range(num_snapshots))
    label_start_indices = [start_idx + past_steps for start_idx in start_indices]

    x_dynamic = _build_window_tensor(dynamic_inputs, start_indices, past_steps)
    y = _build_window_tensor(dynamic_labels, label_start_indices, future_steps)
    future_drivers = _build_window_tensor(future_drivers_stack, label_start_indices, future_steps)

    payload = {
        "static": {
            "x_static": x_static,
            "edge_index": edge_index,
            "edge_attr": edge_attr_static,
            "node_type": node_type_tensor.cpu().numpy().astype(np.int64, copy=False),
            "edge_type": edge_type_tensor.cpu().numpy().astype(np.int64, copy=False),
        },
        "windows": {
            "x_dynamic": x_dynamic,
            "y": y,
            "future_drivers": future_drivers,
            "time_index": np.asarray(start_indices, dtype=np.int64),
        },
        "metadata": {
            "past_steps": int(past_steps),
            "future_steps": int(future_steps),
            "num_snapshots": int(num_snapshots),
            "num_nodes": int(num_nodes),
            "num_edges": int(edge_index.shape[1]),
            "available_input_vars": available_input_vars,
            "available_label_vars": available_label_vars,
            "available_driver_vars": available_driver_vars,
            "available_aux_vars": available_aux_vars,
        },
    }

    if available_aux_vars:
        payload["windows"]["x_aux"] = _build_window_tensor(dynamic_aux, start_indices, past_steps)
        payload["windows"]["y_aux"] = _build_window_tensor(dynamic_aux, label_start_indices, future_steps)

    return payload


def _create_hdf_dataset(group, name, array, chunk_on_first_dim=False):
    """Create an HDF5 dataset with light compression."""
    kwargs = {}
    if isinstance(array, np.ndarray) and array.size > 0:
        kwargs["compression"] = "gzip"
        kwargs["compression_opts"] = 4
        if chunk_on_first_dim and array.ndim >= 1:
            kwargs["chunks"] = (1,) + array.shape[1:]

    group.create_dataset(name, data=array, **kwargs)


def save_snapshot_payload_hdf(payload, event_name, save_dir):
    """Persist one event payload as a structured HDF5 file."""
    if payload is None:
        print(f"No snapshots to save for event '{event_name}'.")
        return None

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{event_name}_snapshots.h5")

    with h5py.File(file_path, "w") as h5f:
        static_group = h5f.create_group("static")
        windows_group = h5f.create_group("windows")

        for name, array in payload["static"].items():
            _create_hdf_dataset(static_group, name, array, chunk_on_first_dim=False)

        for name, array in payload["windows"].items():
            _create_hdf_dataset(windows_group, name, array, chunk_on_first_dim=(name != "time_index"))

        h5f.attrs["metadata_json"] = json.dumps(payload["metadata"])
        h5f.attrs["format_version"] = "graph_snapshots_hdf_v1"

    print(f"\nSaved {payload['metadata']['num_snapshots']} snapshots for event '{event_name}' to:\n  {file_path}")
    return file_path


def load_snapshot_from_hdf(file_path, snapshot_idx):
    """
    Load a single snapshot from an event HDF5 file and reconstruct a PyG `Data`.

    This helper is meant for future on-the-fly Dataset implementations.
    """
    with h5py.File(file_path, "r") as h5f:
        x_static = h5f["static"]["x_static"][...]
        edge_index = h5f["static"]["edge_index"][...]
        edge_attr = h5f["static"]["edge_attr"][...]
        node_type = h5f["static"]["node_type"][...]
        edge_type = h5f["static"]["edge_type"][...]

        x_dynamic = h5f["windows"]["x_dynamic"][snapshot_idx]
        y = h5f["windows"]["y"][snapshot_idx]
        future_drivers = h5f["windows"]["future_drivers"][snapshot_idx]
        time_index = h5f["windows"]["time_index"][snapshot_idx]

        snapshot_data = {
            "x": torch.from_numpy(np.concatenate([x_static, x_dynamic.reshape(x_dynamic.shape[0], -1)], axis=1)).float(),
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_attr": torch.from_numpy(edge_attr).float(),
            "y": torch.from_numpy(y.reshape(y.shape[0], -1)).float(),
            "future_drivers": torch.from_numpy(future_drivers.reshape(future_drivers.shape[0], -1)).float(),
            "time_index": torch.tensor([int(time_index)], dtype=torch.long),
            "node_type": torch.from_numpy(node_type).long(),
            "edge_type": torch.from_numpy(edge_type).long(),
        }

        if "x_aux" in h5f["windows"]:
            x_aux = h5f["windows"]["x_aux"][snapshot_idx]
            snapshot_data["x_aux"] = torch.from_numpy(x_aux.reshape(x_aux.shape[0], -1)).float()

        if "y_aux" in h5f["windows"]:
            y_aux = h5f["windows"]["y_aux"][snapshot_idx]
            snapshot_data["y_aux"] = torch.from_numpy(y_aux.reshape(y_aux.shape[0], -1)).float()

    return Data(**snapshot_data)


def print_example_snapshot_from_payload(payload):
    """Print a summary using the first snapshot reconstructed from the payload."""
    if payload is None or payload["metadata"]["num_snapshots"] == 0:
        print("\nNo HDF snapshots created to display.")
        return

    static = payload["static"]
    windows = payload["windows"]
    first_idx = 0

    x_dynamic = windows["x_dynamic"][first_idx]
    y = windows["y"][first_idx]
    future_drivers = windows["future_drivers"][first_idx]

    snapshot = Data(
        x=torch.from_numpy(np.concatenate([static["x_static"], x_dynamic.reshape(x_dynamic.shape[0], -1)], axis=1)).float(),
        edge_index=torch.from_numpy(static["edge_index"]).long(),
        edge_attr=torch.from_numpy(static["edge_attr"]).float(),
        y=torch.from_numpy(y.reshape(y.shape[0], -1)).float(),
        future_drivers=torch.from_numpy(future_drivers.reshape(future_drivers.shape[0], -1)).float(),
        time_index=torch.tensor([int(windows["time_index"][first_idx])], dtype=torch.long),
        node_type=torch.from_numpy(static["node_type"]).long(),
        edge_type=torch.from_numpy(static["edge_type"]).long(),
    )

    if "y_aux" in windows:
        y_aux = windows["y_aux"][first_idx]
        snapshot.y_aux = torch.from_numpy(y_aux.reshape(y_aux.shape[0], -1)).float()

    if "x_aux" in windows:
        x_aux = windows["x_aux"][first_idx]
        snapshot.x_aux = torch.from_numpy(x_aux.reshape(x_aux.shape[0], -1)).float()

    print("\n================ EXAMPLE HDF SNAPSHOT (first window) ================")
    print(snapshot)
    print(f"\nNode features 'x' shape: {snapshot.x.shape}")
    print(f"Edge index 'edge_index' shape: {snapshot.edge_index.shape}")
    print(f"Edge features 'edge_attr' shape: {snapshot.edge_attr.shape}")
    print(f"Node labels 'y' shape: {snapshot.y.shape}")
    print(f"Time index of this window: {snapshot.time_index.item()}")
    print("====================================================================\n")


def check_hdf_snapshots_for_nan(cfg):
    """Validate saved HDF5 graph files for NaN/Inf corruption."""
    print("\n--- Running HDF Snapshot Integrity Check ---")
    snapshot_dir = cfg["paths"].get("snapshot_hdf_dir")
    if not snapshot_dir or snapshot_dir == cfg["paths"].get("graphs_hdf_root"):
        snapshot_dir = resolve_hdf_graph_paths(cfg)["snapshot_hdf_dir"]

    h5_files = sorted(glob.glob(os.path.join(snapshot_dir, "*.h5")))
    if not h5_files:
        print(f"⚠️  No .h5 files found in {snapshot_dir} to check.")
        return False

    total_files = len(h5_files)
    total_snapshots_checked = 0
    bad_files_found = []

    for h5_file in tqdm(h5_files, desc="Checking .h5 files"):
        file_basename = os.path.basename(h5_file)
        try:
            with h5py.File(h5_file, "r") as h5f:
                num_snapshots = int(h5f["windows"]["time_index"].shape[0])
                total_snapshots_checked += num_snapshots

                x_static = h5f["static"]["x_static"][...]
                edge_attr = h5f["static"]["edge_attr"][...]

                if (
                    np.isnan(x_static).any()
                    or np.isinf(x_static).any()
                    or np.isnan(edge_attr).any()
                    or np.isinf(edge_attr).any()
                ):
                    bad_files_found.append((file_basename, -1, "static graph contains NaN/Inf"))
                    continue

                for snapshot_idx in range(num_snapshots):
                    x_dynamic = h5f["windows"]["x_dynamic"][snapshot_idx]
                    y = h5f["windows"]["y"][snapshot_idx]
                    future_drivers = h5f["windows"]["future_drivers"][snapshot_idx]

                    problems = []
                    if np.isnan(x_dynamic).any():
                        problems.append("'x_dynamic' has NaN")
                    if np.isinf(x_dynamic).any():
                        problems.append("'x_dynamic' has Inf")
                    if np.isnan(y).any():
                        problems.append("'y' has NaN")
                    if np.isinf(y).any():
                        problems.append("'y' has Inf")
                    if np.isnan(future_drivers).any():
                        problems.append("'future_drivers' has NaN")
                    if np.isinf(future_drivers).any():
                        problems.append("'future_drivers' has Inf")

                    if "x_aux" in h5f["windows"]:
                        x_aux = h5f["windows"]["x_aux"][snapshot_idx]
                        if np.isnan(x_aux).any():
                            problems.append("'x_aux' has NaN")
                        if np.isinf(x_aux).any():
                            problems.append("'x_aux' has Inf")

                    if "y_aux" in h5f["windows"]:
                        y_aux = h5f["windows"]["y_aux"][snapshot_idx]
                        if np.isnan(y_aux).any():
                            problems.append("'y_aux' has NaN")
                        if np.isinf(y_aux).any():
                            problems.append("'y_aux' has Inf")

                    if problems:
                        bad_files_found.append((file_basename, snapshot_idx, ", ".join(problems)))
                        break

        except Exception as exc:
            print(f"\nError loading {file_basename}: {exc}")
            bad_files_found.append((file_basename, -1, f"Failed to load: {exc}"))

    print(f"\nChecked {total_snapshots_checked} snapshots across {total_files} files.")

    if not bad_files_found:
        print("✅ SUCCESS: All HDF snapshots are clean (no NaN or Inf values found).")
        print("----------------------------------------\n")
        return True

    print(f"🔥 ERROR: Found {len(bad_files_found)} file(s) with corrupt data.")
    for filename, snap_idx, reason_str in bad_files_found:
        if snap_idx == -1:
            print(f"  - {filename}: {reason_str}")
        else:
            print(f"  - {filename} (at snapshot index {snap_idx}): {reason_str}")

    print("----------------------------------------\n")
    return False


def _normalize_dynamic_timeseries(raw_ts, nodes_df, all_stats, cfg):
    """Apply the same dynamic normalization logic used in `gnn4cf_graph_builder.py`."""
    norm_cfg = cfg.get("normalization", {})
    feature_methods = norm_cfg.get("feature_methods", {}).get("dynamic", {})
    default_method = norm_cfg.get("default_methods", {}).get("dynamic", "zscore")

    boundary_mask = (nodes_df["node_type"] == "bghost").to_numpy()

    for var in raw_ts.keys():
        method = feature_methods.get(var, default_method)
        if method == "no_norm":
            continue

        if var == "sea_level":
            if var in all_stats and all_stats[var].get("method") == method:
                var_stats = all_stats[var]
            else:
                sea_level_boundary = raw_ts[var][:, boundary_mask]
                if method == "minmax":
                    _, var_stats = base.min_max_normalize(sea_level_boundary)
                elif method == "zscore":
                    _, var_stats = base.zscore_normalize(sea_level_boundary)
                elif method == "log1p_zscore":
                    _, var_stats = base.log1p_zscore_normalize(sea_level_boundary)
                else:
                    print(f"  WARNING: Unknown method '{method}' for sea_level. Using zscore.")
                    method = "zscore"
                    _, var_stats = base.zscore_normalize(sea_level_boundary)

                all_stats[var] = var_stats

            sea_level_normalized = raw_ts[var].copy()
            sea_level_boundary = raw_ts[var][:, boundary_mask]

            if var_stats["method"] == "minmax":
                sea_level_boundary_norm, _ = base.min_max_normalize(
                    sea_level_boundary,
                    min_val=var_stats["min"],
                    max_val=var_stats["max"],
                )
            elif var_stats["method"] == "zscore":
                sea_level_boundary_norm, _ = base.zscore_normalize(
                    sea_level_boundary,
                    mean=var_stats["mean"],
                    std=var_stats["std"],
                )
            elif var_stats["method"] == "log1p_zscore":
                sea_level_boundary_norm, _ = base.log1p_zscore_normalize(
                    sea_level_boundary,
                    log1p_mean=var_stats["log1p_mean"],
                    log1p_std=var_stats["log1p_std"],
                )
            else:
                raise ValueError(f"Unsupported sea_level normalization method: {var_stats['method']}")

            sea_level_normalized[:, boundary_mask] = sea_level_boundary_norm
            raw_ts[var] = sea_level_normalized
            continue

        if var in all_stats and all_stats[var].get("method") == method:
            var_stats = all_stats[var]
        else:
            if method == "minmax":
                _, var_stats = base.min_max_normalize(raw_ts[var])
            elif method == "zscore":
                _, var_stats = base.zscore_normalize(raw_ts[var])
            elif method == "log1p_zscore":
                _, var_stats = base.log1p_zscore_normalize(raw_ts[var])
            else:
                print(f"  WARNING: Unknown method '{method}' for {var}. Using default '{default_method}'.")
                method = default_method
                if method == "minmax":
                    _, var_stats = base.min_max_normalize(raw_ts[var])
                elif method == "zscore":
                    _, var_stats = base.zscore_normalize(raw_ts[var])
                elif method == "log1p_zscore":
                    _, var_stats = base.log1p_zscore_normalize(raw_ts[var])
                else:
                    raise ValueError(f"Unsupported default normalization method: {default_method}")

            all_stats[var] = var_stats

        if var_stats["method"] == "minmax":
            raw_ts[var], _ = base.min_max_normalize(
                raw_ts[var],
                min_val=var_stats["min"],
                max_val=var_stats["max"],
            )
        elif var_stats["method"] == "zscore":
            raw_ts[var], _ = base.zscore_normalize(
                raw_ts[var],
                mean=var_stats["mean"],
                std=var_stats["std"],
            )
        elif var_stats["method"] == "log1p_zscore":
            raw_ts[var], _ = base.log1p_zscore_normalize(
                raw_ts[var],
                log1p_mean=var_stats["log1p_mean"],
                log1p_std=var_stats["log1p_std"],
            )
        else:
            raise ValueError(f"Unsupported normalization method: {var_stats['method']}")

    return raw_ts, all_stats


# ============================================================================
# Representative nodes and synthetic boundary→interior edges
# ============================================================================
def farthest_point_sampling(coords: np.ndarray, n_samples: int, random_state: int = 0) -> np.ndarray:
    """
    Simple farthest-point sampling (FPS) on 2D coordinates.

    Args:
        coords: Array of shape (N, 2) with (x, y) coordinates.
        n_samples: Desired number of representative points.
        random_state: RNG seed for reproducibility.

    Returns:
        indices: Array of length M (M <= n_samples, M <= N) with selected row indices into `coords`.
    """
    N = coords.shape[0]
    if N == 0:
        return np.array([], dtype=int)

    n_samples = min(n_samples, N)

    rng = np.random.default_rng(random_state)
    first = int(rng.integers(0, N))

    selected = [first]
    dists = np.linalg.norm(coords - coords[first], axis=1)

    for _ in range(1, n_samples):
        idx = int(np.argmax(dists))
        if dists[idx] <= 0.0:
            break
        selected.append(idx)
        new_d = np.linalg.norm(coords - coords[idx], axis=1)
        dists = np.minimum(dists, new_d)

    return np.asarray(selected, dtype=int)


def attach_bc_line_id_to_nodes(nodes_df, geom_data, face_attributes_df):
    """
    Attach bc_line_id to boundary ghost nodes in nodes_df when available.

    Uses external_faces['bc_line_id','face_index'] and face_attributes_df['face_id','bc_node'].
    """
    ext_faces = geom_data.get("external_faces")
    if ext_faces is None or ext_faces.empty or "bc_line_id" not in ext_faces.columns:
        return nodes_df

    try:
        ext_df = ext_faces.copy()
        fc_merge = face_attributes_df.merge(
            ext_df,
            left_on="face_id",
            right_on="face_index",
            how="inner",
        )
        if "bc_node" not in fc_merge.columns:
            return nodes_df

        bc_map = (
            fc_merge[["bc_node", "bc_line_id"]]
            .dropna(subset=["bc_node", "bc_line_id"])
            .astype({"bc_node": int, "bc_line_id": int})
            .drop_duplicates(subset=["bc_node"])
        )
        node_to_bc = dict(zip(bc_map["bc_node"], bc_map["bc_line_id"]))
        nodes_df = nodes_df.copy()
        nodes_df["bc_line_id"] = nodes_df["node_id"].map(node_to_bc)
    except Exception as e:
        print(f"[REPS] Warning: could not attach bc_line_id to nodes_df: {e}")
    return nodes_df


def build_representative_edges_per_bc(
    nodes_df,
    rep_node_ids: np.ndarray,
    k_neighbors_per_bc: int,
):
    """
    For each representative comp node, connect to k nearest boundary nodes
    on EACH BC line (when bc_line_id is available).

    Returns:
        list of (src_node_id, dst_node_id) tuples with src=bghost, dst=rep.
    """
    b_mask = nodes_df["node_type"] == "bghost"
    cols = ["node_id", "x", "y"]
    if "bc_line_id" in nodes_df.columns:
        cols.append("bc_line_id")
    b_nodes = nodes_df.loc[b_mask, cols].reset_index(drop=True)
    if b_nodes.empty or len(rep_node_ids) == 0:
        return []

    rep_df = nodes_df.set_index("node_id").loc[rep_node_ids]
    rep_df = rep_df[["x", "y"]].reset_index()
    rep_coords = rep_df[["x", "y"]].to_numpy(dtype=float)
    rep_ids = rep_df["node_id"].to_numpy(dtype=int)

    new_edges = []

    if "bc_line_id" in b_nodes.columns and not b_nodes["bc_line_id"].isna().all():
        valid_b = b_nodes.dropna(subset=["bc_line_id"]).copy()
        if not valid_b.empty:
            valid_b["bc_line_id"] = valid_b["bc_line_id"].astype(int)
            by_line = {bc: df.reset_index(drop=True) for bc, df in valid_b.groupby("bc_line_id")}
            for i_rep in range(rep_coords.shape[0]):
                rep_xy = rep_coords[i_rep]
                dst_id = int(rep_ids[i_rep])
                for _, df_line in by_line.items():
                    coords_line = df_line[["x", "y"]].to_numpy(dtype=float)
                    ids_line = df_line["node_id"].to_numpy(dtype=int)
                    if coords_line.shape[0] == 0:
                        continue
                    d_line = np.linalg.norm(coords_line - rep_xy[None, :], axis=1)
                    k_line = min(k_neighbors_per_bc, coords_line.shape[0])
                    if k_line <= 0:
                        continue
                    idx_sel = np.argpartition(d_line, kth=min(k_line - 1, d_line.size - 1))[:k_line]
                    for idx in idx_sel:
                        src_id = int(ids_line[int(idx)])
                        new_edges.append((src_id, dst_id))

    return new_edges


# ============================================================================
# Diagnostic plots (reused from representative_nodes_demo)
# ============================================================================
def compute_hop_distance_to_reps(
    edges_df,
    n_comp: int,
    rep_node_ids: np.ndarray,
):
    """
    Compute hop distance from each comp node to its nearest representative
    on the comp-comp (internal) graph.
    """
    from collections import deque as _deque

    internal = edges_df[edges_df["edge_type"] == "internal"]
    if internal.empty:
        hop_dist = np.full(n_comp, np.inf)
        return hop_dist, np.inf, 0.0

    # Build adjacency list (undirected): adj[node_id] = set of neighbor node_ids
    adj = [set() for _ in range(n_comp)]
    for _, r in internal.iterrows():
        s, d = int(r["src"]), int(r["dst"])
        if 0 <= s < n_comp and 0 <= d < n_comp:
            adj[s].add(d)
            adj[d].add(s)

    hop_dist = np.full(n_comp, np.inf)
    q = _deque()
    for rid in rep_node_ids:
        rid = int(rid)
        if 0 <= rid < n_comp:
            hop_dist[rid] = 0
            q.append(rid)

    while q:
        u = q.popleft()
        d_u = hop_dist[u]
        for v in adj[u]:
            if hop_dist[v] > d_u + 1:
                hop_dist[v] = d_u + 1
                q.append(v)

    reachable = np.isfinite(hop_dist)
    d_max = float(np.max(hop_dist[reachable])) if np.any(reachable) else np.inf
    d_mean = float(np.mean(hop_dist[reachable])) if np.any(reachable) else 0.0
    return hop_dist, d_max, d_mean


def plot_reps_and_coverage_single_figure(
    nodes_df,
    edges_df,
    rep_node_ids: np.ndarray,
    rep_edges,
    hop_dist: np.ndarray,
    k_layers: int,
    output_dir: str | None = None,
    filename: str = "representative_coverage.png",
):
    """
    Single figure with three subplots:
      1) Representatives + synthetic boundary→rep edges
      2) Hop distance to nearest rep
      3) Within-K vs beyond-K vs unreachable
    """
    n_comp = int(nodes_df["node_type"].eq("comp").sum())
    comp = nodes_df[nodes_df["node_type"] == "comp"].copy()
    comp = comp[comp["node_id"] < n_comp]
    if comp.empty:
        return

    comp["hop_dist"] = comp["node_id"].map(lambda i: hop_dist[int(i)] if int(i) < len(hop_dist) else np.nan)
    comp["hop_dist"] = comp["hop_dist"].fillna(np.inf)
    comp["reachable"] = np.isfinite(comp["hop_dist"])
    comp["within_k"] = (comp["hop_dist"] <= k_layers) & comp["reachable"]

    rep_set = set(rep_node_ids.astype(int))
    reps_df = comp[comp["node_id"].isin(rep_set)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1) Representatives and synthetic edges
    ax = axes[0]
    ax.scatter(comp["x"], comp["y"], s=4, c="lightgray", alpha=0.6, label="computational nodes")
    bghost = nodes_df[nodes_df["node_type"] == "bghost"]
    if not bghost.empty:
        ax.scatter(
            bghost["x"],
            bghost["y"],
            s=10,
            c="royalblue",
            alpha=0.8,
            label="boundary ghost nodes",
        )
    if not reps_df.empty:
        ax.scatter(
            reps_df["x"],
            reps_df["y"],
            s=40,
            c="crimson",
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            label="representative comp nodes",
            zorder=5,
        )
    for src_id, dst_id in rep_edges:
        src = nodes_df.loc[nodes_df["node_id"] == src_id].iloc[0]
        dst = nodes_df.loc[nodes_df["node_id"] == dst_id].iloc[0]
        ax.plot(
            [src["x"], dst["x"]],
            [src["y"], dst["y"]],
            color="crimson",
            alpha=0.25,
            linewidth=0.5,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Reps and synthetic boundary→rep edges")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")

    # 2) Hop distance to nearest rep
    ax = axes[1]
    reachable = comp[comp["reachable"]]
    unreachable = comp[~comp["reachable"]]
    if not unreachable.empty:
        ax.scatter(unreachable["x"], unreachable["y"], s=2, c="gray", alpha=0.5, label="unreachable")
    sc = None
    if not reachable.empty:
        d_vals = reachable["hop_dist"].values
        vmax = max(k_layers, int(np.nanmax(d_vals)) + 1) if np.any(np.isfinite(d_vals)) else k_layers
        sc = ax.scatter(
            reachable["x"],
            reachable["y"],
            s=4,
            c=reachable["hop_dist"],
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
    if not reps_df.empty:
        ax.scatter(reps_df["x"], reps_df["y"], s=60, c="red", marker="*", edgecolors="black", linewidths=0.5, label="reps", zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Hop distance to nearest rep")
    if sc is not None:
        plt.colorbar(sc, ax=ax, label="hops")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    # 3) Within K vs beyond K
    ax = axes[2]
    within = comp[comp["within_k"]]
    beyond = comp[~comp["within_k"] & comp["reachable"]]
    unreach = comp[~comp["reachable"]]
    if not unreach.empty:
        ax.scatter(unreach["x"], unreach["y"], s=2, c="gray", alpha=0.5, label="unreachable")
    if not within.empty:
        ax.scatter(within["x"], within["y"], s=4, c="green", alpha=0.6, label=f"within {k_layers} hops")
    if not beyond.empty:
        ax.scatter(beyond["x"], beyond["y"], s=4, c="red", alpha=0.6, label=f"beyond {k_layers} hops")
    if not reps_df.empty:
        ax.scatter(reps_df["x"], reps_df["y"], s=60, c="darkred", marker="*", edgecolors="black", linewidths=0.5, label="reps", zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    d_max_val = float(np.nanmax(hop_dist[np.isfinite(hop_dist)])) if np.any(np.isfinite(hop_dist)) else np.inf
    status = "OK" if d_max_val <= k_layers else "FAIL"
    d_str = f"{d_max_val:.0f}" if np.isfinite(d_max_val) else "inf"
    ax.set_title(f"Coverage (K={k_layers}): {status} (d_max={d_str})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save figure if an output directory is provided (e.g., graphs root).
    if output_dir is not None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, filename)
            plt.savefig(out_path, dpi=300)
            print(f"[REPS] Saved representative coverage figure to: {out_path}")
        except Exception as e:
            print(f"[REPS] Warning: failed to save representative coverage figure: {e}")
    else:
        # Fallback to interactive display when no output directory is specified.
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HDF graph snapshots from event HDF files (with optional representative-based synthetic edges).")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to config YAML (default: config.yml)",
    )
    args = parser.parse_args()

    cfg = base.load_config(args.config)
    graph_paths = apply_hdf_graph_paths_to_config(cfg)
    ensure_hdf_graph_dirs(graph_paths)
    config_copy_path = save_runtime_config_copy(cfg, graph_paths)

    norm_cfg = cfg.get("normalization", {})
    norm_enabled = norm_cfg.get("enabled", False)
    stats_path = graph_paths["normalization_stats_path"]
    all_stats = base.load_stats_file(stats_path)

    # Representative / synthetic-edge configuration (from config.yml)
    reps_cfg = cfg.get("representatives", {})
    reps_enabled = reps_cfg.get("enabled", True)
    num_reps = int(reps_cfg.get("num_reps", 120))
    k_neighbors_per_bc = int(reps_cfg.get("k_neighbors_per_bc", 3))
    # Optional interior layers for sanity print; training model still uses its own config.
    k_layers_interior = int(
        reps_cfg.get(
            "k_layers_interior",
            cfg.get("model", {}).get("interior", {}).get("nmessage_passing_steps", 5),
        )
    )

    print("\n--- HDF Graph Output Layout ---")
    print(f"Graph root: {graph_paths['graph_root']}")
    print(f"WD normalization label: {graph_paths['wd_norm_label']}")
    print(f"HDF snapshots: {graph_paths['snapshot_hdf_dir']}")
    print(f"Train/val snapshots: {graph_paths['train_val_snapshot_dir']}")
    print(f"Test snapshots: {graph_paths['test_snapshot_dir']}")
    print(f"Initial test graphs: {graph_paths['initial_test_graph_dir']}")
    print(f"Normalization stats: {graph_paths['normalization_stats_path']}")
    print(f"Config copy: {config_copy_path}")

    print("--- Processing Static Data (once for all events) ---")
    print(f"[REPS] enabled={reps_enabled}, num_reps={num_reps}, "
          f"k_neighbors_per_bc={k_neighbors_per_bc}, k_layers_interior={k_layers_interior}")
    cell_attributes_df = base.load_cell_attributes(cfg)
    face_attributes_df = base.load_face_attributes(cfg)

    hdf_dir = cfg["paths"]["copy_each_hdf_to"]
    try:
        sample_hdf_path = glob.glob(os.path.join(hdf_dir, "*.hdf"))[0]
    except IndexError as exc:
        raise FileNotFoundError(f"No .hdf files found in '{hdf_dir}'") from exc

    print(f"Using sample HDF for geometry: {os.path.basename(sample_hdf_path)}")
    geom_data = base.load_hdf_required(sample_hdf_path, cfg)
    nodes_df, index_map, masks = base.build_nodes(geom_data, face_attributes_df)
    # Attach bc_line_id to boundary ghost nodes when possible
    nodes_df = attach_bc_line_id_to_nodes(nodes_df, geom_data, face_attributes_df)
    edges_df = base.build_edges(face_attributes_df, n_comp=geom_data["n_comp"])

    # ------------------------------------------------------------------
    # Optional: add synthetic boundary→representative edges
    # Representatives are selected only from interior comp nodes that are
    # NOT already connected to boundary (exclude "first row").
    # ------------------------------------------------------------------
    rep_node_ids = None
    rep_edges = None
    if reps_enabled and num_reps > 0 and k_neighbors_per_bc > 0:
        n_comp = int(geom_data.get("n_comp", 0))
        # First row: comp nodes that already have a boundary edge (exclude from FPS)
        boundary_edges = edges_df[edges_df["edge_type"] == "boundary"]
        first_row_comp_ids = set()
        if not boundary_edges.empty:
            for col in ["src", "dst"]:
                first_row_comp_ids.update(boundary_edges[col].astype(int).tolist())
        first_row_comp_ids = {nid for nid in first_row_comp_ids if 0 <= nid < n_comp}

        comp_mask = nodes_df["node_type"] == "comp"
        comp_nodes_all = nodes_df.loc[comp_mask, ["node_id", "x", "y"]].reset_index(drop=True)
        # Restrict FPS to inland comp nodes only (exclude first row)
        candidates_mask = ~comp_nodes_all["node_id"].isin(first_row_comp_ids)
        comp_nodes = comp_nodes_all[candidates_mask].reset_index(drop=True)

        if comp_nodes.empty:
            print("[REPS] WARNING: no inland computational nodes (after excluding first row); skipping synthetic edges.")
        else:
            comp_coords = comp_nodes[["x", "y"]].to_numpy(dtype=float)
            comp_ids = comp_nodes["node_id"].to_numpy(dtype=int)
            fps_idx = farthest_point_sampling(comp_coords, n_samples=num_reps, random_state=0)
            rep_node_ids = comp_ids[fps_idx]
            print(f"[REPS] Excluded {len(first_row_comp_ids)} first-row comp nodes; selected {len(rep_node_ids)} representative inland nodes (FPS).")

            rep_edges = build_representative_edges_per_bc(
                nodes_df=nodes_df,
                rep_node_ids=rep_node_ids,
                k_neighbors_per_bc=k_neighbors_per_bc,
            )
            # Deduplicate
            rep_edges = list(set(rep_edges))
            print(f"[REPS] Added {len(rep_edges)} synthetic boundary→rep edges "
                  f"({len(rep_node_ids)} reps × up to {k_neighbors_per_bc} per BC line).")

            if rep_edges:
                import pandas as pd

                rep_edges_df = pd.DataFrame(rep_edges, columns=["src", "dst"])

                # Assign unique synthetic face_ids after the last real face_id,
                # and mark them as synthetic for later inspection.
                try:
                    max_face_id = int(face_attributes_df["face_id"].max())
                except Exception:
                    max_face_id = 0
                synthetic_face_ids = np.arange(max_face_id + 1, max_face_id + 1 + len(rep_edges_df), dtype=int)
                rep_edges_df["face_id"] = synthetic_face_ids
                rep_edges_df["is_synthetic"] = 1

                # Ensure original edges have is_synthetic=0 so we can distinguish them.
                if "is_synthetic" not in edges_df.columns:
                    edges_df["is_synthetic"] = 0

                # For static edge features, extend face_attributes_df with synthetic
                # face rows that reuse a template face's properties.
                if not face_attributes_df.empty:
                    template_row = face_attributes_df.iloc[0].copy()
                    synth_face_rows = []
                    for fid in synthetic_face_ids:
                        r = template_row.copy()
                        r["face_id"] = int(fid)
                        synth_face_rows.append(r)
                    if synth_face_rows:
                        face_attributes_df = pd.concat(
                            [face_attributes_df, pd.DataFrame(synth_face_rows)],
                            ignore_index=True,
                        )

                rep_edges_df["edge_type"] = "boundary"

                edges_df = pd.concat([edges_df, rep_edges_df], ignore_index=True)
    else:
        print("[REPS] Representative-based synthetic edges DISABLED (using original graph).")

    # ------------------------------------------------------------------
    # Coverage check (before graph generation): Pass/Fail
    # ------------------------------------------------------------------
    if reps_enabled and rep_node_ids is not None:
        n_comp = int(geom_data.get("n_comp", 0))
        hop_dist, d_max, d_mean = compute_hop_distance_to_reps(
            edges_df=edges_df,
            n_comp=n_comp,
            rep_node_ids=rep_node_ids,
        )
        n_reachable = int(np.sum(np.isfinite(hop_dist)))
        n_unreachable = n_comp - n_reachable
        ok = d_max <= k_layers_interior and n_unreachable == 0
        print("\n" + "=" * 60)
        print("  REPRESENTATIVE COVERAGE CHECK (before graph generation)")
        print("=" * 60)
        print(f"  k_layers_interior (K) : {k_layers_interior}")
        print(f"  d_max (max hops to rep): {d_max:.0f}" if np.isfinite(d_max) else "  d_max (max hops to rep): inf")
        print(f"  d_mean                 : {d_mean:.2f}")
        print(f"  reachable comp nodes   : {n_reachable} / {n_comp}")
        if n_unreachable > 0:
            print(f"  unreachable            : {n_unreachable}")
        print(f"  ---")
        status = "PASS" if ok else "FAIL"
        print(f"  Result: {status}  (d_max <= K and no unreachable? {'Yes' if ok else 'No'})")
        print("=" * 60 + "\n")
        if not ok:
            print("[REPS] WARNING: Coverage check FAILED. Consider increasing num_reps or k_layers_interior in config.")

    nodes_static_df = base.attach_static_node_features(
        nodes_df,
        geom_data,
        cell_attributes_df,
    )
    edges_static_df = base.attach_static_edge_features(
        edges_df,
        face_attributes_df,
        nodes_df,
    )
    nodes_static_df = base.load_and_merge_external_features(
        nodes_static_df,
        cfg["paths"].get("external_node_features", []),
    )
    nodes_static_df = nodes_static_df.fillna(0.0)

    if norm_enabled:
        print("Applying global normalization to static features...")
        nodes_static_df, all_stats = base.normalize_global_static_features(nodes_static_df, all_stats, cfg)
        edges_static_df, all_stats = base.normalize_global_static_features(edges_static_df, all_stats, cfg)

    final_nodes_static_df = base.select_final_features(
        nodes_static_df,
        cfg,
        ["node_id"],
        "node",
    )
    final_edges_static_df = base.select_final_features(
        edges_static_df,
        cfg,
        ["src", "dst", "face_id"],
        "edge",
    )
    base.summarize_graph(nodes_df, edges_df)

    node_type_map = {"comp": 0, "bghost": 1, "non_bc_ghost": 2}
    node_type_tensor = torch.tensor(nodes_df["node_type"].map(node_type_map).values, dtype=torch.long)
    edge_type_map = {"internal": 0, "boundary": 1}
    edge_type_tensor = torch.tensor(edges_df["edge_type"].map(edge_type_map).values, dtype=torch.long)

    hdf_files = sorted(glob.glob(os.path.join(hdf_dir, "*.hdf")))
    last_payload = None

    for hdf_path in hdf_files:
        event_name = os.path.splitext(os.path.basename(hdf_path))[0]
        print(f"\n--- Processing Event: {event_name} ---")
        hdf_data = base.load_hdf_required(hdf_path, cfg)

        window_cfg = cfg.get("window", {})
        warmup_steps = window_cfg.get("warmup_steps", 0)
        warmup_enabled = window_cfg.get("warmup_enabled", False)
        print(f"  [WARMUP] warmup_enabled={warmup_enabled}, warmup_steps={warmup_steps}")
        hdf_data = base.trim_warmup_from_hdf_data(hdf_data, warmup_steps, enabled=warmup_enabled)

        raw_ts = base.gather_and_compute_node_timeseries(hdf_data, nodes_df, face_attributes_df, cfg)

        if norm_enabled:
            raw_ts, all_stats = _normalize_dynamic_timeseries(raw_ts, nodes_df, all_stats, cfg)

        payload = create_graph_snapshot_payload_hdf(
            final_nodes_static_df,
            edges_df,
            final_edges_static_df,
            raw_ts,
            cfg,
            node_type_tensor,
            edge_type_tensor,
        )

        save_snapshot_payload_hdf(payload, event_name, graph_paths["snapshot_hdf_dir"])
        last_payload = payload

    if norm_enabled:
        base.save_stats_file(all_stats, stats_path)

    print("\n--- All events processed successfully! ---")
    check_hdf_snapshots_for_nan(cfg)

    # Show example graph summary for the last event only (not after each event).
    if last_payload is not None:
        print("\n--- Example snapshot (last event) ---")
        print_example_snapshot_from_payload(last_payload)

    print("\n--- Starting Interactive Feature Inspector ---")
    print("Enter 'node <id>', 'face <id>', or 'exit' to quit.")

    while True:
        try:
            user_input = input("> ").strip().lower()
            if user_input == "exit":
                break

            parts = user_input.split()
            if len(parts) != 2:
                print("Invalid format. Use 'node 123' or 'face 456'.")
                continue

            entity_type, entity_id_str = parts
            entity_id = int(entity_id_str)

            if entity_type == "node":
                base.print_node_features_by_id(
                    node_id=entity_id,
                    nodes_df=nodes_df,
                    nodes_static_df=final_nodes_static_df,
                    raw_ts=raw_ts,
                    cfg=cfg,
                    index_map=index_map,
                )
            elif entity_type == "face":
                base.print_edge_features_by_id(
                    face_id=entity_id,
                    edges_df=edges_df,
                    edges_static_df=final_edges_static_df,
                    nodes_df=nodes_df,
                    raw_ts=raw_ts,
                    cfg=cfg,
                    index_map=index_map,
                )
            else:
                print(f"Unknown entity type '{entity_type}'. Use 'node' or 'face'.")

        except ValueError:
            print("Invalid ID. Please enter a number.")
        except Exception as exc:
            print(f"An unexpected error occurred: {exc}")

    # Summarize representative placement and hop coverage after inspection.
    if reps_enabled and rep_node_ids is not None and rep_edges:
        try:
            hop_dist, d_max, d_mean = compute_hop_distance_to_reps(
                edges_df=edges_df,
                n_comp=int(geom_data.get("n_comp", 0)),
                rep_node_ids=rep_node_ids,
            )
            print(f"[REPS] Hop coverage diagnostics: d_max={d_max:.0f}, d_mean={d_mean:.2f}, "
                  f"K_layers_interior={k_layers_interior}")
            plot_reps_and_coverage_single_figure(
                nodes_df=nodes_df,
                edges_df=edges_df,
                rep_node_ids=rep_node_ids,
                rep_edges=rep_edges,
                hop_dist=hop_dist,
                k_layers=k_layers_interior,
                output_dir=graph_paths.get("graph_root"),
                filename="representative_coverage.png",
            )
        except Exception as e:
            print(f"[REPS] Warning: failed to generate representative diagnostics plots: {e}")
