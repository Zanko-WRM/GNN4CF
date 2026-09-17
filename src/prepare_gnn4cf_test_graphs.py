# -*- coding: utf-8 -*-

"""
Prepare rollout-ready initial test graphs for GNN4CF evaluation.

For each selected test event, cache the first graph snapshot, future driver
sequence, and ground-truth water-depth sequence from its HDF5 dataset.
Use the same manifest split, resolved paths, and feature ordering as training.
"""

import argparse
import json
import os
import pickle
import re
from datetime import datetime

import h5py
import torch
import yaml
from torch_geometric.data import Data

from gnn4cf_hdf_graph_dataset import apply_hdf_graph_paths_to_config
from train_gnn4cf import (
    apply_run_output_paths_to_config,
    get_feature_counts_and_indices_hdf_for_config,
    list_hdf_event_files,
    parse_event_name_from_hdf_filename,
)


TEST_META_FILENAME = "test_extraction_metadata_hdf.json"
INITIAL_GRAPHS_FILENAME = "initial_test_graphs.pkl"
INITIAL_GRAPHS_MANIFEST_FILENAME = "initial_test_graphs_manifest.json"


def load_resolved_config(config_path):
    """Load YAML and resolve graph/run output paths shared with training."""
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    apply_hdf_graph_paths_to_config(config)
    apply_run_output_paths_to_config(config)
    config.setdefault("paths", {})
    config["paths"]["config_path"] = config_path
    return config


def sanitize_event_name(event_name):
    """Convert an event name into a filesystem-safe artifact stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(event_name)).strip("_")


def normalize_event_name(event_name):
    """Normalize event names across metadata, filenames, and saved artifacts."""
    event_name = str(event_name)
    event_name = event_name.replace("_snapshots.h5", "").replace("_snapshots", "")
    event_name = re.sub(r"^Flood_model\.p\d+_", "", event_name)
    return event_name


def _metadata_path(config):
    return os.path.join(config["paths"]["test_snapshot_dir"], TEST_META_FILENAME)


def _initial_graph_cache_path(config):
    return os.path.join(config["paths"]["initial_test_graph_dir"], INITIAL_GRAPHS_FILENAME)


def _initial_graph_manifest_path(config):
    return os.path.join(config["paths"]["initial_test_graph_dir"], INITIAL_GRAPHS_MANIFEST_FILENAME)


def load_test_event_names(config):
    """Load cached test-event names produced by the training split step."""
    meta_path = _metadata_path(config)
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    names = metadata.get("test_event_names", None)
    if names is None:
        return None
    return [normalize_event_name(name) for name in names]


def resolve_test_event_files(config, max_events=None):
    """
    Return rollout event files ordered consistently with training metadata.

    If test split metadata exists, only those test events are used and the order
    matches `test_event_names`. Otherwise all HDF event files are used.
    """
    snapshot_hdf_dir = config["paths"]["snapshot_hdf_dir"]
    all_files = list_hdf_event_files(snapshot_hdf_dir)
    if not all_files:
        raise FileNotFoundError(f"No HDF event files found in '{snapshot_hdf_dir}'.")

    file_map = {}
    for file_path in all_files:
        _, event_name = parse_event_name_from_hdf_filename(file_path)
        normalized = normalize_event_name(event_name)
        file_map[normalized] = file_path

    test_event_names = load_test_event_names(config)
    if test_event_names:
        ordered = [file_map[normalize_event_name(name)] for name in test_event_names if normalize_event_name(name) in file_map]
    else:
        ordered = sorted(all_files)

    if max_events is not None:
        ordered = ordered[: int(max_events)]

    if not ordered:
        raise ValueError("No rollout event files were selected after applying test-event filtering.")

    return ordered


def _concat_overlapping_future_windows(window_tensor):
    """
    Collapse overlapping future windows from `[S, N, W, F]` into `[N, S+W-1, F]`.

    Window 0 contributes all `W` steps; each later window contributes only its
    final step because the earlier `W-1` steps overlap with the previous window.
    """
    if window_tensor.ndim != 4:
        raise ValueError(f"Expected window tensor [S, N, W, F], got shape {tuple(window_tensor.shape)}")

    chunks = [window_tensor[0]]
    if window_tensor.shape[0] > 1:
        chunks.extend(window_tensor[idx, :, -1:, :] for idx in range(1, window_tensor.shape[0]))
    return torch.cat(chunks, dim=1)


def build_rollout_entry_from_hdf(file_path, config, feature_counts):
    """Build one rollout cache entry from one event HDF file."""
    event_name = parse_event_name_from_hdf_filename(file_path)[1]
    sanitized_event_name = sanitize_event_name(event_name)

    n_state_vars = feature_counts["n_state_vars"]
    n_driver_vars = feature_counts["n_driver_vars"]
    n_label_vars = feature_counts["n_label_vars"]

    with h5py.File(file_path, "r") as h5f:
        x_static = torch.from_numpy(h5f["static"]["x_static"][...]).float()
        edge_index = torch.from_numpy(h5f["static"]["edge_index"][...]).long()
        edge_attr = torch.from_numpy(h5f["static"]["edge_attr"][...]).float()
        node_type = torch.from_numpy(h5f["static"]["node_type"][...]).long()
        edge_type = torch.from_numpy(h5f["static"]["edge_type"][...]).long()

        x_dynamic = torch.from_numpy(h5f["windows"]["x_dynamic"][...]).float()
        y_windows = torch.from_numpy(h5f["windows"]["y"][...]).float()
        future_driver_windows = torch.from_numpy(h5f["windows"]["future_drivers"][...]).float()
        time_index = torch.from_numpy(h5f["windows"]["time_index"][...]).long()

    if x_dynamic.shape[0] == 0:
        raise ValueError(f"No rollout windows found in '{file_path}'.")

    first_x_dynamic = x_dynamic[0]  # [N, past_steps, n_state+n_driver]
    initial_graph = Data(
        x=torch.cat([x_static, first_x_dynamic.reshape(first_x_dynamic.shape[0], -1)], dim=1),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        edge_type=edge_type,
        event_name=event_name,
        sanitized_event_name=sanitized_event_name,
        source_event_file=file_path,
    )

    drivers_from_x = first_x_dynamic[:, :, n_state_vars:]  # [N, past_steps, n_driver_vars]
    full_future_drivers = _concat_overlapping_future_windows(future_driver_windows)
    full_length_drivers = torch.cat([drivers_from_x, full_future_drivers], dim=1).reshape(first_x_dynamic.shape[0], -1)

    y_truth_full = _concat_overlapping_future_windows(y_windows)

    total_time_steps = int(full_length_drivers.shape[1] // max(1, n_driver_vars))
    future_time_steps = int(y_truth_full.shape[1])

    return {
        "event_name": event_name,
        "sanitized_event_name": sanitized_event_name,
        "event_file": file_path,
        "initial_graph": initial_graph,
        "full_length_drivers": full_length_drivers,
        "y_truth_full": y_truth_full,
        "time_index_start": int(time_index[0].item()) if len(time_index) else 0,
        "num_snapshots": int(x_dynamic.shape[0]),
        "num_nodes": int(initial_graph.x.shape[0]),
        "total_time_steps": total_time_steps,
        "future_time_steps": future_time_steps,
        "n_driver_vars": int(n_driver_vars),
        "n_label_vars": int(n_label_vars),
    }


def build_initial_test_graph_entries(config, feature_counts, max_events=None):
    """Build rollout cache entries for all selected test events."""
    event_files = resolve_test_event_files(config, max_events=max_events)
    entries = []

    for file_path in event_files:
        entry = build_rollout_entry_from_hdf(file_path, config, feature_counts)
        entries.append(entry)
        print(
            f"Prepared rollout cache for {entry['event_name']}: "
            f"nodes={entry['num_nodes']}, snapshots={entry['num_snapshots']}, "
            f"T_total={entry['total_time_steps']}, T_future={entry['future_time_steps']}"
        )

    return entries


def save_initial_test_graph_cache(entries, config):
    """Persist rollout cache and a sidecar manifest under the resolved graph root."""
    output_dir = config["paths"]["initial_test_graph_dir"]
    os.makedirs(output_dir, exist_ok=True)

    pickle_payload = [
        (entry["initial_graph"], entry["full_length_drivers"], entry["y_truth_full"])
        for entry in entries
    ]

    cache_path = _initial_graph_cache_path(config)
    with open(cache_path, "wb") as f:
        pickle.dump(pickle_payload, f)

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_name": config.get("config_name"),
        "config_path": config["paths"].get("config_path"),
        "graph_root": config["paths"].get("graph_root"),
        "initial_graph_cache": cache_path,
        "entries": [
            {
                "initial_graph_index": idx,
                "event_name": entry["event_name"],
                "sanitized_event_name": entry["sanitized_event_name"],
                "event_file": entry["event_file"],
                "num_nodes": entry["num_nodes"],
                "num_snapshots": entry["num_snapshots"],
                "total_time_steps": entry["total_time_steps"],
                "future_time_steps": entry["future_time_steps"],
                "n_driver_vars": entry["n_driver_vars"],
                "n_label_vars": entry["n_label_vars"],
            }
            for idx, entry in enumerate(entries)
        ],
    }

    manifest_path = _initial_graph_manifest_path(config)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return cache_path, manifest_path


def load_initial_test_graph_cache(config):
    """
    Load cached rollout entries if they exist.

    Returns a normalized list of dicts with:
    - event_name
    - sanitized_event_name
    - initial_graph
    - full_length_drivers
    - y_truth_full
    - initial_graph_index
    """
    cache_path = _initial_graph_cache_path(config)
    if not os.path.exists(cache_path):
        return None

    manifest_path = _initial_graph_manifest_path(config)
    manifest_entries = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_entries = manifest.get("entries", [])

    with open(cache_path, "rb") as f:
        cached = pickle.load(f)

    fallback_event_names = load_test_event_names(config) or []
    normalized = []
    for idx, item in enumerate(cached):
        if isinstance(item, dict):
            initial_graph = item["initial_graph"]
            full_length_drivers = item["full_length_drivers"]
            y_truth_full = item["y_truth_full"]
            event_name = item.get("event_name")
            sanitized_event_name = item.get("sanitized_event_name")
        else:
            initial_graph, full_length_drivers, y_truth_full = item[:3]
            event_name = getattr(initial_graph, "event_name", None)
            sanitized_event_name = getattr(initial_graph, "sanitized_event_name", None)

        meta = manifest_entries[idx] if idx < len(manifest_entries) else {}
        event_name = event_name or meta.get("event_name")
        if event_name is None and idx < len(fallback_event_names):
            event_name = fallback_event_names[idx]
        if event_name is None:
            event_name = f"event_{idx:04d}"

        sanitized_event_name = sanitized_event_name or meta.get("sanitized_event_name") or sanitize_event_name(event_name)

        normalized.append(
            {
                "initial_graph_index": idx,
                "event_name": event_name,
                "sanitized_event_name": sanitized_event_name,
                "event_file": meta.get("event_file"),
                "initial_graph": initial_graph,
                "full_length_drivers": full_length_drivers,
                "y_truth_full": y_truth_full,
            }
        )

    return normalized


def ensure_initial_test_graph_cache(config, feature_counts, rebuild=False, max_events=None):
    """Load existing rollout cache or rebuild it if missing / requested."""
    if not rebuild:
        cached = load_initial_test_graph_cache(config)
        if cached is not None:
            return cached

    entries = build_initial_test_graph_entries(config, feature_counts, max_events=max_events)
    save_initial_test_graph_cache(entries, config)
    return load_initial_test_graph_cache(config)


def run_interactive_inspector(entries, config, feature_counts):
    """Optional lightweight REPL for inspecting one cached rollout entry."""
    if not entries:
        print("No rollout entries available for interactive inspection.")
        return

    initial_graph = entries[0]["initial_graph"]
    full_length_drivers = entries[0]["full_length_drivers"]
    n_driver_vars = feature_counts["n_driver_vars"]
    total_time_steps = full_length_drivers.shape[1] // max(1, n_driver_vars)

    print("\n" + "=" * 80)
    print("Interactive Rollout Cache Inspector")
    print("=" * 80)
    print("Type a node id to inspect, or press Enter / type 'quit' to exit.")
    print(f"Available node ids: 0 to {initial_graph.x.shape[0] - 1}")
    print(f"Driver variables: {config.get('features', {}).get('dynamic_input_drivers', [])}")
    print(f"Total time steps: {total_time_steps}")

    while True:
        user_input = input("\nNode id: ").strip().lower()
        if user_input in ("", "q", "quit", "exit"):
            break

        try:
            node_id = int(user_input)
        except ValueError:
            print("Please enter an integer node id.")
            continue

        if node_id < 0 or node_id >= initial_graph.x.shape[0]:
            print(f"Node id must be between 0 and {initial_graph.x.shape[0] - 1}.")
            continue

        node_type_val = None
        if hasattr(initial_graph, "node_type"):
            node_type_val = int(initial_graph.node_type[node_id].item())

        driver_window = full_length_drivers[node_id].reshape(total_time_steps, n_driver_vars)
        print(f"\nNode {node_id}")
        print(f"  node_type: {node_type_val}")
        print(f"  x shape: {tuple(initial_graph.x[node_id].shape)}")
        print(f"  first driver step: {driver_window[0].tolist() if total_time_steps > 0 else 'n/a'}")
        print(f"  last driver step:  {driver_window[-1].tolist() if total_time_steps > 0 else 'n/a'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare rollout-ready initial test graphs for GNN4CF evaluation.")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config YAML.")
    parser.add_argument("--max-events", type=int, default=None, help="Limit the number of events processed.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the cache even if it already exists.")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the interactive node inspector after cache creation.",
    )
    args = parser.parse_args()

    config = load_resolved_config(args.config)
    feature_counts = get_feature_counts_and_indices_hdf_for_config(config)
    entries = ensure_initial_test_graph_cache(
        config,
        feature_counts,
        rebuild=args.rebuild,
        max_events=args.max_events,
    )

    cache_path = _initial_graph_cache_path(config)
    manifest_path = _initial_graph_manifest_path(config)
    print(f"\nSaved / loaded {len(entries)} rollout entries")
    print(f"  cache:    {cache_path}")
    print(f"  manifest: {manifest_path}")

    if not args.no_interactive:
        run_interactive_inspector(entries, config, feature_counts)


if __name__ == "__main__":
    main()
