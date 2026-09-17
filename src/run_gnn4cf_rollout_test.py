# -*- coding: utf-8 -*-

"""
Run autoregressive rollout evaluation for a trained GNN4CF model.

Construct the model from configuration and load its checkpoint strictly.
Read or prepare cached test graphs, predict each event using known future
forcing, and save water-depth predictions, ground truth, metrics, and an
event manifest to the resolved results directory.
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from gnn4cf_model import GNNModel
from gnn4cf_training_utils import rollout_autoregressive, set_random_seed
from train_gnn4cf import (
    build_dynamic_input_feature_names,
    get_feature_counts_and_indices_hdf_for_config,
    get_rainfall_conditioning_kwargs,
)
from prepare_gnn4cf_test_graphs import (
    ensure_initial_test_graph_cache,
    load_initial_test_graph_cache,
    load_resolved_config,
    sanitize_event_name,
)


ROLLOUT_MANIFEST_FILENAME = "rollout_manifest.json"


def compute_metrics(preds, y_truth, label_vars):
    """Compute MSE and RMSE for each label variable plus overall totals."""
    y_truth_reshaped = y_truth.permute(1, 0, 2)

    metrics = {}
    for var_idx, var_name in enumerate(label_vars):
        if var_idx >= preds.shape[2]:
            continue
        pred_var = preds[:, :, var_idx]
        truth_var = y_truth_reshaped[:, :, var_idx]
        mse = torch.mean((pred_var - truth_var) ** 2).item()
        rmse = torch.sqrt(torch.mean((pred_var - truth_var) ** 2)).item()
        metrics[f"mse_{var_name}"] = mse
        metrics[f"rmse_{var_name}"] = rmse

    total_mse = torch.mean((preds - y_truth_reshaped) ** 2).item()
    total_rmse = torch.sqrt(torch.mean((preds - y_truth_reshaped) ** 2)).item()
    metrics["mse_total"] = total_mse
    metrics["rmse_total"] = total_rmse
    return metrics


def build_model_from_config(config, feature_counts, device):
    """Instantiate GNN4CF from the supplied configuration."""
    cfg_model = config["model"]
    cfg_window = config["window"]

    shared_steps = cfg_model["nmessage_passing_steps"]
    shared_layers = cfg_model["nmlp_layers"]
    shared_hidden = cfg_model["mlp_hidden_dim"]

    interior_cfg = cfg_model.get("interior", {}) or {}
    coupling_cfg = cfg_model.get("coupling", {}) or {}
    boundary_skip = cfg_model.get("boundary_skip", True)
    boundary_preserve_weight = cfg_model.get("boundary_preserve_weight", None)
    residual = cfg_model.get("residual", True)

    bc_cfg = config.get("boundary_conditioning", {}) or {}
    dynamic_input_feature_names = build_dynamic_input_feature_names(config)
    rainfall_kwargs = get_rainfall_conditioning_kwargs(
        config,
        dynamic_input_feature_names=dynamic_input_feature_names,
    )

    model = GNNModel(
        predictor_step=cfg_window["predictor_step"],
        n_static_node_comp=feature_counts["n_static_node_comp"],
        n_static_node_bghost=feature_counts["n_static_node_bghost"],
        n_static_edge_internal=feature_counts["n_static_edge_internal"],
        n_static_edge_boundary=feature_counts["n_static_edge_boundary"],
        n_dynamic_node_vars=feature_counts["n_dynamic_node"],
        n_dynamic_edge_vars=feature_counts["n_dynamic_edge"],
        n_label_vars=feature_counts["n_label_vars"],
        latent_dim=cfg_model["latent_dim"],
        nmessage_passing_steps=shared_steps,
        nmlp_layers=shared_layers,
        mlp_hidden_dim=shared_hidden,
        nmessage_passing_steps_interior=interior_cfg.get("nmessage_passing_steps"),
        nmlp_layers_interior=interior_cfg.get("nmlp_layers"),
        mlp_hidden_dim_interior=interior_cfg.get("mlp_hidden_dim"),
        nmessage_passing_steps_coupling=coupling_cfg.get("nmessage_passing_steps"),
        nmlp_layers_coupling=coupling_cfg.get("nmlp_layers"),
        mlp_hidden_dim_coupling=coupling_cfg.get("mlp_hidden_dim"),
        enable_coupling=coupling_cfg.get("enable_coupling", True),
        coupling_gate_mode=coupling_cfg.get("coupling_gate_mode", "learned"),
        residual=residual,
        boundary_skip=boundary_skip,
        boundary_preserve_weight=boundary_preserve_weight,
        boundary_conditioning_enabled=bool(bc_cfg.get("enabled", False)),
        boundary_conditioning_mode=str(bc_cfg.get("mode", "concat")),
        inject_every_processor_step=bool(bc_cfg.get("inject_every_processor_step", True)),
        use_current_state=bool(bc_cfg.get("use_current_state", True)),
        use_history_state=bool(bc_cfg.get("use_history_state", True)),
        use_geometry=bool(bc_cfg.get("use_geometry", True)),
        geometry_interaction=str(bc_cfg.get("geometry_interaction", "multiply")),
        current_encoder_hidden_dim=int(bc_cfg.get("current_encoder_hidden_dim", 64)),
        history_encoder_hidden_dim=int(bc_cfg.get("history_encoder_hidden_dim", 64)),
        geometry_encoder_hidden_dim=int(bc_cfg.get("geometry_encoder_hidden_dim", 64)),
        update_mlp_hidden_dim=int(bc_cfg.get("update_mlp_hidden_dim", 64)),
        film_hidden_dim=int(bc_cfg.get("film_hidden_dim", 64)),
        **rainfall_kwargs,
    ).to(device)
    return model


def resolve_checkpoint_path(config, checkpoint_path=None):
    """Resolve the rollout checkpoint path from the run-scoped config paths."""
    if checkpoint_path:
        return os.path.abspath(checkpoint_path)

    checkpoint_dir = config["paths"]["checkpoint_dir"]
    best_model_path = os.path.join(checkpoint_dir, "best_model_hdf.pth")
    full_checkpoint_path = os.path.join(checkpoint_dir, "cf_checkpoint_hdf.pth")

    if os.path.exists(best_model_path):
        return best_model_path
    if os.path.exists(full_checkpoint_path):
        return full_checkpoint_path

    raise FileNotFoundError(
        f"No checkpoint found in '{checkpoint_dir}'. "
        "Expected 'best_model_hdf.pth' or 'cf_checkpoint_hdf.pth'."
    )


def load_trained_model(config, feature_counts, checkpoint_path, device):
    """Load a GNN4CF checkpoint strictly and return an evaluation-mode model."""
    model = build_model_from_config(config, feature_counts, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_rollout_prediction(
    model,
    initial_graph,
    full_length_drivers,
    config,
    feature_counts,
    device,
    y_truth=None,
    debug=False,
    probe_nodes=None,
):
    """Run one autoregressive rollout starting from one cached initial graph."""
    window_cfg = config["window"]
    features_cfg = config["features"]

    predictor_step = int(window_cfg["predictor_step"])
    past_steps = int(window_cfg["past_steps"])
    label_vars = tuple(window_cfg.get("label_vars", ["wd"]))
    driver_vars = tuple(features_cfg.get("dynamic_input_drivers", []))

    n_static_features = feature_counts["n_static_node"]
    n_state_vars = feature_counts["n_state_vars"]
    n_driver_vars = feature_counts["n_driver_vars"]

    if n_driver_vars <= 0:
        raise ValueError("Rollout requires at least one driver variable.")

    total_time_steps = full_length_drivers.shape[1] // n_driver_vars
    if full_length_drivers.shape[1] % n_driver_vars != 0:
        raise ValueError(
            f"full_length_drivers width {full_length_drivers.shape[1]} is not divisible by n_driver_vars={n_driver_vars}."
        )

    future_steps_rollout = total_time_steps - past_steps
    if future_steps_rollout <= 0:
        raise ValueError(
            f"Invalid rollout horizon: total_time_steps={total_time_steps}, past_steps={past_steps}."
        )

    future_drivers = full_length_drivers[:, past_steps * n_driver_vars :]
    y_truth_future = y_truth

    initial_x_full = initial_graph.x.to(device)
    edge_index = initial_graph.edge_index.to(device)
    edge_attr = initial_graph.edge_attr.to(device)
    node_type = initial_graph.node_type.to(device) if hasattr(initial_graph, "node_type") else None
    edge_type = initial_graph.edge_type.to(device) if hasattr(initial_graph, "edge_type") else None
    future_drivers = future_drivers.to(device)

    if y_truth_future is not None:
        y_truth_future = y_truth_future.to(device)

    preds = rollout_autoregressive(
        model=model,
        initial_x_full=initial_x_full,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        edge_type=edge_type,
        future_drivers=future_drivers,
        n_static_features=n_static_features,
        past_steps=past_steps,
        n_state_vars=n_state_vars,
        n_driver_vars=n_driver_vars,
        predictor_step=predictor_step,
        probe_nodes=probe_nodes,
        label_vars=label_vars,
        driver_vars=driver_vars,
        debug=debug,
        y_truth=y_truth_future,
    )
    return preds


def save_event_artifacts(results_dir, entry, preds, y_truth, metrics):
    """Save canonical per-event rollout artifacts and return manifest metadata."""
    os.makedirs(results_dir, exist_ok=True)

    event_name = entry["event_name"]
    sanitized_event_name = sanitize_event_name(entry["sanitized_event_name"])

    preds_path = os.path.join(results_dir, f"{sanitized_event_name}_predictions.pt")
    y_truth_path = os.path.join(results_dir, f"{sanitized_event_name}_ground_truth.pt")
    metrics_path = os.path.join(results_dir, f"{sanitized_event_name}_metrics.json")

    torch.save(preds.cpu(), preds_path)
    torch.save(y_truth.cpu(), y_truth_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "event_name": event_name,
        "sanitized_event_name": sanitized_event_name,
        "initial_graph_index": entry["initial_graph_index"],
        "event_file": entry.get("event_file"),
        "predictions_file": os.path.basename(preds_path),
        "ground_truth_file": os.path.basename(y_truth_path),
        "metrics_file": os.path.basename(metrics_path),
        "pred_shape": list(preds.shape),
        "y_truth_shape": list(y_truth.shape),
        "metrics": metrics,
    }


def save_rollout_manifest(config, results_dir, checkpoint_path, records):
    """Persist a machine-readable manifest for visualization and downstream checks."""
    manifest_path = os.path.join(results_dir, ROLLOUT_MANIFEST_FILENAME)
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_name": config.get("config_name"),
        "config_path": config["paths"].get("config_path"),
        "checkpoint_path": checkpoint_path,
        "results_dir": results_dir,
        "records": records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Run autoregressive rollout inference for a trained GNN4CF model.")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config YAML.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to the resolved rollout_predictions_dir.",
    )
    parser.add_argument("--event-name", type=str, default=None, help="Optional single event name to evaluate.")
    parser.add_argument("--max-events", type=int, default=None, help="Limit the number of cached events used.")
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=None,
        help="Optional cap on future rollout steps, useful for fast smoke tests.",
    )
    parser.add_argument("--rebuild-initial-graphs", action="store_true", help="Rebuild the initial graph cache first.")
    parser.add_argument("--debug", action="store_true", help="Enable rollout debug mode.")
    args = parser.parse_args()

    config = load_resolved_config(args.config)
    feature_counts = get_feature_counts_and_indices_hdf_for_config(config)

    seed = config.get("training", {}).get("seed", 42)
    set_random_seed(seed)

    entries = ensure_initial_test_graph_cache(
        config,
        feature_counts,
        rebuild=args.rebuild_initial_graphs,
        max_events=args.max_events,
    )

    if args.event_name:
        target = sanitize_event_name(args.event_name)
        entries = [
            entry
            for entry in entries
            if entry["event_name"] == args.event_name or entry["sanitized_event_name"] == target
        ]
        if not entries:
            raise ValueError(f"No cached rollout entry matched event '{args.event_name}'.")
    elif args.max_events is not None:
        entries = entries[: int(args.max_events)]

    if not entries:
        raise ValueError("No rollout entries available for inference.")

    results_dir = os.path.abspath(args.results_dir) if args.results_dir else config["paths"]["rollout_predictions_dir"]
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Saving rollout artifacts to: {results_dir}")

    model = load_trained_model(config, feature_counts, checkpoint_path, device)
    label_vars = config["window"].get("label_vars", ["wd"])

    metrics_all = []
    pred_arrays = []
    truth_arrays = []
    manifest_records = []

    for entry in tqdm(entries, desc="Running rollout events"):
        full_length_drivers = entry["full_length_drivers"]
        y_truth_for_rollout = entry["y_truth_full"]
        if args.max_rollout_steps is not None:
            rollout_steps = min(int(args.max_rollout_steps), int(y_truth_for_rollout.shape[1]))
            n_driver_vars = feature_counts["n_driver_vars"]
            past_steps = int(config["window"]["past_steps"])
            full_length_drivers = full_length_drivers[:, : (past_steps + rollout_steps) * n_driver_vars]
            y_truth_for_rollout = y_truth_for_rollout[:, :rollout_steps, :]

        preds = run_rollout_prediction(
            model=model,
            initial_graph=entry["initial_graph"],
            full_length_drivers=full_length_drivers,
            config=config,
            feature_counts=feature_counts,
            device=device,
            y_truth=y_truth_for_rollout,
            debug=args.debug,
        )

        y_truth_for_eval = y_truth_for_rollout
        if preds.shape[0] != y_truth_for_eval.shape[1]:
            target_steps = min(preds.shape[0], y_truth_for_eval.shape[1])
            preds = preds[:target_steps]
            y_truth_for_eval = y_truth_for_eval[:, :target_steps, :]

        metrics = compute_metrics(preds.cpu(), y_truth_for_eval.cpu(), label_vars)
        metrics["event_name"] = entry["event_name"]
        metrics_all.append(metrics)

        record = save_event_artifacts(results_dir, entry, preds, y_truth_for_eval, metrics)
        manifest_records.append(record)

        pred_arrays.append(preds.cpu().numpy())
        truth_arrays.append(y_truth_for_eval.cpu().numpy())

    manifest_path = save_rollout_manifest(config, results_dir, checkpoint_path, manifest_records)

    if pred_arrays and len({tuple(arr.shape) for arr in pred_arrays}) == 1:
        np.save(os.path.join(results_dir, "rollout_predictions.npy"), np.stack(pred_arrays, axis=0))
        np.save(os.path.join(results_dir, "rollout_ground_truth.npy"), np.stack(truth_arrays, axis=0))

    with open(os.path.join(results_dir, "rollout_metrics.pkl"), "wb") as f:
        pickle.dump(metrics_all, f)
    with open(os.path.join(results_dir, "rollout_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)

    print("\nRollout complete.")
    print(f"  events:        {len(entries)}")
    print(f"  manifest:      {manifest_path}")
    print(f"  results dir:   {results_dir}")


if __name__ == "__main__":
    main()
