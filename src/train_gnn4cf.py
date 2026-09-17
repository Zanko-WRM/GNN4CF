# -*- coding: utf-8 -*-

"""
Train GNN4CF using HDF5 graph datasets with optional multi-GPU DDP.

Select event files and grouped train/validation/test splits from a manifest,
reconstruct snapshots lazily, and initialize the boundary-aware model.
Training combines supervised, autoregressive stability, and shallow-depth
loss terms. Run artifacts include checkpoints, logs, the resolved configuration,
and optional W&B tracking.
"""

import argparse
import csv
import glob
import json
import logging
import os
import pickle
import random
import re
import signal
import traceback
from datetime import datetime
from functools import partial

import h5py
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
import yaml

import gnn4cf_training_utils as gnn_utils
from gnn4cf_model import GNNModel
from gnn4cf_training_utils import set_random_seed
from gnn4cf_hdf_graph_dataset import apply_hdf_graph_paths_to_config


def ensure_dir(directory):
    """Create a directory if it does not already exist."""
    if directory:
        os.makedirs(directory, exist_ok=True)


TRAINING_SIGNAL_STATE = {
    "stop_requested": False,
    "signal_name": None,
}


def reset_training_signal_state():
    TRAINING_SIGNAL_STATE["stop_requested"] = False
    TRAINING_SIGNAL_STATE["signal_name"] = None


def _signal_name(signum):
    try:
        return signal.Signals(signum).name
    except Exception:
        return str(signum)


def request_graceful_stop(signum, _frame):
    """Mark the current training run for a graceful stop at the next safe point."""
    TRAINING_SIGNAL_STATE["stop_requested"] = True
    TRAINING_SIGNAL_STATE["signal_name"] = _signal_name(signum)


def install_training_signal_handlers():
    """Install signal handlers that ask training to stop after a safe checkpoint."""
    previous_handlers = {}
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_graceful_stop)
    return previous_handlers


def restore_training_signal_handlers(previous_handlers):
    """Restore original process signal handlers."""
    for signum, handler in (previous_handlers or {}).items():
        signal.signal(signum, handler)


def training_stop_requested():
    return bool(TRAINING_SIGNAL_STATE["stop_requested"])


def get_training_stop_signal_name():
    return TRAINING_SIGNAL_STATE.get("signal_name")


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, path):
    """Save training state to a checkpoint file."""
    model_to_save = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load training state from a checkpoint file."""
    if not os.path.exists(path):
        ddp_print("No checkpoint found. Starting from scratch.")
        return 0, float("inf")

    ddp_print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint load failed for GNN4CF training. "
            "Checkpoints are loaded strictly. If rainfall_conditioning.enabled=true, "
            "the checkpoint must include rainfall_conditioner.* weights. Start "
            "with an empty checkpoint directory or resume from a compatible "
            f"GNN4CF checkpoint. Check the checkpoint at '{path}'. "
            f"Original error: {exc}"
        ) from exc

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]


def count_parameters(model):
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dynamic_input_feature_names(config):
    """Return the per-history-step dynamic node feature order used in HDF x."""
    features_cfg = config.get("features", {}) or {}
    state_vars = list(features_cfg.get("dynamic_input_state_variables", []) or [])
    driver_vars = list(features_cfg.get("dynamic_input_drivers", []) or [])
    dynamic_input_feature_names = state_vars + driver_vars
    if not dynamic_input_feature_names:
        raise ValueError(
            "Could not build dynamic_input_feature_names from "
            "features.dynamic_input_state_variables + features.dynamic_input_drivers."
        )
    return dynamic_input_feature_names


def get_rainfall_conditioning_kwargs(config, dynamic_input_feature_names):
    """
    Translate the simplified dynamic-only rainfall_conditioning block into
    GNNModel constructor kwargs.

    Static context and rainfall history kwargs are intentionally not supported
    here. Static node features are encoded by the node encoder; rainfall FiLM
    reinjects current rainfall_rate and acc_rainfall only.
    """
    rain_cfg = config.get("rainfall_conditioning", {}) or {}
    enabled = bool(rain_cfg.get("enabled", False))

    removed_keys = [
        "use_current",
        "use_history",
        "use_static_context",
        "static_context_features",
        "static_interaction",
        "history_encoder_hidden_dim",
        "static_encoder_hidden_dim",
        "rainfall_static_context_features",
        "rainfall_history_encoder_hidden_dim",
        "rainfall_static_encoder_hidden_dim",
        "rainfall_static_interaction",
        "static_comp_feature_names",
    ]
    present_removed = [key for key in removed_keys if key in rain_cfg]
    if present_removed and is_main_process():
        ddp_print(
            "WARNING: Ignoring unsupported rainfall static/history config keys: "
            f"{present_removed}. Rainfall FiLM uses current dynamic forcing only."
        )

    if enabled and not dynamic_input_feature_names:
        raise ValueError(
            "rainfall_conditioning.enabled=true requires dynamic_input_feature_names "
            "derived from features.dynamic_input_state_variables + features.dynamic_input_drivers."
        )

    return {
        "rainfall_conditioning_enabled": enabled,
        "rainfall_conditioning_mode": str(rain_cfg.get("mode", "film")),
        "rainfall_inject_before_interior": bool(rain_cfg.get("inject_before_interior", True)),
        "rainfall_inject_every_interior_step": bool(rain_cfg.get("inject_every_interior_step", True)),
        "rainfall_features": list(rain_cfg.get("rainfall_features", ["rainfall_rate", "acc_rainfall"]) or []),
        "rainfall_current_encoder_hidden_dim": int(rain_cfg.get("current_encoder_hidden_dim", 64)),
        "rainfall_film_hidden_dim": int(rain_cfg.get("film_hidden_dim", 64)),
        "dynamic_input_feature_names": dynamic_input_feature_names,
    }


def log_model_setup(config, feature_counts, cfg_model, bc_cfg, rainfall_kwargs, dynamic_input_feature_names):
    """Print resolved model settings and optionally record them in W&B."""
    ddp_print("\n--- GNN4CF model construction settings ---")
    ddp_print("  model file: gnn4cf_model.py")
    ddp_print(f"  latent_dim: {cfg_model.get('latent_dim')}")
    ddp_print(f"  mlp_hidden_dim: {cfg_model.get('mlp_hidden_dim')}")
    ddp_print(f"  nmlp_layers: {cfg_model.get('nmlp_layers')}")
    ddp_print(f"  shared message passing steps: {cfg_model.get('nmessage_passing_steps')}")
    ddp_print(f"  interior settings: {cfg_model.get('interior', {}) or {}}")
    ddp_print(f"  coupling settings: {cfg_model.get('coupling', {}) or {}}")
    ddp_print(f"  boundary_conditioning: {bc_cfg}")
    ddp_print(f"  rainfall_conditioning enabled: {rainfall_kwargs['rainfall_conditioning_enabled']}")
    ddp_print(f"  rainfall mode: {rainfall_kwargs['rainfall_conditioning_mode']}")
    ddp_print(f"  rainfall inject_before_interior: {rainfall_kwargs['rainfall_inject_before_interior']}")
    ddp_print(f"  rainfall inject_every_interior_step: {rainfall_kwargs['rainfall_inject_every_interior_step']}")
    ddp_print(f"  rainfall_features: {rainfall_kwargs['rainfall_features']}")
    ddp_print(f"  dynamic_input_feature_names: {dynamic_input_feature_names}")
    ddp_print(f"  number of dynamic node variables: {feature_counts['n_dynamic_node']}")
    ddp_print("--------------------------------------\n")

    config.setdefault("_runtime", {})
    config["_runtime"]["model_file"] = "gnn4cf_model.py"
    config["_runtime"]["dynamic_input_feature_names"] = dynamic_input_feature_names
    config["_runtime"]["rainfall_conditioning_resolved"] = {
        key: value for key, value in rainfall_kwargs.items()
        if key != "dynamic_input_feature_names"
    }

    if is_main_process() and config.get("wandb", {}).get("enabled", False) and wandb.run:
        wandb.config.update(
            {
                "_runtime/model_file": config["_runtime"]["model_file"],
                "_runtime/dynamic_input_feature_names": dynamic_input_feature_names,
                "_runtime/rainfall_conditioning_resolved": config["_runtime"]["rainfall_conditioning_resolved"],
            },
            allow_val_change=True,
        )


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def ddp_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def init_distributed_mode():
    if not torch.cuda.is_available():
        return False, 0, 1, torch.device("cpu")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")

    return distributed, rank, world_size, device


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def barrier():
    if is_distributed():
        dist.barrier()


def reduce_metrics_dict(metrics, device):
    if not is_distributed() or not metrics:
        return metrics

    reduced = {}
    world_size = get_world_size()
    for key, value in metrics.items():
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = (tensor / world_size).item()
    return reduced


def configure_worker_runtime():
    if not is_main_process():
        import builtins

        builtins.print = lambda *args, **kwargs: None
        gnn_utils.tqdm = partial(tqdm, disable=True)
        if hasattr(gnn_utils, "debug_output_ordering"):
            gnn_utils.debug_output_ordering = lambda *args, **kwargs: ({}, {})


def setup_wandb_metric_layout():
    """Define a consistent 5-panel metric namespace for W&B dashboards."""
    if not wandb.run:
        return

    wandb.define_metric("epoch")
    for pattern in [
        "summary/*",
        "optimization/*",
        "branch/*",
        "component/*",
        "performance/*",
        "flood/*",
    ]:
        wandb.define_metric(pattern, step_metric="epoch")


def update_wandb_dataset_summary(selection_info, split_summary, use_grouped_split):
    """Store dataset/split metadata in the active W&B run summary."""
    if not wandb.run:
        return

    if selection_info:
        wandb.run.summary["dataset/selected_event_files"] = len(
            selection_info.get("selected_event_files", [])
        )
        wandb.run.summary["dataset/selected_groups"] = len(
            selection_info.get("selected_groups", [])
        )

    if use_grouped_split and split_summary:
        wandb.run.summary["dataset/split_strategy"] = "grouped_event"
        wandb.run.summary["dataset/train_group_count"] = split_summary.get("train_group_count", 0)
        wandb.run.summary["dataset/val_group_count"] = split_summary.get("val_group_count", 0)
        wandb.run.summary["dataset/test_group_count"] = split_summary.get("test_group_count", 0)
        wandb.run.summary["dataset/train_event_count"] = split_summary.get("train_event_count", 0)
        wandb.run.summary["dataset/val_event_count"] = split_summary.get("val_event_count", 0)
        wandb.run.summary["dataset/test_event_count"] = split_summary.get("test_event_count", 0)


def build_wandb_log_metrics(
    *,
    epoch,
    current_lr,
    config,
    train_metrics,
    val_metrics,
    val_loss_key_prefix,
):
    """Assemble a W&B payload grouped by optimization / branch / component / performance / flood."""
    loss_weights = config.get("loss", {}).get("loss_weights", {})
    flood_aware_cfg = config.get("loss", {}).get("flood_aware", {})
    stability_enabled = float(loss_weights.get("stability", 0.0)) > 0.0
    classification_enabled = bool(
        flood_aware_cfg.get("enabled", False)
        and flood_aware_cfg.get("classification", {}).get("enabled", False)
    )
    shallow_cfg = config.get("loss", {}).get("shallow_depth_loss", {})
    shallow_depth_enabled = bool(shallow_cfg.get("enabled", False))
    log_metrics = {
        "epoch": epoch,
        "optimization/learning_rate": current_lr,
    }

    if "total_loss" in train_metrics:
        log_metrics["optimization/train_total"] = train_metrics["total_loss"]
        log_metrics["summary/train_loss"] = train_metrics["total_loss"]
    if val_metrics and "total_loss" in val_metrics:
        log_metrics["optimization/val_total"] = val_metrics["total_loss"]
        log_metrics["summary/val_loss"] = val_metrics["total_loss"]

    for branch_name in ["real", "stability"]:
        if branch_name == "stability" and not stability_enabled:
            continue
        total_key = f"{branch_name}_total"
        main_key = f"{branch_name}_main_loss"
        shallow_term_key = f"{branch_name}_shallow_depth_term"
        shallow_raw_key = f"{branch_name}_shallow_depth_loss"
        shallow_fraction_key = f"{branch_name}_shallow_depth_active_fraction"
        shallow_count_key = f"{branch_name}_shallow_depth_active_count"
        shallow_rmse_key = f"{branch_name}_shallow_depth_rmse_m"
        shallow_mae_key = f"{branch_name}_shallow_depth_mae_m"
        cls_term_key = f"{branch_name}_classification_term"
        cls_raw_key = f"{branch_name}_classification_loss"
        cls_acc_key = f"{branch_name}_classification_accuracy"
        cls_precision_key = f"{branch_name}_classification_precision"
        cls_recall_key = f"{branch_name}_classification_recall"
        cls_f1_key = f"{branch_name}_classification_f1"

        if total_key in train_metrics:
            log_metrics[f"branch/train/{branch_name}_total"] = train_metrics[total_key]
        if val_metrics and branch_name == val_loss_key_prefix and total_key in val_metrics:
            log_metrics[f"branch/val/{branch_name}_total"] = val_metrics[total_key]

        if main_key in train_metrics:
            log_metrics[f"component/train/{branch_name}_main"] = train_metrics[main_key]
        if val_metrics and branch_name == val_loss_key_prefix and main_key in val_metrics:
            log_metrics[f"component/val/{branch_name}_main"] = val_metrics[main_key]

        if shallow_depth_enabled and shallow_term_key in train_metrics:
            log_metrics[f"component/train/{branch_name}_shallow_depth_term"] = train_metrics[shallow_term_key]
        if shallow_depth_enabled and val_metrics and branch_name == val_loss_key_prefix and shallow_term_key in val_metrics:
            log_metrics[f"component/val/{branch_name}_shallow_depth_term"] = val_metrics[shallow_term_key]

        if shallow_depth_enabled and shallow_raw_key in train_metrics:
            log_metrics[f"component/train/{branch_name}_shallow_depth_raw"] = train_metrics[shallow_raw_key]
        if shallow_depth_enabled and val_metrics and branch_name == val_loss_key_prefix and shallow_raw_key in val_metrics:
            log_metrics[f"component/val/{branch_name}_shallow_depth_raw"] = val_metrics[shallow_raw_key]

        for shallow_metric_key, shallow_metric_name in [
            (shallow_fraction_key, "shallow_depth_active_fraction"),
            (shallow_count_key, "shallow_depth_active_count"),
            (shallow_rmse_key, "shallow_depth_rmse_m"),
            (shallow_mae_key, "shallow_depth_mae_m"),
        ]:
            if shallow_depth_enabled and shallow_metric_key in train_metrics:
                log_metrics[f"performance/train/{branch_name}_{shallow_metric_name}"] = train_metrics[
                    shallow_metric_key
                ]
            if (
                shallow_depth_enabled
                and val_metrics
                and branch_name == val_loss_key_prefix
                and shallow_metric_key in val_metrics
            ):
                log_metrics[f"performance/val/{branch_name}_{shallow_metric_name}"] = val_metrics[
                    shallow_metric_key
                ]

        if classification_enabled and cls_term_key in train_metrics:
            log_metrics[f"component/train/{branch_name}_classification_term"] = train_metrics[cls_term_key]
        if classification_enabled and val_metrics and branch_name == val_loss_key_prefix and cls_term_key in val_metrics:
            log_metrics[f"component/val/{branch_name}_classification_term"] = val_metrics[cls_term_key]

        if classification_enabled and cls_raw_key in train_metrics:
            log_metrics[f"component/train/{branch_name}_classification_raw"] = train_metrics[cls_raw_key]
        if classification_enabled and val_metrics and branch_name == val_loss_key_prefix and cls_raw_key in val_metrics:
            log_metrics[f"component/val/{branch_name}_classification_raw"] = val_metrics[cls_raw_key]

        if classification_enabled and cls_acc_key in train_metrics:
            log_metrics[f"performance/train/{branch_name}_classification_accuracy"] = train_metrics[cls_acc_key]
        if classification_enabled and val_metrics and branch_name == val_loss_key_prefix and cls_acc_key in val_metrics:
            log_metrics[f"performance/val/{branch_name}_classification_accuracy"] = val_metrics[cls_acc_key]
        if classification_enabled and cls_precision_key in train_metrics:
            log_metrics[f"performance/train/{branch_name}_classification_precision"] = train_metrics[cls_precision_key]
        if classification_enabled and val_metrics and branch_name == val_loss_key_prefix and cls_precision_key in val_metrics:
            log_metrics[f"performance/val/{branch_name}_classification_precision"] = val_metrics[cls_precision_key]
        if classification_enabled and cls_recall_key in train_metrics:
            log_metrics[f"performance/train/{branch_name}_classification_recall"] = train_metrics[cls_recall_key]
        if classification_enabled and val_metrics and branch_name == val_loss_key_prefix and cls_recall_key in val_metrics:
            log_metrics[f"performance/val/{branch_name}_classification_recall"] = val_metrics[cls_recall_key]
        if classification_enabled and cls_f1_key in train_metrics:
            log_metrics[f"performance/train/{branch_name}_classification_f1"] = train_metrics[cls_f1_key]
        if classification_enabled and val_metrics and branch_name == val_loss_key_prefix and cls_f1_key in val_metrics:
            log_metrics[f"performance/val/{branch_name}_classification_f1"] = val_metrics[cls_f1_key]

    if val_metrics:
        selected_total_key = f"{val_loss_key_prefix}_total"
        if selected_total_key in val_metrics:
            log_metrics["branch/val/selected_total"] = val_metrics[selected_total_key]

    for branch_name in ["real", "stability"]:
        for metric_name in [
            "rmse_wd_normalized",
            "mae_wd_normalized",
            "rmse_wd_m",
            "mae_wd_m",
            "rmse_wd_wet_normalized",
            "mae_wd_wet_normalized",
            "rmse_wd_wet_m",
            "mae_wd_wet_m",
        ]:
            train_key = f"{branch_name}_{metric_name}"
            if train_key in train_metrics:
                log_metrics[f"performance/train/{branch_name}_{metric_name}"] = train_metrics[train_key]
            if val_metrics and branch_name == val_loss_key_prefix and train_key in val_metrics:
                log_metrics[f"performance/val/{branch_name}_{metric_name}"] = val_metrics[train_key]

    if val_metrics:
        selected_rmse_norm_key = f"{val_loss_key_prefix}_rmse_wd_normalized"
        selected_mae_norm_key = f"{val_loss_key_prefix}_mae_wd_normalized"
        selected_rmse_m_key = f"{val_loss_key_prefix}_rmse_wd_m"
        selected_mae_m_key = f"{val_loss_key_prefix}_mae_wd_m"
        selected_rmse_wet_norm_key = f"{val_loss_key_prefix}_rmse_wd_wet_normalized"
        selected_mae_wet_norm_key = f"{val_loss_key_prefix}_mae_wd_wet_normalized"
        selected_rmse_wet_m_key = f"{val_loss_key_prefix}_rmse_wd_wet_m"
        selected_mae_wet_m_key = f"{val_loss_key_prefix}_mae_wd_wet_m"
        if selected_rmse_norm_key in val_metrics:
            log_metrics["performance/val/selected_rmse_wd_normalized"] = val_metrics[selected_rmse_norm_key]
            log_metrics["summary/val_rmse_wd_normalized"] = val_metrics[selected_rmse_norm_key]
        if selected_mae_norm_key in val_metrics:
            log_metrics["performance/val/selected_mae_wd_normalized"] = val_metrics[selected_mae_norm_key]
        if selected_rmse_m_key in val_metrics:
            log_metrics["performance/val/selected_rmse_wd_m"] = val_metrics[selected_rmse_m_key]
            log_metrics["summary/val_rmse_wd_m"] = val_metrics[selected_rmse_m_key]
        if selected_mae_m_key in val_metrics:
            log_metrics["performance/val/selected_mae_wd_m"] = val_metrics[selected_mae_m_key]
        if selected_rmse_wet_norm_key in val_metrics:
            log_metrics["performance/val/selected_rmse_wd_wet_normalized"] = val_metrics[selected_rmse_wet_norm_key]
        if selected_mae_wet_norm_key in val_metrics:
            log_metrics["performance/val/selected_mae_wd_wet_normalized"] = val_metrics[selected_mae_wet_norm_key]
        if selected_rmse_wet_m_key in val_metrics:
            log_metrics["performance/val/selected_rmse_wd_wet_m"] = val_metrics[selected_rmse_wet_m_key]
            log_metrics["summary/val_rmse_wd_wet_m"] = val_metrics[selected_rmse_wet_m_key]
        if selected_mae_wet_m_key in val_metrics:
            log_metrics["performance/val/selected_mae_wd_wet_m"] = val_metrics[selected_mae_wet_m_key]

    for split_name, metrics_dict in [("train", train_metrics), ("val", val_metrics or {})]:
        if not metrics_dict:
            continue

        for var in config.get("window", {}).get("label_vars", ["wd", "vx", "vy"]):
            total_var_key = f"total_loss_{var}"
            if total_var_key in metrics_dict:
                log_metrics[f"component/{split_name}/total_loss_{var}"] = metrics_dict[total_var_key]
            real_var_key = f"real_loss_{var}"
            if real_var_key in metrics_dict:
                log_metrics[f"component/{split_name}/real_loss_{var}"] = metrics_dict[real_var_key]
            stab_var_key = f"stability_loss_{var}"
            if stab_var_key in metrics_dict:
                log_metrics[f"component/{split_name}/stability_loss_{var}"] = metrics_dict[stab_var_key]

        for flood_key in [
            "flooded_nodes_pct",
            "flood_precision",
            "flood_recall",
            "flood_f1",
            "mean_flood_depth_normalized",
            "max_flood_depth_normalized",
            "mean_depth_all_normalized",
        ]:
            if flood_key in metrics_dict:
                log_metrics[f"flood/{split_name}/{flood_key}"] = metrics_dict[flood_key]

    return log_metrics


def resolve_per_gpu_batch_size(global_batch_size):
    world_size = get_world_size()
    if world_size <= 1:
        return int(global_batch_size)

    if global_batch_size < world_size:
        raise ValueError(
            f"Global batch size ({global_batch_size}) is smaller than world size ({world_size}). "
            "Increase batch size or reduce the number of GPUs."
        )

    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size ({global_batch_size}) must be divisible by world size ({world_size}) "
            "for this multi-GPU script."
        )

    return int(global_batch_size // world_size)


def load_stats_file(file_path):
    """Load normalization statistics if the JSON file exists."""
    if not os.path.exists(file_path):
        ddp_print(f"Statistics file not found at {file_path}.")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_event_name_from_hdf_filename(filename):
    """Extract `(bc_name, event_name)` from an HDF event filename."""
    base_name = os.path.basename(filename)
    base_name = re.sub(r"_snapshots\.h5$", "", base_name)
    base_name = re.sub(r"\.h5$", "", base_name)
    base_name = re.sub(r"\.hdf$", "", base_name)
    base_name = re.sub(r"^Flood_model\.p\d+_", "", base_name)

    parts = base_name.split("_")
    if len(parts) >= 3 and parts[0].startswith("BC") and parts[0][2:].isdigit():
        return parts[0], base_name
    return None, base_name


STRUCTURED_EVENT_RE = re.compile(r"^(BC\d+)_S(\d+)_R(\d+)$")


def parse_structured_event_from_path(file_path):
    """Parse `BCx_Sn_Rm` event names into structured components."""
    bc_name, event_name = parse_event_name_from_hdf_filename(file_path)
    match = STRUCTURED_EVENT_RE.match(event_name)
    if not match:
        return None

    parsed_bc, s_idx, r_idx = match.groups()
    s_idx = int(s_idx)
    r_idx = int(r_idx)
    pair_index = s_idx if s_idx == r_idx else None
    return {
        "bc_name": parsed_bc,
        "event_name": event_name,
        "file_path": file_path,
        "s_idx": s_idx,
        "r_idx": r_idx,
        "pair_index": pair_index,
        "group_key": f"S{s_idx}_R{r_idx}",
    }


def build_file_signatures(file_paths):
    """Build a stable file-signature list for cache validation."""
    signatures = []
    for file_path in sorted(set(file_paths)):
        file_path = os.path.abspath(file_path)
        stat = os.stat(file_path)
        signatures.append(
            {
                "path": file_path,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return signatures


def file_signatures_match(expected_signatures, current_files):
    """Return `True` when cached signatures still match the current files."""
    if expected_signatures is None:
        return False
    try:
        return expected_signatures == build_file_signatures(current_files)
    except FileNotFoundError:
        return False


def group_structured_event_files(event_files, bc_names=None):
    """Group parsed event files by `(S,R)` so paired BC variants stay together."""
    bc_names = bc_names or ["BC1", "BC2", "BC3"]
    grouped = {}
    skipped_files = []

    for file_path in event_files:
        parsed = parse_structured_event_from_path(file_path)
        if parsed is None:
            skipped_files.append(file_path)
            continue

        group = grouped.setdefault(
            parsed["group_key"],
            {
                "group_key": parsed["group_key"],
                "s_idx": parsed["s_idx"],
                "r_idx": parsed["r_idx"],
                "pair_index": parsed["pair_index"],
                "files": {},
                "event_names": {},
            },
        )
        group["files"][parsed["bc_name"]] = file_path
        group["event_names"][parsed["bc_name"]] = parsed["event_name"]

    ordered_groups = sorted(
        grouped.values(),
        key=lambda item: (
            item["pair_index"] if item["pair_index"] is not None else 10**9,
            item["s_idx"],
            item["r_idx"],
        ),
    )
    return ordered_groups, skipped_files


def select_hdf_event_files_from_config(snapshot_base_dir, config):
    """
    Select a subset of event files from the HDF snapshot directory based on config.

    Supported mode:
      event_selection.mode = paired_bc_groups
    """
    all_event_files = list_hdf_event_files(snapshot_base_dir)
    selection_cfg = config.get("event_selection", {}) or {}
    if not selection_cfg.get("enabled", False):
        return {
            "selected_event_files": all_event_files,
            "selected_groups": [],
            "selection_enabled": False,
            "all_event_files": all_event_files,
        }

    mode = selection_cfg.get("mode", "paired_bc_groups")
    if mode != "paired_bc_groups":
        raise ValueError(
            f"Unsupported event_selection.mode '{mode}'. "
            "Expected 'paired_bc_groups'."
        )

    bc_names = selection_cfg.get("bc_names", ["BC1", "BC2", "BC3"])
    require_complete_group = bool(selection_cfg.get("require_complete_group", True))
    require_matching_s_r = bool(selection_cfg.get("require_matching_s_r", True))
    min_pair_index = int(selection_cfg.get("min_pair_index", 1))
    max_pair_index = selection_cfg.get("max_pair_index", None)
    max_pair_index = int(max_pair_index) if max_pair_index is not None else None

    grouped_events, skipped_files = group_structured_event_files(all_event_files, bc_names=bc_names)

    selected_groups = []
    incomplete_groups = []
    for group in grouped_events:
        if require_matching_s_r and group["pair_index"] is None:
            continue
        if group["pair_index"] is not None:
            if group["pair_index"] < min_pair_index:
                continue
            if max_pair_index is not None and group["pair_index"] > max_pair_index:
                continue

        missing_bcs = [bc_name for bc_name in bc_names if bc_name not in group["files"]]
        if require_complete_group and missing_bcs:
            incomplete_groups.append(
                {
                    "group_key": group["group_key"],
                    "missing_bcs": missing_bcs,
                }
            )
            continue

        selected_groups.append(group)

    selected_event_files = []
    for group in selected_groups:
        for bc_name in bc_names:
            file_path = group["files"].get(bc_name)
            if file_path:
                selected_event_files.append(file_path)

    return {
        "selected_event_files": selected_event_files,
        "selected_groups": selected_groups,
        "selection_enabled": True,
        "all_event_files": all_event_files,
        "bc_names": bc_names,
        "skipped_files": skipped_files,
        "incomplete_groups": incomplete_groups,
        "min_pair_index": min_pair_index,
        "max_pair_index": max_pair_index,
    }


def build_grouped_event_split_summary(train_groups, val_groups, test_groups, bc_names):
    """Assemble a lightweight summary dict for grouped-event split reporting."""
    return {
        "train_groups": [group["group_key"] for group in train_groups],
        "val_groups": [group["group_key"] for group in val_groups],
        "test_groups": [group["group_key"] for group in test_groups],
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "test_group_count": len(test_groups),
        "train_event_count": sum(len(group["files"]) for group in train_groups),
        "val_event_count": sum(len(group["files"]) for group in val_groups),
        "test_event_count": sum(len(group["files"]) for group in test_groups),
        "bc_names": list(bc_names),
    }


def save_grouped_test_metadata_hdf(test_meta_path, split_summary, split_cfg, selection_info, test_event_files):
    """Persist grouped-event test metadata for rollout/test scripts."""
    ensure_dir(os.path.dirname(test_meta_path))
    test_event_names = [
        parse_event_name_from_hdf_filename(file_path)[1]
        for file_path in sorted(test_event_files)
    ]
    metadata = {
        "created_date": datetime.now().isoformat(),
        "split_strategy": "grouped_event",
        "split_mode": split_cfg.get("mode", "random_grouped"),
        "split_manifest_path": split_cfg.get("split_manifest_path"),
        "group_key_column": split_cfg.get("group_key_column"),
        "split_column": split_cfg.get("split_column"),
        "seed": int(split_cfg.get("seed", 42)),
        "shuffle_groups": bool(split_cfg.get("shuffle_groups", True)),
        "test_event_names": test_event_names,
        "n_test_events": len(test_event_names),
        "group_summary": split_summary,
        "event_selection": {
            "enabled": bool(selection_info.get("selection_enabled", False)),
            "bc_names": selection_info.get("bc_names", ["BC1", "BC2", "BC3"]),
            "min_pair_index": selection_info.get("min_pair_index"),
            "max_pair_index": selection_info.get("max_pair_index"),
        },
        "selected_event_file_signatures": build_file_signatures(selection_info["selected_event_files"]),
    }
    with open(test_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _resolve_split_manifest_path(split_cfg):
    """Resolve and validate a grouped split manifest CSV path."""
    manifest_path = split_cfg.get("split_manifest_path", split_cfg.get("manifest_path"))
    if not manifest_path:
        raise ValueError(
            "training.grouped_event_split.mode='manifest' requires "
            "training.grouped_event_split.split_manifest_path."
        )

    manifest_path = os.path.abspath(os.path.expanduser(str(manifest_path)))
    if os.path.isdir(manifest_path):
        default_name = split_cfg.get(
            "split_manifest_filename",
            "proposed_stratified_split_220_40_40_groups.csv",
        )
        manifest_path = os.path.join(manifest_path, default_name)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Grouped split manifest not found: {manifest_path}")
    return manifest_path


def load_grouped_split_manifest(split_cfg):
    """
    Load a group-level train/val/test split manifest.

    The manifest must contain a group key column, e.g. `group_key`, and a split
    column, e.g. `proposed_split_220_40_40`, with values `train`, `val`, `test`.
    Other values such as `reserve` or `ood_holdout` are ignored for normal
    training.
    """
    manifest_path = _resolve_split_manifest_path(split_cfg)
    group_key_column = split_cfg.get("group_key_column", "group_key")
    split_column = split_cfg.get("split_column", "split")
    allowed_splits = {"train", "val", "test"}

    rows = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if group_key_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Grouped split manifest {manifest_path} is missing group key column "
                f"'{group_key_column}'. Columns: {reader.fieldnames}"
            )
        if split_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Grouped split manifest {manifest_path} is missing split column "
                f"'{split_column}'. Columns: {reader.fieldnames}"
            )

        seen = {}
        for line_no, row in enumerate(reader, start=2):
            group_key = str(row.get(group_key_column, "")).strip()
            split_name = str(row.get(split_column, "")).strip().lower()
            if not group_key:
                raise ValueError(f"Empty group key in {manifest_path} at line {line_no}")
            if split_name not in allowed_splits:
                continue
            if group_key in seen:
                raise ValueError(
                    f"Duplicate train/val/test group '{group_key}' in {manifest_path}; "
                    f"first seen at line {seen[group_key]}, again at line {line_no}."
                )
            seen[group_key] = line_no
            rows.append(
                {
                    "group_key": group_key,
                    "split": split_name,
                    "line_no": line_no,
                }
            )

    if not rows:
        raise ValueError(
            f"Grouped split manifest {manifest_path} did not contain any train/val/test rows "
            f"in column '{split_column}'."
        )

    return {
        "manifest_path": manifest_path,
        "group_key_column": group_key_column,
        "split_column": split_column,
        "rows": rows,
    }


def split_groups_from_manifest(selected_groups, split_cfg, bc_names):
    """Select train/val/test groups according to an external group split CSV."""
    manifest_info = load_grouped_split_manifest(split_cfg)
    groups_by_key = {group["group_key"]: group for group in selected_groups}
    train_groups = []
    val_groups = []
    test_groups = []
    missing_groups = []

    split_to_target = {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }

    for row in manifest_info["rows"]:
        group_key = row["group_key"]
        group = groups_by_key.get(group_key)
        if group is None:
            missing_groups.append(group_key)
            continue
        missing_bcs = [bc_name for bc_name in bc_names if bc_name not in group["files"]]
        if missing_bcs:
            raise ValueError(
                f"Manifest group '{group_key}' is missing required BC files: {missing_bcs}"
            )
        split_to_target[row["split"]].append(group)

    if missing_groups:
        preview = ", ".join(missing_groups[:20])
        if len(missing_groups) > 20:
            preview += f", ... ({len(missing_groups)} total missing)"
        raise ValueError(
            "Some groups from the split manifest are not available after event_selection "
            f"filtering or in the snapshot directory: {preview}. "
            "Check event_selection.max_pair_index/min_pair_index, bc_names, and graph snapshots."
        )

    expected_counts = {
        "train": split_cfg.get("train_groups"),
        "val": split_cfg.get("val_groups"),
        "test": split_cfg.get("test_groups"),
    }
    observed_counts = {
        "train": len(train_groups),
        "val": len(val_groups),
        "test": len(test_groups),
    }
    for split_name, expected in expected_counts.items():
        if expected is None:
            continue
        expected = int(expected)
        if expected > 0 and observed_counts[split_name] != expected:
            raise ValueError(
                f"Manifest split '{split_name}' has {observed_counts[split_name]} groups, "
                f"but config requested {expected}."
            )

    ddp_print("\n--- Using Manifest-Driven Grouped Event Split ---")
    ddp_print(f"  Manifest: {manifest_info['manifest_path']}")
    ddp_print(f"  Group key column: {manifest_info['group_key_column']}")
    ddp_print(f"  Split column: {manifest_info['split_column']}")
    ddp_print(
        f"  Groups -> train: {len(train_groups)}, val: {len(val_groups)}, test: {len(test_groups)}"
    )

    return train_groups, val_groups, test_groups, manifest_info


def load_or_create_grouped_event_manifests_hdf(config, snapshot_base_dir, selection_info):
    """Build train/val manifests from grouped `(S,R)` event splits."""
    cfg_train = config["training"]
    cfg_paths = config["paths"]
    split_cfg = cfg_train.get("grouped_event_split", {}) or {}
    if not split_cfg.get("enabled", False):
        raise ValueError("Grouped-event split requested without enabling training.grouped_event_split.")

    bc_names = split_cfg.get(
        "bc_names",
        selection_info.get("bc_names", ["BC1", "BC2", "BC3"]),
    )
    selected_groups = selection_info.get("selected_groups", [])
    if not selected_groups and selection_info.get("all_event_files"):
        selected_groups, skipped_files = group_structured_event_files(
            selection_info["all_event_files"],
            bc_names=bc_names,
        )
        selection_info["selected_groups"] = selected_groups
        selection_info["selected_event_files"] = [
            group["files"][bc_name]
            for group in selected_groups
            for bc_name in bc_names
            if bc_name in group["files"]
        ]
        selection_info["bc_names"] = bc_names
        selection_info["skipped_files"] = skipped_files

    if not selected_groups:
        raise ValueError(
            "No grouped HDF events were selected. Check event_selection and source HDF files."
        )

    split_mode = split_cfg.get("mode", "random_grouped")
    shuffle_groups = bool(split_cfg.get("shuffle_groups", True))
    split_seed = int(split_cfg.get("seed", cfg_train.get("seed", 42)))
    manifest_info = None

    if split_mode == "manifest":
        train_groups, val_groups, test_groups, manifest_info = split_groups_from_manifest(
            selected_groups,
            split_cfg,
            bc_names,
        )
        train_group_count = len(train_groups)
        val_group_count = len(val_groups)
        test_group_count = len(test_groups)
    elif split_mode in ("random_grouped", "random", "grouped_random"):
        train_group_count = int(split_cfg.get("train_groups", 0))
        val_group_count = int(split_cfg.get("val_groups", 0))
        test_group_count = int(split_cfg.get("test_groups", 0))
        required_group_count = train_group_count + val_group_count + test_group_count
        if required_group_count <= 0:
            raise ValueError("Grouped-event split requires positive train/val/test group counts.")
        if len(selected_groups) < required_group_count:
            raise ValueError(
                f"Grouped-event split requires {required_group_count} groups, but only "
                f"{len(selected_groups)} complete groups are available after selection."
            )

        ordered_groups = list(selected_groups)
        if shuffle_groups:
            rng = random.Random(split_seed)
            rng.shuffle(ordered_groups)

        ordered_groups = ordered_groups[:required_group_count]
        train_groups = ordered_groups[:train_group_count]
        val_groups = ordered_groups[train_group_count:train_group_count + val_group_count]
        test_groups = ordered_groups[train_group_count + val_group_count:required_group_count]
    else:
        raise ValueError(
            f"Unsupported training.grouped_event_split.mode '{split_mode}'. "
            "Expected 'random_grouped' or 'manifest'."
        )

    train_event_files = [
        group["files"][bc_name]
        for group in train_groups
        for bc_name in bc_names
        if bc_name in group["files"]
    ]
    val_event_files = [
        group["files"][bc_name]
        for group in val_groups
        for bc_name in bc_names
        if bc_name in group["files"]
    ]
    test_event_files = [
        group["files"][bc_name]
        for group in test_groups
        for bc_name in bc_names
        if bc_name in group["files"]
    ]

    split_summary = build_grouped_event_split_summary(train_groups, val_groups, test_groups, bc_names)

    train_val_cache_dir = cfg_paths.get(
        "train_val_cache_dir",
        os.path.join(cfg_paths.get("output_dir", "output_data"), "train_val_cache"),
    )
    train_cache_path = os.path.join(train_val_cache_dir, "train_manifest_hdf.pkl")
    val_cache_path = os.path.join(train_val_cache_dir, "val_manifest_hdf.pkl")
    cache_meta_path = os.path.join(train_val_cache_dir, "cache_metadata_hdf.json")
    use_cached_split = cfg_train.get("use_cached_train_val", False)

    test_output_dir = cfg_paths.get(
        "test_snapshot_dir",
        os.path.join(cfg_paths.get("output_dir", "output_data"), "test_snapshots"),
    )
    test_meta_path = os.path.join(test_output_dir, "test_extraction_metadata_hdf.json")

    expected_cache_metadata = {
        "split_strategy": "grouped_event",
        "split_mode": split_mode,
        "seed": split_seed,
        "shuffle_groups": shuffle_groups,
        "train_groups": train_group_count,
        "val_groups": val_group_count,
        "test_groups": test_group_count,
        "bc_names": list(bc_names),
        "split_manifest_path": manifest_info["manifest_path"] if manifest_info else None,
        "group_key_column": manifest_info["group_key_column"] if manifest_info else None,
        "split_column": manifest_info["split_column"] if manifest_info else None,
        "split_manifest_file_signatures": (
            build_file_signatures([manifest_info["manifest_path"]]) if manifest_info else None
        ),
        "selected_event_file_signatures": build_file_signatures(selection_info["selected_event_files"]),
        "train_event_file_signatures": build_file_signatures(train_event_files),
        "val_event_file_signatures": build_file_signatures(val_event_files),
        "test_event_file_signatures": build_file_signatures(test_event_files),
        "group_summary": split_summary,
    }

    train_manifest = None
    val_manifest = None

    if use_cached_split and os.path.exists(train_cache_path) and os.path.exists(val_cache_path) and os.path.exists(cache_meta_path):
        ddp_print("\n--- Loading Cached Grouped HDF Train/Val Manifests ---")
        ddp_print(f"  Cache directory: {train_val_cache_dir}")
        try:
            with open(cache_meta_path, "r", encoding="utf-8") as f:
                cache_meta = json.load(f)

            cache_valid = (
                cache_meta.get("split_strategy") == "grouped_event"
                and cache_meta.get("split_mode") == expected_cache_metadata["split_mode"]
                and cache_meta.get("seed") == expected_cache_metadata["seed"]
                and cache_meta.get("shuffle_groups") == expected_cache_metadata["shuffle_groups"]
                and cache_meta.get("train_groups") == expected_cache_metadata["train_groups"]
                and cache_meta.get("val_groups") == expected_cache_metadata["val_groups"]
                and cache_meta.get("test_groups") == expected_cache_metadata["test_groups"]
                and cache_meta.get("bc_names") == expected_cache_metadata["bc_names"]
                and cache_meta.get("split_manifest_path") == expected_cache_metadata["split_manifest_path"]
                and cache_meta.get("group_key_column") == expected_cache_metadata["group_key_column"]
                and cache_meta.get("split_column") == expected_cache_metadata["split_column"]
                and (
                    expected_cache_metadata["split_manifest_file_signatures"] is None
                    or file_signatures_match(
                        cache_meta.get("split_manifest_file_signatures"),
                        [expected_cache_metadata["split_manifest_path"]],
                    )
                )
                and file_signatures_match(
                    cache_meta.get("selected_event_file_signatures"),
                    selection_info["selected_event_files"],
                )
                and file_signatures_match(
                    cache_meta.get("train_event_file_signatures"),
                    train_event_files,
                )
                and file_signatures_match(
                    cache_meta.get("val_event_file_signatures"),
                    val_event_files,
                )
                and file_signatures_match(
                    cache_meta.get("test_event_file_signatures"),
                    test_event_files,
                )
            )

            if cache_valid:
                with open(train_cache_path, "rb") as f:
                    train_manifest = pickle.load(f)
                with open(val_cache_path, "rb") as f:
                    val_manifest = pickle.load(f)
                ddp_print(
                    f"  ✅ Loaded {len(train_manifest)} train samples and {len(val_manifest)} val samples"
                )
            else:
                ddp_print("  ℹ️  Cached grouped manifests are stale. Rebuilding them now.")
        except Exception as exc:
            ddp_print(f"  ⚠️  Failed to load grouped manifest cache: {exc}")
            train_manifest = None
            val_manifest = None

    if train_manifest is None or val_manifest is None:
        ddp_print("\n--- Creating Grouped HDF Train/Val Manifest Split ---")
        ddp_print(
            f"  Groups -> train: {train_group_count}, val: {val_group_count}, test: {test_group_count}"
        )
        train_manifest = build_hdf_snapshot_manifest(snapshot_base_dir, event_files=train_event_files)
        val_manifest = build_hdf_snapshot_manifest(snapshot_base_dir, event_files=val_event_files)

        manifest_to_check = train_manifest or val_manifest
        if manifest_to_check:
            ddp_print("\n--- First indexed HDF snapshot ---")
            first_graph = HDFSnapshotDataset([manifest_to_check[0]])[0]
            x_has_nan = torch.isnan(first_graph.x).any()
            edge_has_nan = torch.isnan(first_graph.edge_attr).any()
            y_has_nan = torch.isnan(first_graph.y).any()
            ddp_print(f"  Snapshot 'x' has NaN: {x_has_nan}")
            ddp_print(f"  Snapshot 'edge_attr' has NaN: {edge_has_nan}")
            ddp_print(f"  Snapshot 'y' has NaN: {y_has_nan}")
            if x_has_nan or edge_has_nan:
                raise ValueError("NaN found in HDF data loaded from disk. Regenerate HDF snapshots.")
            ddp_print("-----------------------------------------------\n")

        if use_cached_split and is_main_process():
            ensure_dir(train_val_cache_dir)
            with open(train_cache_path, "wb") as f:
                pickle.dump(train_manifest, f)
            with open(val_cache_path, "wb") as f:
                pickle.dump(val_manifest, f)

            cache_metadata = dict(expected_cache_metadata)
            cache_metadata.update(
                {
                    "created_date": datetime.now().isoformat(),
                    "n_train": len(train_manifest),
                    "n_val": len(val_manifest),
                }
            )
            with open(cache_meta_path, "w", encoding="utf-8") as f:
                json.dump(cache_metadata, f, indent=2)
            ddp_print(
                f"  ✅ Saved {len(train_manifest)} train samples and {len(val_manifest)} val samples"
            )

    if is_main_process():
        save_grouped_test_metadata_hdf(
            test_meta_path=test_meta_path,
            split_summary=split_summary,
            split_cfg=split_cfg,
            selection_info=selection_info,
            test_event_files=test_event_files,
        )
    barrier()

    return {
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "train_event_files": train_event_files,
        "val_event_files": val_event_files,
        "test_event_files": test_event_files,
        "summary": split_summary,
    }


def list_hdf_event_files(snapshot_base_dir):
    """List all event HDF files under the snapshot directory."""
    pattern = os.path.join(snapshot_base_dir, "*_snapshots.h5")
    return sorted(glob.glob(pattern))


def load_snapshot_from_hdf(file_path, snapshot_idx):
    """Load one snapshot window from an event HDF file as a PyG `Data` object."""
    with h5py.File(file_path, "r") as h5f:
        x_static = torch.from_numpy(h5f["static"]["x_static"][...]).float()
        edge_index = torch.from_numpy(h5f["static"]["edge_index"][...]).long()
        edge_attr = torch.from_numpy(h5f["static"]["edge_attr"][...]).float()
        node_type = torch.from_numpy(h5f["static"]["node_type"][...]).long()
        edge_type = torch.from_numpy(h5f["static"]["edge_type"][...]).long()

        x_dynamic = torch.from_numpy(h5f["windows"]["x_dynamic"][snapshot_idx]).float()
        y = torch.from_numpy(h5f["windows"]["y"][snapshot_idx]).float()
        future_drivers = torch.from_numpy(h5f["windows"]["future_drivers"][snapshot_idx]).float()
        time_index = int(h5f["windows"]["time_index"][snapshot_idx])

        snapshot_data = {
            "x": torch.cat([x_static, x_dynamic.reshape(x_dynamic.shape[0], -1)], dim=1),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": y.reshape(y.shape[0], -1),
            "future_drivers": future_drivers.reshape(future_drivers.shape[0], -1),
            "time_index": torch.tensor([time_index], dtype=torch.long),
            "node_type": node_type,
            "edge_type": edge_type,
        }

        if "x_aux" in h5f["windows"]:
            x_aux = torch.from_numpy(h5f["windows"]["x_aux"][snapshot_idx]).float()
            snapshot_data["x_aux"] = x_aux.reshape(x_aux.shape[0], -1)

        if "y_aux" in h5f["windows"]:
            y_aux = torch.from_numpy(h5f["windows"]["y_aux"][snapshot_idx]).float()
            snapshot_data["y_aux"] = y_aux.reshape(y_aux.shape[0], -1)

    return Data(**snapshot_data)


class HDFSnapshotDataset(Dataset):
    """Lazy Dataset that reconstructs one graph snapshot per manifest entry."""

    def __init__(self, manifest):
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        sample = self.manifest[idx]
        return load_snapshot_from_hdf(sample["file_path"], sample["snapshot_idx"])


def build_hdf_snapshot_manifest(snapshot_base_dir, event_files=None):
    """Build a flat manifest of all snapshot windows stored in event HDF files."""
    manifest = []
    source_files = event_files if event_files is not None else list_hdf_event_files(snapshot_base_dir)

    for file_path in source_files:
        _, event_name = parse_event_name_from_hdf_filename(file_path)
        with h5py.File(file_path, "r") as h5f:
            num_snapshots = int(h5f["windows"]["time_index"].shape[0])
            for snapshot_idx in range(num_snapshots):
                manifest.append(
                    {
                        "file_path": file_path,
                        "event_name": event_name,
                        "snapshot_idx": snapshot_idx,
                    }
                )

    return manifest


def extract_test_event_files_by_bc_hdf(
    snapshot_base_dir,
    test_events_per_bc=20,
    target_bcs=None,
    seed=42,
    event_files=None,
):
    """Split event files into BC-grouped test files and remaining files."""
    target_bcs = target_bcs or ["BC1", "BC2", "BC3"]
    all_files = event_files if event_files is not None else list_hdf_event_files(snapshot_base_dir)
    grouped = {bc: [] for bc in target_bcs}
    remaining_event_files = []

    for file_path in all_files:
        bc_name, _ = parse_event_name_from_hdf_filename(file_path)
        if bc_name in grouped:
            grouped[bc_name].append(file_path)
        else:
            remaining_event_files.append(file_path)

    rng = random.Random(seed)
    test_event_files = []

    for bc_name in target_bcs:
        candidates = sorted(grouped.get(bc_name, []))
        if not candidates:
            ddp_print(f"  WARNING: No HDF events found for boundary condition {bc_name}")
            continue

        n_select = min(int(test_events_per_bc), len(candidates))
        selected = sorted(rng.sample(candidates, n_select))
        selected_set = set(selected)
        test_event_files.extend(selected)
        remaining_event_files.extend([f for f in candidates if f not in selected_set])

    return sorted(test_event_files), sorted(remaining_event_files)


def _apply_selector(columns, selector):
    if not selector:
        return list(columns)
    if "include" in selector:
        include = selector["include"]
        return [col for col in columns if col in include]
    if "include_prefix" in selector:
        prefixes = selector["include_prefix"]
        return [col for col in columns if any(col.startswith(prefix) for prefix in prefixes)]
    return list(columns)


def _build_static_node_feature_names(config):
    features_cfg = config.get("features", {})
    selectors = config.get("feature_selectors", {}).get("node", {})
    feature_list = features_cfg.get("static_node_features", [])

    feature_names = []
    for feature in feature_list:
        if feature == "node_coordinates":
            columns = ["x", "y"]
        elif feature == "terrain_stats":
            columns = _apply_selector(["zmin", "zmean", "zmax", "relief"], selectors.get("terrain_stats"))
        elif feature in ("hypsometry_polyfit", "hypsometry_curves"):
            raise ValueError("GNN4CF does not support curve or polynomial inputs.")
        elif feature == "external_gis_features":
            columns = selectors.get("external_gis_features", {}).get(
                "include",
                features_cfg.get("external_features", []),
            )
        else:
            columns = [feature]
        feature_names.extend(columns)

    return feature_names


def _build_static_edge_feature_names(config):
    features_cfg = config.get("features", {})
    selectors = config.get("feature_selectors", {}).get("edge", {})
    feature_list = features_cfg.get("static_edge_features", [])

    feature_names = []
    for feature in feature_list:
        if feature == "geometry_stats":
            columns = _apply_selector(["length", "nx", "ny", "d_n"], selectors.get("geometry_stats"))
        elif feature == "face_curves":
            raise ValueError("GNN4CF does not support face-curve inputs.")
        elif feature == "relative_coordinates":
            columns = ["dx", "dy"]
        else:
            columns = [feature]
        feature_names.extend(columns)

    return feature_names


def resolve_normalization_stats_path(config):
    """Resolve the normalization-stats path from the current HDF graph config."""
    cfg_paths = config.get("paths", {})
    return cfg_paths.get(
        "normalization_stats_path",
        os.path.join(
            cfg_paths.get("output_dir", "output_data"),
            cfg_paths.get("normalization_stats_filename", "normalization_stats.json"),
        ),
    )


def resolve_run_output_paths(config):
    """Resolve per-run output directories under `paths.runs_root`."""
    cfg_paths = config.get("paths", {})
    config_name = config.get("config_name")
    if not config_name:
        config_name = cfg_paths.get("graph_dir_name")
    if not config_name:
        graph_root = cfg_paths.get("graph_root")
        config_name = os.path.basename(graph_root) if graph_root else "default_run"

    graphs_hdf_root = cfg_paths.get("graphs_hdf_root")
    default_runs_root = os.path.join(
        os.path.dirname(graphs_hdf_root) if graphs_hdf_root else cfg_paths.get("output_dir", "output_data"),
        "runs",
    )
    runs_root = cfg_paths.get("runs_root", default_runs_root)
    run_dir = os.path.join(runs_root, config_name)

    return {
        "run_name": config_name,
        "run_dir": run_dir,
        "checkpoint_dir": os.path.join(run_dir, cfg_paths.get("checkpoints_dir_name", "checkpoints")),
        "log_dir": os.path.join(run_dir, cfg_paths.get("logs_dir_name", "logs")),
        "rollout_predictions_dir": os.path.join(
            run_dir,
            cfg_paths.get("rollout_predictions_dir_name", "rollout_predictions"),
        ),
        "wandb_dir": os.path.join(run_dir, cfg_paths.get("wandb_dir_name", "wandb")),
        "wandb_run_id_path": os.path.join(run_dir, "wandb_run_id.txt"),
        "run_config_path": os.path.join(run_dir, "config_used.yml"),
    }


def apply_run_output_paths_to_config(config):
    """Store resolved run-output paths back into the config."""
    resolved = resolve_run_output_paths(config)
    config.setdefault("paths", {}).update(resolved)
    return resolved


def save_runtime_run_config_copy(config):
    """Persist the effective training config inside the run directory."""
    run_config_path = config.get("paths", {}).get("run_config_path")
    if not run_config_path:
        return None
    ensure_dir(os.path.dirname(run_config_path))
    with open(run_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return run_config_path


def get_or_create_wandb_run_id(config):
    """Return a persistent W&B run id for this run directory."""
    run_id_path = config.get("paths", {}).get("wandb_run_id_path")
    if not run_id_path:
        return None

    ensure_dir(os.path.dirname(run_id_path))
    if os.path.exists(run_id_path):
        with open(run_id_path, "r", encoding="utf-8") as f:
            run_id = f.read().strip()
        if run_id:
            return run_id

    run_id = wandb.util.generate_id()
    with open(run_id_path, "w", encoding="utf-8") as f:
        f.write(run_id)
    return run_id


def get_feature_counts_and_indices_hdf_for_config(config):
    """Compute feature counts from config and verify them against one HDF sample."""
    cfg_window = config.get("window", {})
    cfg_features = config.get("features", {})
    snapshot_hdf_dir = resolve_snapshot_hdf_dir(config)
    h5_files = list_hdf_event_files(snapshot_hdf_dir)

    if not h5_files:
        raise FileNotFoundError(
            f"No HDF snapshot files found in '{snapshot_hdf_dir}'. "
            "Run `gnn4cf_hdf_graph_dataset.py` with the matching config first."
        )

    sample_graph = load_snapshot_from_hdf(h5_files[0], 0)
    static_node_features = _build_static_node_feature_names(config)
    static_edge_features = _build_static_edge_feature_names(config)

    n_state_vars = len(cfg_window.get("label_vars", []))
    n_driver_vars = len(cfg_features.get("dynamic_input_drivers", []))
    n_dynamic_node = n_state_vars + n_driver_vars
    n_dynamic_edge = 0
    n_label_vars = n_state_vars
    past_steps = int(cfg_window.get("past_steps", 0))
    future_steps = int(cfg_window.get("future_steps", 0))
    n_static_node = len(static_node_features)
    n_static_edge = len(static_edge_features)

    expected_x_dim = n_static_node + (n_dynamic_node * past_steps)
    expected_y_dim = n_label_vars * future_steps
    expected_future_driver_dim = n_driver_vars * future_steps

    if sample_graph.x.shape[1] != expected_x_dim:
        raise ValueError(
            f"HDF node feature mismatch. Expected {expected_x_dim} from config "
            f"(static={n_static_node} + dynamic={n_dynamic_node} * past_steps={past_steps}), "
            f"but sample graph has {sample_graph.x.shape[1]}."
        )
    if sample_graph.edge_attr.shape[1] != n_static_edge:
        raise ValueError(
            f"HDF edge feature mismatch. Expected {n_static_edge} static edge features from config, "
            f"but sample graph has {sample_graph.edge_attr.shape[1]}."
        )
    if sample_graph.y.shape[1] != expected_y_dim:
        raise ValueError(
            f"HDF label mismatch. Expected {expected_y_dim} (= {n_label_vars} * {future_steps}), "
            f"but sample graph has {sample_graph.y.shape[1]}."
        )
    if sample_graph.future_drivers.shape[1] != expected_future_driver_dim:
        raise ValueError(
            f"HDF future-driver mismatch. Expected {expected_future_driver_dim} "
            f"(= {n_driver_vars} * {future_steps}), but sample graph has "
            f"{sample_graph.future_drivers.shape[1]}."
        )

    return {
        "n_static_node": n_static_node,
        "n_static_node_comp": n_static_node,
        "n_static_node_bghost": n_static_node,
        "n_dynamic_node": n_dynamic_node,
        "n_static_edge": n_static_edge,
        "n_static_edge_internal": n_static_edge,
        "n_static_edge_boundary": n_static_edge,
        "n_dynamic_edge": n_dynamic_edge,
        "n_state_vars": n_state_vars,
        "n_driver_vars": n_driver_vars,
        "n_label_vars": n_label_vars,
    }


def resolve_snapshot_hdf_dir(config):
    """Resolve the HDF snapshot directory with backward-compatible fallbacks."""
    cfg_paths = config.get("paths", {})
    return cfg_paths.get("snapshot_hdf_dir", cfg_paths.get("snapshot_dir", "graph_snapshots_hdf"))


def load_cached_test_split_hdf(snapshot_base_dir, test_meta_path, test_events_per_bc, target_bcs, extraction_seed):
    """Load cached event-level test split metadata if parameters still match."""
    if not os.path.exists(test_meta_path):
        return None, None

    try:
        with open(test_meta_path, "r", encoding="utf-8") as f:
            test_meta = json.load(f)

        params_match = (
            test_meta.get("events_per_bc") == test_events_per_bc
            and test_meta.get("target_bcs") == target_bcs
            and test_meta.get("seed") == extraction_seed
        )
        if not params_match:
            return None, None

        h5_files = list_hdf_event_files(snapshot_base_dir)
        test_event_names = set(test_meta.get("test_event_names", []))
        all_event_names = {
            os.path.splitext(os.path.basename(f))[0].replace("_snapshots", ""): f
            for f in h5_files
        }
        remaining_event_names = set(all_event_names.keys()) - test_event_names
        remaining_event_files = sorted(
            [all_event_names[name] for name in remaining_event_names if name in all_event_names]
        )
        return test_meta, remaining_event_files
    except Exception as exc:
        ddp_print(f"  ⚠️  Failed to load cached HDF test split metadata: {exc}")
        return None, None


def save_test_split_metadata_hdf(test_meta_path, test_events_per_bc, target_bcs, extraction_seed, remaining_event_files, snapshot_base_dir):
    """Persist event-level HDF test split metadata for reuse."""
    all_h5_files = set(list_hdf_event_files(snapshot_base_dir))
    remaining_event_files_set = set(remaining_event_files or [])
    test_event_files_set = all_h5_files - remaining_event_files_set
    test_event_names = [
        os.path.splitext(os.path.basename(f))[0].replace("_snapshots", "")
        for f in sorted(test_event_files_set)
    ]

    test_meta = {
        "created_date": datetime.now().isoformat(),
        "events_per_bc": test_events_per_bc,
        "target_bcs": target_bcs,
        "seed": extraction_seed,
        "test_event_names": test_event_names,
        "n_test_events": len(test_event_names),
    }
    with open(test_meta_path, "w", encoding="utf-8") as f:
        json.dump(test_meta, f, indent=2)


def load_or_create_train_val_manifests_hdf(config, snapshot_base_dir, remaining_event_files, extract_test_by_bc):
    """
    Build train/val manifests from HDF event files.

    Dataset handling:
    - test split by event
    - train/val split by snapshot over the remaining event pool
    """
    cfg_train = config["training"]
    cfg_paths = config["paths"]
    use_cached_split = cfg_train.get("use_cached_train_val", False)
    train_val_cache_dir = cfg_paths.get(
        "train_val_cache_dir",
        os.path.join(cfg_paths.get("output_dir", "output_data"), "train_val_cache"),
    )
    train_cache_path = os.path.join(train_val_cache_dir, "train_manifest_hdf.pkl")
    val_cache_path = os.path.join(train_val_cache_dir, "val_manifest_hdf.pkl")
    cache_meta_path = os.path.join(train_val_cache_dir, "cache_metadata_hdf.json")

    if is_distributed() and not use_cached_split:
        ddp_print(
            "WARNING: Multi-GPU HDF training is running without cached train/val manifests. "
            "Each rank will rebuild the manifest locally."
        )

    train_manifest = None
    val_manifest = None

    if use_cached_split and os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        ddp_print(f"\n--- Loading Cached Train/Val HDF Manifests ---")
        ddp_print(f"  Cache directory: {train_val_cache_dir}")
        try:
            with open(train_cache_path, "rb") as f:
                train_manifest = pickle.load(f)
            with open(val_cache_path, "rb") as f:
                val_manifest = pickle.load(f)

            ddp_print(f"  ✅ Loaded {len(train_manifest)} train samples and {len(val_manifest)} val samples")
            if os.path.exists(cache_meta_path):
                with open(cache_meta_path, "r", encoding="utf-8") as f:
                    cache_meta = json.load(f)
                ddp_print(f"  📊 Cache created: {cache_meta.get('created_date', 'unknown')}")
                ddp_print(
                    f"  📊 Original split: {cache_meta.get('n_total', 'unknown')} -> "
                    f"Train: {cache_meta.get('n_train', 'unknown')}, Val: {cache_meta.get('n_val', 'unknown')}"
                )
        except Exception as exc:
            ddp_print(f"  ⚠️  Failed to load cached manifests: {exc}")
            train_manifest = None
            val_manifest = None

    if train_manifest is None or val_manifest is None:
        ddp_print(f"\n--- Creating Train/Val HDF Manifest Split ---")
        manifest_source_files = remaining_event_files if remaining_event_files is not None else list_hdf_event_files(snapshot_base_dir)
        if len(manifest_source_files) == 0:
            raise ValueError(
                "No event files remain for train/val after test extraction. "
                "Reduce `training.test_extraction.events_per_bc` or disable test extraction."
            )

        all_manifest = build_hdf_snapshot_manifest(snapshot_base_dir, event_files=manifest_source_files)

        # Sanity check against corrupted HDF samples before training starts.
        if all_manifest:
            ddp_print("\n--- First indexed HDF snapshot ---")
            first_graph = HDFSnapshotDataset([all_manifest[0]])[0]
            x_has_nan = torch.isnan(first_graph.x).any()
            edge_has_nan = torch.isnan(first_graph.edge_attr).any()
            y_has_nan = torch.isnan(first_graph.y).any()
            ddp_print(f"  Snapshot 'x' has NaN: {x_has_nan}")
            ddp_print(f"  Snapshot 'edge_attr' has NaN: {edge_has_nan}")
            ddp_print(f"  Snapshot 'y' has NaN: {y_has_nan}")
            if x_has_nan or edge_has_nan:
                raise ValueError("NaN found in HDF data loaded from disk. Regenerate HDF snapshots.")
            ddp_print("-----------------------------------------------\n")

        set_random_seed(cfg_train.get("seed", 42))
        random.shuffle(all_manifest)

        split_ratios = cfg_train.get("split_ratios", {"train": 0.85, "val": 0.15})
        n_total = len(all_manifest)
        n_train = int(n_total * split_ratios["train"])

        train_manifest = all_manifest[:n_train]
        val_manifest = all_manifest[n_train:]

        if use_cached_split and is_main_process():
            ensure_dir(train_val_cache_dir)
            ddp_print(f"\n--- Saving Train/Val HDF Manifests to Cache ---")
            ddp_print(f"  Cache directory: {train_val_cache_dir}")
            try:
                with open(train_cache_path, "wb") as f:
                    pickle.dump(train_manifest, f)
                with open(val_cache_path, "wb") as f:
                    pickle.dump(val_manifest, f)

                cache_meta = {
                    "created_date": datetime.now().isoformat(),
                    "n_total": n_total,
                    "n_train": len(train_manifest),
                    "n_val": len(val_manifest),
                    "train_ratio": split_ratios["train"],
                    "val_ratio": split_ratios["val"],
                    "seed": cfg_train.get("seed", 42),
                    "test_extraction_enabled": extract_test_by_bc,
                }
                with open(cache_meta_path, "w", encoding="utf-8") as f:
                    json.dump(cache_meta, f, indent=2)

                ddp_print(f"  ✅ Saved {len(train_manifest)} train samples and {len(val_manifest)} val samples")
            except Exception as exc:
                ddp_print(f"  ⚠️  Failed to save HDF manifest cache: {exc}")
        barrier()

    return train_manifest, val_manifest


def build_dataloaders_hdf(train_manifest, val_manifest, cfg_train):
    """Create train/val dataloaders with optional distributed samplers."""
    train_dataset = HDFSnapshotDataset(train_manifest)
    val_dataset = HDFSnapshotDataset(val_manifest)

    global_batch_size = int(cfg_train["batch_size"])
    per_gpu_batch_size = resolve_per_gpu_batch_size(global_batch_size)
    num_workers = int(cfg_train.get("num_workers", 5))

    if is_distributed():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
        ) if len(val_dataset) > 0 else None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    ddp_print(
        f"Batch configuration -> global: {global_batch_size}, "
        f"per-GPU: {per_gpu_batch_size}, world_size: {get_world_size()}"
    )

    return train_dataset, val_dataset, train_loader, val_loader, train_sampler


def train_gnn_hdf(config, device):
    """Main HDF / on-the-fly multi-GPU training entrypoint."""
    cfg_train = config["training"]
    cfg_paths = config["paths"]
    cfg_model = config["model"]
    cfg_window = config["window"]

    try:
        feature_counts = get_feature_counts_and_indices_hdf_for_config(config)
    except FileNotFoundError as exc:
        ddp_print(f"ERROR: {exc}")
        ddp_print("Please run `gnn4cf_hdf_graph_dataset.py` first to generate HDF snapshots.")
        return
    except Exception as exc:
        ddp_print(f"ERROR during HDF feature counting: {exc}")
        traceback.print_exc()
        return

    snapshot_base_dir = resolve_snapshot_hdf_dir(config)
    selection_info = select_hdf_event_files_from_config(snapshot_base_dir, config)
    selected_event_files = selection_info["selected_event_files"]
    if not selected_event_files:
        raise ValueError(
            f"No HDF event files were selected from '{snapshot_base_dir}'. "
            "Check event_selection settings and available snapshot files."
        )

    if selection_info.get("selection_enabled", False):
        ddp_print(f"\n{'=' * 60}")
        ddp_print("Applying Grouped Event Selection")
        ddp_print(f"{'=' * 60}")
        ddp_print(
            f"  Selected groups: {len(selection_info['selected_groups'])} "
            f"({len(selected_event_files)} event files)"
        )
        ddp_print(
            f"  Pair index range: {selection_info.get('min_pair_index')} "
            f"-> {selection_info.get('max_pair_index')}"
        )
        if selection_info.get("incomplete_groups"):
            ddp_print(
                f"  Skipped incomplete groups: {len(selection_info['incomplete_groups'])}"
            )
        if selection_info.get("skipped_files"):
            ddp_print(
                f"  Skipped unstructured files: {len(selection_info['skipped_files'])}"
            )
        ddp_print(f"{'=' * 60}\n")

    grouped_split_cfg = cfg_train.get("grouped_event_split", {}) or {}
    use_grouped_split = bool(grouped_split_cfg.get("enabled", False))
    split_summary = {}
    test_event_count = 0

    if use_grouped_split:
        grouped_split = load_or_create_grouped_event_manifests_hdf(
            config=config,
            snapshot_base_dir=snapshot_base_dir,
            selection_info=selection_info,
        )
        train_manifest = grouped_split["train_manifest"]
        val_manifest = grouped_split["val_manifest"]
        split_summary = grouped_split["summary"]
        test_event_count = len(grouped_split["test_event_files"])
    else:
        test_extraction_cfg = cfg_train.get("test_extraction", {})
        extract_test_by_bc = test_extraction_cfg.get("enabled", False)
        remaining_event_files = list(selected_event_files)

        if extract_test_by_bc:
            test_events_per_bc = test_extraction_cfg.get("events_per_bc", 20)
            target_bcs = test_extraction_cfg.get("target_bcs", ["BC1", "BC2", "BC3"])
            extraction_seed = test_extraction_cfg.get("seed", cfg_train.get("seed", 42))

            test_output_dir = cfg_paths.get(
                "test_snapshot_dir",
                os.path.join(cfg_paths.get("output_dir", "output_data"), "test_snapshots"),
            )
            test_meta_path = os.path.join(test_output_dir, "test_extraction_metadata_hdf.json")

            cached_test_meta, cached_remaining_event_files = load_cached_test_split_hdf(
                snapshot_base_dir,
                test_meta_path,
                test_events_per_bc,
                target_bcs,
                extraction_seed,
            )
            if cached_test_meta is not None:
                test_event_count = int(cached_test_meta.get("n_test_events", 0))
                remaining_event_files = [
                    file_path for file_path in cached_remaining_event_files if file_path in set(selected_event_files)
                ]
                ddp_print(f"\n--- Loading Cached HDF Test Event Split ---")
                ddp_print(f"  Cache directory: {test_output_dir}")
                ddp_print(f"  ✅ Loaded {test_event_count} cached test events")
            else:
                ddp_print(f"\n{'=' * 60}")
                ddp_print("Extracting HDF Test Events by Boundary Condition")
                ddp_print(f"{'=' * 60}")
                ddp_print(f"  Target BCs: {target_bcs}")
                ddp_print(f"  Events per BC: {test_events_per_bc}")
                ddp_print(f"{'=' * 60}\n")

                test_event_files, remaining_event_files = extract_test_event_files_by_bc_hdf(
                    snapshot_base_dir,
                    test_events_per_bc=test_events_per_bc,
                    target_bcs=target_bcs,
                    seed=extraction_seed,
                    event_files=selected_event_files,
                )
                test_event_count = len(test_event_files)

                if is_main_process():
                    ensure_dir(test_output_dir)
                    save_test_split_metadata_hdf(
                        test_meta_path,
                        test_events_per_bc,
                        target_bcs,
                        extraction_seed,
                        remaining_event_files,
                        snapshot_base_dir,
                    )
                    ddp_print(f"  ✅ Saved HDF test split metadata to: {test_meta_path}")
                barrier()

        train_manifest, val_manifest = load_or_create_train_val_manifests_hdf(
            config,
            snapshot_base_dir,
            remaining_event_files,
            extract_test_by_bc,
        )

    ddp_print(f"\n--- Train/Val Split Summary ---")
    ddp_print(
        f"Total Snapshots (train/val): {len(train_manifest) + len(val_manifest)} -> "
        f"Train: {len(train_manifest)}, Val: {len(val_manifest)}"
    )
    if use_grouped_split:
        ddp_print(
            f"Grouped Events -> Train: {split_summary.get('train_group_count', 0)}, "
            f"Val: {split_summary.get('val_group_count', 0)}, "
            f"Test: {split_summary.get('test_group_count', 0)}"
        )
        ddp_print(
            f"Event Files -> Train: {split_summary.get('train_event_count', 0)}, "
            f"Val: {split_summary.get('val_event_count', 0)}, "
            f"Test: {split_summary.get('test_event_count', 0)}"
        )
    elif cfg_train.get("test_extraction", {}).get("enabled", False):
        ddp_print(f"Test Events (extracted by BC): {test_event_count}")
    ddp_print(f"{'=' * 60}\n")

    if is_main_process() and config.get("wandb", {}).get("enabled", False) and wandb.run:
        update_wandb_dataset_summary(selection_info, split_summary, use_grouped_split)

    if not train_manifest:
        raise ValueError("No training data found after split. Check test extraction or split ratios.")

    train_dataset, val_dataset, train_loader, val_loader, train_sampler = build_dataloaders_hdf(
        train_manifest,
        val_manifest,
        cfg_train,
    )

    shared_steps = cfg_model["nmessage_passing_steps"]
    shared_layers = cfg_model["nmlp_layers"]
    shared_hidden = cfg_model["mlp_hidden_dim"]

    interior_cfg = cfg_model.get("interior", {}) or {}
    coupling_cfg = cfg_model.get("coupling", {}) or {}
    boundary_skip = cfg_model.get("boundary_skip", True)
    boundary_preserve_weight = cfg_model.get("boundary_preserve_weight", None)
    residual = cfg_model.get("residual", True)

    # Boundary conditioning reads the top-level boundary_conditioning section.
    bc_cfg = config.get("boundary_conditioning", {}) or {}
    bc_enabled = bool(bc_cfg.get("enabled", False))
    bc_mode = str(bc_cfg.get("mode", "concat"))
    bc_inject_every_step = bool(bc_cfg.get("inject_every_processor_step", True))
    bc_use_current = bool(bc_cfg.get("use_current_state", True))
    bc_use_history = bool(bc_cfg.get("use_history_state", True))
    bc_use_geometry = bool(bc_cfg.get("use_geometry", True))
    bc_geometry_interaction = str(bc_cfg.get("geometry_interaction", "multiply"))
    bc_current_hid = int(bc_cfg.get("current_encoder_hidden_dim", 64))
    bc_history_hid = int(bc_cfg.get("history_encoder_hidden_dim", 64))
    bc_geometry_hid = int(bc_cfg.get("geometry_encoder_hidden_dim", 64))
    bc_update_mlp_hid = int(bc_cfg.get("update_mlp_hidden_dim", 64))
    bc_film_hid = int(bc_cfg.get("film_hidden_dim", 64))

    dynamic_input_feature_names = build_dynamic_input_feature_names(config)
    rainfall_kwargs = get_rainfall_conditioning_kwargs(
        config,
        dynamic_input_feature_names=dynamic_input_feature_names,
    )
    log_model_setup(
        config=config,
        feature_counts=feature_counts,
        cfg_model=cfg_model,
        bc_cfg=bc_cfg,
        rainfall_kwargs=rainfall_kwargs,
        dynamic_input_feature_names=dynamic_input_feature_names,
    )

    normalization_stats = {}
    stats_path = resolve_normalization_stats_path(config)
    if os.path.exists(stats_path):
        normalization_stats = load_stats_file(stats_path)
    config.setdefault("_runtime", {})
    config["_runtime"]["normalization_stats"] = normalization_stats

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
        boundary_conditioning_enabled=bc_enabled,
        boundary_conditioning_mode=bc_mode,
        inject_every_processor_step=bc_inject_every_step,
        use_current_state=bc_use_current,
        use_history_state=bc_use_history,
        use_geometry=bc_use_geometry,
        geometry_interaction=bc_geometry_interaction,
        current_encoder_hidden_dim=bc_current_hid,
        history_encoder_hidden_dim=bc_history_hid,
        geometry_encoder_hidden_dim=bc_geometry_hid,
        update_mlp_hidden_dim=bc_update_mlp_hid,
        film_hidden_dim=bc_film_hid,
        **rainfall_kwargs,
    ).to(device)

    ddp_print(f"Model Instantiated: {model.__class__.__name__}")
    ddp_print(f"Number of trainable parameters: {count_parameters(model):,}")

    flood_aware_config = config.get("loss", {}).get("flood_aware", {})
    if flood_aware_config.get("enabled", False):
        threshold_real = float(flood_aware_config.get("threshold_real", 0.0))
        ddp_print(f"\n--- Flood-Aware Loss Configuration ---")
        ddp_print(f"  Threshold (real units): {threshold_real:.6f} m")
        try:
            if normalization_stats:
                if "wd" in normalization_stats:
                    threshold_normalized = gnn_utils.convert_real_threshold_to_normalized(
                        threshold_real, normalization_stats, var_name="wd"
                    )
                    config["loss"]["flood_aware"]["threshold_normalized"] = threshold_normalized
                    config["loss"]["flood_aware"]["normalization_stats"] = normalization_stats
                    ddp_print(f"  Threshold (normalized): {threshold_normalized:.6f}")
                else:
                    config["loss"]["flood_aware"]["threshold_normalized"] = threshold_real
                    ddp_print(f"  Threshold (normalized): {threshold_real:.6f} (using real value directly)")
            else:
                config["loss"]["flood_aware"]["threshold_normalized"] = threshold_real
                ddp_print(f"  Threshold (normalized): {threshold_real:.6f} (stats not found)")
        except Exception as exc:
            ddp_print(f"  ⚠️  Warning: Could not load normalization stats: {exc}")
            config["loss"]["flood_aware"]["threshold_normalized"] = threshold_real
            ddp_print(f"  Threshold (normalized): {threshold_real:.6f} (using real value directly)")
        ddp_print(f"----------------------------------------\n")

    shallow_cfg = config.get("loss", {}).get("shallow_depth_loss", {})
    if shallow_cfg.get("enabled", False):
        ddp_print("\n--- Shallow-Depth Auxiliary Loss Configuration ---")
        ddp_print(f"  Active depth band: [{float(shallow_cfg.get('min_depth_m', 0.01)):.4f}, "
                  f"{float(shallow_cfg.get('max_depth_m', shallow_cfg.get('threshold_m', 0.5))):.4f}) m")
        ddp_print(f"  lambda_shallow: {float(shallow_cfg.get('lambda_shallow', shallow_cfg.get('weight', 0.1))):.6f}")
        ddp_print(f"  loss_type: {shallow_cfg.get('loss_type', 'smooth_l1')}")
        ddp_print("  optimization space: normalized WD")
        ddp_print("  diagnostics space: meters (shallow_depth_rmse_m, shallow_depth_mae_m)")
        ddp_print("-----------------------------------------------\n")

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg_train["learning_rate"],
        weight_decay=float(cfg_train.get("weight_decay", 1e-8)),
    )

    scheduler_type = cfg_train.get("scheduler_type", "StepLR").lower()
    scheduler_params = cfg_train.get("scheduler_params", {})
    if scheduler_type == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get("mode", "min"),
            factor=float(scheduler_params.get("factor", 0.5)),
            patience=int(scheduler_params.get("patience", 10)),
            min_lr=float(scheduler_params.get("min_lr", 1e-6)),
        )
    elif scheduler_type == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_params.get("T_max", int(cfg_train.get("epochs", 500)))),
            eta_min=float(scheduler_params.get("eta_min", 1e-6)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg_train.get("scheduler_step", 10)),
            gamma=float(cfg_train.get("scheduler_gamma", 0.1)),
        )

    scheduler_info = {
        "type": scheduler_type,
        "needs_validation_metric": scheduler_type == "reducelronplateau",
    }

    checkpoint_dir = cfg_paths.get("checkpoint_dir", os.path.join(cfg_paths.get("output_dir", "output_data"), "checkpoints"))
    if is_main_process():
        ensure_dir(checkpoint_dir)
    barrier()
    checkpoint_path = os.path.join(checkpoint_dir, "cf_checkpoint_hdf.pth")
    best_model_path = os.path.join(checkpoint_dir, "best_model_hdf.pth")

    logger = None
    log_dir = cfg_paths.get("log_dir")
    if log_dir and is_main_process():
        ensure_dir(log_dir)
        log_file = os.path.join(log_dir, f"training_hdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
        ddp_print(f"📝 Logging to: {log_file}")

    save_every = cfg_paths.get("save_every", 1)

    start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )

    ddp_print(f"Starting from Epoch: {start_epoch}, Best Val Loss: {best_val_loss:.4e}")
    initial_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
    ddp_print(f"Initial Learning Rate: {initial_lr:.6e}")
    if logger:
        logger.info(f"Training started from epoch {start_epoch}, best val loss: {best_val_loss:.4e}")
        logger.info(f"Initial learning rate: {initial_lr:.6e}")

    for epoch in range(start_epoch, cfg_train["epochs"]):
        if training_stop_requested():
            stop_signal = get_training_stop_signal_name()
            ddp_print(
                f"Graceful stop requested by signal {stop_signal or 'unknown'} before starting epoch {epoch + 1}. "
                "Exiting without advancing training."
            )
            if logger:
                logger.info(
                    f"Graceful stop requested by signal {stop_signal or 'unknown'} before epoch {epoch + 1}."
                )
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ddp_print(f"\n--- Epoch {epoch + 1}/{cfg_train['epochs']} ---")
        if logger:
            logger.info(f"Epoch {epoch + 1}/{cfg_train['epochs']} started")

        train_metrics = gnn_utils.train_loop(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_info=scheduler_info,
            device=device,
            config=config,
            feature_counts=feature_counts,
        )
        train_metrics = reduce_metrics_dict(train_metrics, device)

        if training_stop_requested():
            stop_signal = get_training_stop_signal_name()
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
            ddp_print(
                f"Graceful stop requested by signal {stop_signal or 'unknown'} after training epoch {epoch + 1}. "
                "Saving checkpoint and exiting before validation."
            )
            if logger:
                logger.info(
                    f"Graceful stop requested by signal {stop_signal or 'unknown'} after training epoch {epoch + 1}. "
                    "Saving checkpoint and exiting before validation."
                )

            if is_main_process():
                save_checkpoint(epoch + 1, model, optimizer, scheduler, best_val_loss, checkpoint_path)
                if logger:
                    logger.info(
                        f"Graceful-stop checkpoint saved at epoch {epoch + 1} before validation."
                    )

                if config.get("wandb", {}).get("enabled", False) and wandb.run:
                    wandb.run.summary["system/stop_signal"] = stop_signal or "unknown"
                    wandb.run.summary["system/graceful_stop_epoch"] = epoch + 1
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "optimization/learning_rate": current_lr,
                            "optimization/train_total": train_metrics["total_loss"],
                            "summary/train_loss": train_metrics["total_loss"],
                            "system/graceful_stop_requested": 1,
                        }
                    )
            barrier()
            break

        validation_uses_stability = float(
            config.get("loss", {}).get("loss_weights", {}).get("stability", 0.0)
        ) > 0.0
        validation_branch_label = "Stability" if validation_uses_stability else "Real"
        val_loss_key_prefix = "stability" if validation_uses_stability else "real"

        if len(val_dataset) > 0:
            val_metrics = gnn_utils.validate_loop(
                model=model,
                dataloader=val_loader,
                device=device,
                config=config,
                feature_counts=feature_counts,
            )
            val_metrics = reduce_metrics_dict(val_metrics, device)
            val_loss = val_metrics.get("total_loss", 0.0)
            val_rmse_wd_m = val_metrics.get(
                f"{val_loss_key_prefix}_rmse_wd_m",
                val_metrics.get("stability_rmse_wd_m", val_metrics.get("real_rmse_wd_m", 0.0)),
            )
            val_rmse_wd_normalized = val_metrics.get(
                f"{val_loss_key_prefix}_rmse_wd_normalized",
                val_metrics.get(
                    "stability_rmse_wd_normalized",
                    val_metrics.get("real_rmse_wd_normalized", 0.0),
                ),
            )
            if scheduler_info["needs_validation_metric"]:
                scheduler.step(val_loss)
        else:
            val_metrics = {}
            val_loss = 0.0
            val_rmse_wd_m = 0.0
            val_rmse_wd_normalized = 0.0

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
        if val_rmse_wd_m > 0:
            val_rmse_text = f"{val_rmse_wd_m:.4f} m"
        else:
            val_rmse_text = f"{val_rmse_wd_normalized:.4f} (normalized)"
        ddp_print(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_metrics['total_loss']:.4e}, "
            f"Val Loss: {val_loss:.4e} ({validation_branch_label}), "
            f"Val WD RMSE: {val_rmse_text}, "
            f"LR: {current_lr:.6e}"
        )
        if logger:
            logger.info(
                f"Epoch {epoch + 1}: Train Loss: {train_metrics['total_loss']:.4e}, "
                f"Val Loss: {val_loss:.4e} ({validation_branch_label}), "
                f"Val WD RMSE: {val_rmse_text}, LR: {current_lr:.6e}"
            )

        if is_main_process() and config.get("wandb", {}).get("enabled", False) and wandb.run:
            log_metrics = build_wandb_log_metrics(
                epoch=epoch + 1,
                current_lr=current_lr,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                val_loss_key_prefix=val_loss_key_prefix,
            )
            wandb.log(log_metrics)

        if is_main_process() and ((epoch + 1) % save_every == 0 or (epoch + 1) == cfg_train["epochs"]):
            save_checkpoint(epoch + 1, model, optimizer, scheduler, best_val_loss, checkpoint_path)
            if logger:
                logger.info(f"Checkpoint saved at epoch {epoch + 1}")

        current_val_loss = val_metrics.get("total_loss", float("inf"))
        if not val_metrics:
            current_val_loss = train_metrics["total_loss"]
            if epoch == (cfg_train["epochs"] - 1) and is_main_process():
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), best_model_path)
                ddp_print("🌟 Saved final model (no validation set).")
        elif current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            if is_main_process():
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), best_model_path)
                ddp_print(f"🌟 Best model saved with val loss: {best_val_loss:.4e}")
            if logger:
                logger.info(f"Best model saved at epoch {epoch + 1} with val loss: {best_val_loss:.4e}")

        if training_stop_requested():
            stop_signal = get_training_stop_signal_name()
            ddp_print(
                f"Graceful stop requested by signal {stop_signal or 'unknown'} after completing epoch {epoch + 1}. "
                "Checkpoint is up to date; exiting now."
            )
            if logger:
                logger.info(
                    f"Graceful stop requested by signal {stop_signal or 'unknown'} after epoch {epoch + 1}. Exiting."
                )
            if is_main_process() and config.get("wandb", {}).get("enabled", False) and wandb.run:
                wandb.run.summary["system/stop_signal"] = stop_signal or "unknown"
                wandb.run.summary["system/graceful_stop_epoch"] = epoch + 1
                wandb.log({"epoch": epoch + 1, "system/graceful_stop_requested": 1})
            break

    ddp_print("✅ HDF / on-the-fly training complete.")
    if logger:
        logger.info("Training completed successfully")

    if is_main_process() and config.get("wandb", {}).get("enabled", False) and wandb.run:
        wandb.finish()

    return model


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train GNN4CF using HDF5 graph datasets with optional multi-GPU DDP.")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config YAML")
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Config not found at {args.config}")
        return
    except Exception as exc:
        print(f"ERROR reading config: {exc}")
        return

    try:
        apply_hdf_graph_paths_to_config(config)
    except Exception as exc:
        print(f"ERROR resolving HDF graph paths from config: {exc}")
        return

    apply_run_output_paths_to_config(config)
    reset_training_signal_state()
    previous_signal_handlers = install_training_signal_handlers()

    distributed, rank, world_size, device = init_distributed_mode()
    configure_worker_runtime()

    seed = config.get("training", {}).get("seed", 42)
    set_random_seed(seed)
    ddp_print(f"Random seed set to {seed}")
    ddp_print(f"Using device: {device}")
    ddp_print(f"Distributed mode: {distributed} (rank={rank}, world_size={world_size})")
    if is_main_process():
        cfg_paths = config.get("paths", {})
        for key in ["run_dir", "checkpoint_dir", "log_dir", "rollout_predictions_dir", "wandb_dir"]:
            ensure_dir(cfg_paths.get(key))
        run_config_path = save_runtime_run_config_copy(config)
        if run_config_path:
            ddp_print(f"Run config saved to: {run_config_path}")
    barrier()

    if not is_main_process():
        config.setdefault("wandb", {})
        config["wandb"]["enabled"] = False

    if config.get("wandb", {}).get("enabled", False):
        try:
            wandb_dir = config.get("paths", {}).get("wandb_dir", None)
            config_name = config.get("config_name", None)
            if config_name:
                wandb_run_name = config_name
            else:
                loss_type = config.get("loss", {}).get("loss_type", "unknown")
                scheduler_type = config.get("training", {}).get("scheduler_type", "unknown")
                wandb_run_name = f"{loss_type}_{scheduler_type}_hdf"
            wandb_run_id = get_or_create_wandb_run_id(config)

            wandb_init_kwargs = {
                "project": config["wandb"]["project"],
                "entity": config["wandb"].get("entity", None),
                "name": wandb_run_name,
                "config": config,
                "tags": config["wandb"].get("tags", []) + ["hdf", "on-the-fly"],
            }
            if wandb_run_id:
                wandb_init_kwargs["id"] = wandb_run_id
                wandb_init_kwargs["resume"] = "allow"
            if wandb_dir:
                ensure_dir(wandb_dir)
                wandb_init_kwargs["dir"] = wandb_dir

            wandb.init(**wandb_init_kwargs)
            setup_wandb_metric_layout()
            ddp_print(f"WandB initialized with run name: {wandb_run_name}")
            if wandb_run_id:
                ddp_print(f"WandB resume id: {wandb_run_id}")
            if wandb_dir:
                ddp_print(f"WandB files saved to: {wandb_dir}")
        except Exception as exc:
            ddp_print(f"Could not initialize WandB: {exc}. Training will continue without logging.")
            config["wandb"]["enabled"] = False

    try:
        train_gnn_hdf(config, device)
    except Exception as exc:
        ddp_print(f"\n--- An error occurred during HDF training ---")
        ddp_print(f"ERROR: {exc}")
        traceback.print_exc()
        ddp_print("-----------------------------------------")
    finally:
        if is_main_process() and config.get("wandb", {}).get("enabled", False) and wandb.run:
            wandb.finish()
            ddp_print("WandB run finished.")
        restore_training_signal_handlers(previous_signal_handlers)
        cleanup_distributed()


if __name__ == "__main__":
    main()
