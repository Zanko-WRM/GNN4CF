# -*- coding: utf-8 -*-

"""
Provide training, validation, loss, metric, and rollout utilities for GNN4CF.

The module implements supervised and autoregressive stability training,
normalized-space water-depth losses, shallow-depth regularization, flood
metrics, and denormalization. Temporal helpers preserve predictor windows and
known-driver pushforward, with water-depth supervision restricted to
computational nodes.

With the manuscript configuration, each teacher-forced (real) and pushforward
stability branch combines the main water-depth loss with the shallow-depth
auxiliary loss. Optional flood-aware classification is disabled in that setup.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
from typing import Dict, List
import os
import json


# =============================================================================
# 1. Random Seed
# =============================================================================

def set_random_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# 2. Flood-Aware Loss Helper Functions
# =============================================================================

def convert_real_threshold_to_normalized(threshold_real, stats_dict, var_name="wd"):
    """
    Converts a threshold from real units (meters) to normalized space.

    Uses the saved normalization statistics to perform the conversion.
    Handles different normalization methods (minmax, zscore, log1p_zscore).

    Args:
        threshold_real: Threshold value in real units (e.g., 0.03 for 3cm)
        stats_dict: Dictionary containing normalization statistics (from normalization_stats.json)
        var_name: Variable name (default: "wd" for water depth)

    Returns:
        threshold_normalized: Threshold value in normalized space
    """
    # Ensure threshold_real is a float (YAML might load it as string)
    threshold_real = float(threshold_real)

    if var_name not in stats_dict:
        raise ValueError(
            f"Variable '{var_name}' not found in normalization stats. "
            f"Available variables: {list(stats_dict.keys())}"
        )

    var_stats = stats_dict[var_name]
    method = var_stats.get("method", "minmax")

    if method == "minmax":
        min_val = float(var_stats["min"])
        max_val = float(var_stats["max"])
        if max_val > min_val:
            threshold_normalized = (threshold_real - min_val) / (max_val - min_val)
        else:
            threshold_normalized = 0.0

    elif method == "zscore":
        mean = float(var_stats["mean"])
        std = float(var_stats["std"])
        if std > 1e-8:
            threshold_normalized = (threshold_real - mean) / std
        else:
            threshold_normalized = 0.0

    elif method == "log1p_zscore":
        # Step 1: Apply log1p transformation
        log1p_value = np.log1p(threshold_real)  # log(1 + threshold_real)

        # Step 2: Apply z-score normalization
        log1p_mean = float(var_stats["log1p_mean"])
        log1p_std = float(var_stats["log1p_std"])
        if log1p_std > 1e-8:
            threshold_normalized = (log1p_value - log1p_mean) / log1p_std
        else:
            threshold_normalized = 0.0

    elif method == "no_norm":
        # No normalization, use real value directly
        threshold_normalized = threshold_real

    else:
        raise ValueError(
            f"Unknown normalization method '{method}' for variable '{var_name}'. "
            f"Supported methods: minmax, zscore, log1p_zscore, no_norm"
        )

    return float(threshold_normalized)


def denormalize_tensor(normalized_tensor, stats_dict, var_name="wd", clamp_nonnegative=True):
    """
    Denormalizes a PyTorch tensor from normalized space back to real units.

    This is the inverse of normalization, used for visualization, metrics computation,
    and other utilities that require real-world units.

    Args:
        normalized_tensor: PyTorch tensor in normalized space [T, N, 1] or any shape
        stats_dict: Dictionary containing normalization statistics (from normalization_stats.json)
        var_name: Variable name (default: "wd" for water depth)

    Returns:
        denormalized_tensor: PyTorch tensor in real units (same shape as input)
                            Optionally clamped to >= 0 for physically nonnegative variables.
    """
    if var_name not in stats_dict:
        raise ValueError(
            f"Variable '{var_name}' not found in normalization stats. "
            f"Available variables: {list(stats_dict.keys())}"
        )

    var_stats = stats_dict[var_name]
    method = var_stats.get("method", "minmax")
    device = normalized_tensor.device
    dtype = normalized_tensor.dtype

    if method == "minmax":
        min_val = torch.tensor(var_stats["min"], device=device, dtype=dtype)
        max_val = torch.tensor(var_stats["max"], device=device, dtype=dtype)
        denormalized = normalized_tensor * (max_val - min_val) + min_val

    elif method == "zscore":
        mean = torch.tensor(var_stats["mean"], device=device, dtype=dtype)
        std = torch.tensor(var_stats["std"], device=device, dtype=dtype)
        denormalized = normalized_tensor * std + mean

    elif method == "log1p_zscore":
        # Step 1: Reverse z-score normalization
        log1p_mean = torch.tensor(var_stats["log1p_mean"], device=device, dtype=dtype)
        log1p_std = torch.tensor(var_stats["log1p_std"], device=device, dtype=dtype)
        log1p_data = normalized_tensor * log1p_std + log1p_mean

        # Step 2: Reverse log1p transformation: exp(x) - 1
        denormalized = torch.expm1(log1p_data)

    elif method == "no_norm":
        # No normalization, use normalized value directly
        denormalized = normalized_tensor

    else:
        raise ValueError(
            f"Unknown normalization method '{method}' for variable '{var_name}'. "
            f"Supported methods: minmax, zscore, log1p_zscore, no_norm"
        )

    # Keep default behavior for physical-depth workflows while allowing
    # diagnostic mode to inspect negative values before clipping.
    if clamp_nonnegative:
        denormalized = torch.clamp(denormalized, min=0.0)

    return denormalized


def compute_flood_metrics(pred, target, threshold_normalized, comp_mask=None):
    """
    Computes flood-specific metrics from predictions and targets.

    Args:
        pred: [T, N, 1] predicted water depth values (in normalized space)
        target: [T, N, 1] target water depth values (in normalized space)
        threshold_normalized: Threshold in normalized space (scalar)
        comp_mask: Optional boolean mask for computational nodes [N]

    Returns:
        dict with flood metrics:
            - flooded_nodes_pct: Percentage of nodes that are flooded (target > threshold)
            - flood_precision: Precision of predicted flooded nodes
            - flood_recall: Recall of predicted flooded nodes
            - flood_f1: F1 score of predicted flooded nodes
            - mean_flood_depth_normalized: Mean water depth across flooded nodes (normalized)
            - max_flood_depth_normalized: Maximum water depth (normalized)
            - mean_depth_all_normalized: Mean water depth across all nodes (normalized)
    """
    # Apply mask if provided
    if comp_mask is not None:
        pred = pred[:, comp_mask, :]
        target = target[:, comp_mask, :]

    # Flatten time and node dimensions: [T, N, 1] -> [T*N]
    target_flat = target.flatten()
    pred_flat = pred.flatten()

    # Compute flooded mask (target > threshold) and predicted flooded mask.
    flooded_mask = (target_flat > threshold_normalized)
    pred_flooded_mask = (pred_flat > threshold_normalized)

    # Flooded node percentage
    total_nodes = target_flat.numel()
    flooded_count = flooded_mask.sum().item()
    flooded_nodes_pct = (flooded_count / total_nodes * 100.0) if total_nodes > 0 else 0.0

    true_positive = torch.logical_and(pred_flooded_mask, flooded_mask).sum().item()
    false_positive = torch.logical_and(pred_flooded_mask, ~flooded_mask).sum().item()
    false_negative = torch.logical_and(~pred_flooded_mask, flooded_mask).sum().item()

    flood_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    flood_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    flood_f1 = (
        2.0 * flood_precision * flood_recall / (flood_precision + flood_recall)
        if (flood_precision + flood_recall) > 0
        else 0.0
    )

    # Mean flood depth (only for flooded nodes) - in normalized space
    if flooded_mask.any():
        mean_flood_depth_normalized = target_flat[flooded_mask].mean().item()
    else:
        mean_flood_depth_normalized = 0.0

    # Max flood depth - in normalized space
    max_flood_depth_normalized = target_flat.max().item()

    # Mean depth across all nodes - in normalized space
    mean_depth_all_normalized = target_flat.mean().item()

    return {
        'flooded_nodes_pct': flooded_nodes_pct,
        'flood_precision': flood_precision,
        'flood_recall': flood_recall,
        'flood_f1': flood_f1,
        'mean_flood_depth_normalized': mean_flood_depth_normalized,
        'max_flood_depth_normalized': max_flood_depth_normalized,
        'mean_depth_all_normalized': mean_depth_all_normalized
    }


def compute_weighted_regression_loss(pred, target, threshold_normalized,
                                    weight_non_flooded, loss_type,
                                    lambda_rel=0.5, lambda_abs=0.5,
                                    alpha=None, max_weight=None):
    """
    Computes regression loss with per-node weighting based on flood threshold.

    Uses bounded soft exponential-to-cap weighting that works entirely
    in normalized space for numerical stability. Weight increases monotonically
    with excess depth above threshold and saturates at max_weight.

    Weight formula: w = w0 + (w_max - w0) * (1 - exp(-alpha * delta_norm))
        where:
        - w0 = weight_non_flooded (baseline weight)
        - w_max = max_weight (maximum weight cap)
        - alpha = alpha (rate parameter controlling how fast weight increases)
        - delta_norm = max(0, target_normalized - threshold_normalized)

    This approach:
    - Works entirely in normalized space (no denormalization needed)
    - Is bounded by design (prevents gradient explosion)
    - Preserves physical meaning (threshold-aware, monotonic increasing)
    - Is numerically stable (no exponential overflow)

    Args:
        pred: [T, N, 1] predicted values (in normalized space)
        target: [T, N, 1] target values (in normalized space)
        threshold_normalized: Threshold in normalized space (scalar) - used for flood mask
        weight_non_flooded: Baseline weight for non-flooded nodes (wd <= threshold)
        loss_type: Loss type ("mse", "mae", "rmse", "relative_mse", "hybrid")
        lambda_rel: Weight for relative component (hybrid loss)
        lambda_abs: Weight for absolute component (hybrid loss)
        alpha: Optional. Rate parameter for bounded exponential weighting.
               If provided and > 0, enables bounded exponential weighting.
               Higher values = faster weight increase with depth.
               Example: alpha=10.0 means weight approaches max_weight quickly.
        max_weight: Optional. Maximum weight cap (required if alpha is provided).
                    Prevents extreme weights and ensures bounded gradients.
                    Example: 50.0 means flooded nodes get at most 50x baseline weight.

    Returns:
        weighted_loss: Scalar weighted loss
    """
    # Compute per-element loss
    if loss_type == "mse":
        element_loss = (pred - target) ** 2
    elif loss_type == "mae":
        element_loss = torch.abs(pred - target)
    elif loss_type == "rmse":
        element_loss = (pred - target) ** 2  # Will take sqrt at end
    elif loss_type == "relative_mse":
        element_loss = (pred - target) ** 2 / (target ** 2 + 1e-8)
    elif loss_type == "hybrid":
        rel_loss = (pred - target) ** 2 / (target ** 2 + 1e-8)
        abs_loss = torch.abs(pred - target)
        element_loss = lambda_rel * rel_loss + lambda_abs * abs_loss
    else:
        # Default to relative_mse
        element_loss = (pred - target) ** 2 / (target ** 2 + 1e-8)

    # Apply weights based on mode
    if alpha is not None and alpha > 0:
        # Bounded exponential weighting mode: weight based on normalized excess depth
        if max_weight is None:
            raise ValueError(
                "alpha requires max_weight to be provided. "
                "This ensures bounded weights and prevents gradient explosion."
            )

        if max_weight <= weight_non_flooded:
            raise ValueError(
                f"max_weight ({max_weight}) must be greater than weight_non_flooded ({weight_non_flooded})."
            )

        # Compute excess depth in normalized space: max(0, target_normalized - threshold_normalized)
        excess_depth_normalized = torch.clamp(target - threshold_normalized, min=0.0)

        # Apply bounded exponential weighting: w = w0 + (w_max - w0) * (1 - exp(-alpha * delta_norm))
        # This is a saturating exponential that:
        # - Starts at w0 when delta_norm = 0
        # - Approaches w_max as delta_norm -> infinity
        # - Is bounded, monotonic, and numerically stable
        alpha_tensor = torch.tensor(alpha, device=target.device, dtype=target.dtype)
        weight_range = max_weight - weight_non_flooded
        weights = weight_non_flooded + weight_range * (1.0 - torch.exp(-alpha_tensor * excess_depth_normalized))

    else:
        # Fallback: uniform weighting (no exponential)
        weights = torch.full_like(target, weight_non_flooded, dtype=target.dtype)

    weighted_element_loss = weights * element_loss

    # Average over all dimensions
    weighted_loss = weighted_element_loss.mean()

    # For RMSE, take sqrt
    if loss_type == "rmse":
        weighted_loss = torch.sqrt(weighted_loss + 1e-8)

    return weighted_loss


def compute_classification_loss(pred, target, threshold_normalized, temperature=1.0):
    """
    Computes binary classification loss for flooded vs non-flooded.

    Converts regression predictions to probabilities using sigmoid and computes
    binary cross-entropy loss.

    Shift sigmoid input by the flood threshold to align the wet/dry boundary.
    This ensures sigmoid's threshold (at 0) matches the flood threshold.

    Args:
        pred: [T, N, 1] predicted values (in normalized space)
        target: [T, N, 1] target values (in normalized space)
        threshold_normalized: Threshold in normalized space (scalar, e.g., -0.4581)
        temperature: Temperature scaling for sigmoid (default: 1.0)

    Returns:
        classification_loss: Scalar BCE loss
        classification_metrics: Dict with accuracy / precision / recall / f1
    """
    # Create binary labels: 1 if flooded (target > threshold), 0 otherwise
    labels = (target > threshold_normalized).float()  # [T, N, 1]

    # Convert predictions to probabilities using sigmoid
    # Shift by the flood threshold so sigmoid(0) marks the wet/dry boundary.
    # Without shift: sigmoid threshold is at 0, but flood threshold might be at -0.4581
    # With shift: sigmoid((pred - threshold)) ensures when pred = threshold, sigmoid(0) = 0.5
    pred_shifted = pred - threshold_normalized  # Shift so threshold is at 0
    pred_scaled = pred_shifted / temperature  # Apply temperature scaling
    probs = torch.sigmoid(pred_scaled)  # [T, N, 1]

    # Compute binary cross-entropy loss
    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    bce_loss = F.binary_cross_entropy(probs, labels, reduction='mean')

    # Compute threshold-based classification metrics.
    pred_binary = (probs > 0.5).float()
    accuracy = (pred_binary == labels).float().mean()
    true_positive = (pred_binary * labels).sum()
    false_positive = (pred_binary * (1.0 - labels)).sum()
    false_negative = ((1.0 - pred_binary) * labels).sum()

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    return bce_loss, {
        "classification_accuracy": accuracy,
        "classification_precision": precision,
        "classification_recall": recall,
        "classification_f1": f1,
    }


def _get_shallow_depth_config(config: Dict) -> Dict:
    """
    Return the optional shallow-depth auxiliary-loss config.

    Canonical config key:
      loss.shallow_depth_loss
    """
    loss_config = config.get("loss", {}) if isinstance(config, dict) else {}
    return loss_config.get("shallow_depth_loss", {}) or {}


def _shallow_depth_enabled(config: Dict) -> bool:
    """Whether the shallow-depth auxiliary loss is active for this run."""
    return bool(_get_shallow_depth_config(config).get("enabled", False))


def _shallow_depth_lambda(config: Dict) -> float:
    """Configurable multiplier for the shallow-depth auxiliary term."""
    shallow_cfg = _get_shallow_depth_config(config)
    return float(shallow_cfg.get("lambda_shallow", shallow_cfg.get("weight", 0.1)))


def _require_wd_normalization_stats(config: Dict) -> Dict:
    """Return WD normalization stats or fail loudly for shallow-depth training."""
    normalization_stats = _get_runtime_normalization_stats(config)
    if not normalization_stats:
        raise ValueError(
            "loss.shallow_depth_loss.enabled=true requires WD normalization stats, "
            "but config['_runtime']['normalization_stats'] is missing or empty. "
            "Run training through train_gnn4cf.py "
            "with a valid paths.normalization_stats_path or generated graph output "
            "that contains normalization_stats.json."
        )
    if "wd" not in normalization_stats:
        raise ValueError(
            "loss.shallow_depth_loss.enabled=true requires "
            "config['_runtime']['normalization_stats']['wd'], but the 'wd' key "
            f"is missing. Available stats keys: {list(normalization_stats.keys())}. "
            "Provide water-depth statistics in the graph/config normalization_stats.json."
        )
    return normalization_stats


def _depth_delta_m_to_normalized(delta_m: float, stats_dict: Dict, reference_depth_m: float = 0.0) -> float:
    """
    Convert a meter-scale depth delta to an approximate normalized-space delta.

    For linear normalizations this is exact. For log1p_zscore it is evaluated
    locally around `reference_depth_m`, which preserves the intended meter-scale
    meaning of huber_beta_m while optimizing in normalized WD space.
    """
    delta_m = float(delta_m)
    reference_depth_m = float(reference_depth_m)
    if delta_m <= 0.0:
        return 1e-8
    lower = convert_real_threshold_to_normalized(reference_depth_m, stats_dict, var_name="wd")
    upper = convert_real_threshold_to_normalized(reference_depth_m + delta_m, stats_dict, var_name="wd")
    return max(abs(float(upper) - float(lower)), 1e-8)


def compute_shallow_depth_loss(pred_wd, target_wd, config: Dict):
    """
    Compute an auxiliary WD loss only on shallow positive water depths.

    Config thresholds remain in meters, but they are converted to normalized WD
    thresholds before masking:
      min_depth_norm <= target_wd_norm < max_depth_norm

    The optimization loss is computed in normalized space to match the main
    regression/flood-aware objectives. Meter-space RMSE/MAE are still reported
    as diagnostics after selecting the same shallow nodes.
    """
    shallow_cfg = _get_shallow_depth_config(config)
    zero = pred_wd.sum() * 0.0
    zero_metric = torch.zeros((), device=pred_wd.device, dtype=pred_wd.dtype)

    if not shallow_cfg.get("enabled", False):
        return None

    normalization_stats = _require_wd_normalization_stats(config)
    min_depth_m = float(shallow_cfg.get("min_depth_m", 0.01))
    max_depth_m = float(shallow_cfg.get("max_depth_m", shallow_cfg.get("threshold_m", 0.5)))
    if max_depth_m <= min_depth_m:
        raise ValueError(
            "loss.shallow_depth_loss.max_depth_m must be greater than "
            f"min_depth_m (got {max_depth_m} <= {min_depth_m})."
        )

    min_depth_norm = convert_real_threshold_to_normalized(
        min_depth_m, normalization_stats, var_name="wd"
    )
    max_depth_norm = convert_real_threshold_to_normalized(
        max_depth_m, normalization_stats, var_name="wd"
    )
    if max_depth_norm <= min_depth_norm:
        raise ValueError(
            "Converted shallow-depth normalized thresholds are not increasing: "
            f"min_depth_m={min_depth_m} -> {min_depth_norm}, "
            f"max_depth_m={max_depth_m} -> {max_depth_norm}. "
            "Check WD normalization stats in normalization_stats.json."
        )

    shallow_mask = (target_wd >= min_depth_norm) & (target_wd < max_depth_norm)
    active_count = shallow_mask.sum().to(dtype=pred_wd.dtype)
    active_fraction = shallow_mask.float().mean()

    if not shallow_mask.any():
        return {
            "shallow_depth_loss": zero,
            "shallow_depth_active_fraction": active_fraction.detach(),
            "shallow_depth_active_count": active_count.detach(),
            "shallow_depth_rmse_m": zero_metric,
            "shallow_depth_mae_m": zero_metric,
        }

    pred_sel_norm = pred_wd[shallow_mask]
    target_sel_norm = target_wd[shallow_mask]
    diff_norm = pred_sel_norm - target_sel_norm

    loss_type = str(shallow_cfg.get("loss_type", "smooth_l1")).lower()
    if loss_type in ("smooth_l1", "huber"):
        beta = _depth_delta_m_to_normalized(
            float(shallow_cfg.get("huber_beta_m", 0.05)),
            normalization_stats,
            reference_depth_m=min_depth_m,
        )
        abs_diff = torch.abs(diff_norm)
        element_loss = torch.where(
            abs_diff < beta,
            0.5 * diff_norm ** 2 / beta,
            abs_diff - 0.5 * beta,
        )
        shallow_loss = element_loss.mean()
    elif loss_type == "mse":
        shallow_loss = torch.mean(diff_norm ** 2)
    elif loss_type == "mae":
        shallow_loss = torch.mean(torch.abs(diff_norm))
    elif loss_type == "relative_mse":
        shallow_loss = torch.sum(diff_norm ** 2) / (torch.sum(target_sel_norm ** 2) + 1e-8)
    else:
        raise ValueError(
            f"Unknown loss.shallow_depth_loss.loss_type='{loss_type}'. "
            "Supported: smooth_l1, huber, mse, mae, relative_mse."
        )

    with torch.no_grad():
        pred_sel_m = denormalize_tensor(
            pred_sel_norm.detach(),
            normalization_stats,
            var_name="wd",
            clamp_nonnegative=False,
        )
        target_sel_m = denormalize_tensor(
            target_sel_norm.detach(),
            normalization_stats,
            var_name="wd",
            clamp_nonnegative=True,
        )
        diff_m = pred_sel_m - target_sel_m
        rmse_m = torch.sqrt(torch.mean(diff_m ** 2) + 1e-8)
        mae_m = torch.mean(torch.abs(diff_m))

    return {
        "shallow_depth_loss": shallow_loss,
        "shallow_depth_active_fraction": active_fraction.detach(),
        "shallow_depth_active_count": active_count.detach(),
        "shallow_depth_rmse_m": rmse_m,
        "shallow_depth_mae_m": mae_m,
    }


# =============================================================================
# Compound-flood regression and auxiliary losses
# =============================================================================

def compound_gnn_loss(
        preds_node: torch.Tensor,
        targets_node: torch.Tensor,
        comp_mask: torch.Tensor,
        config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Compute regression and optional shallow-depth/classification losses.

    Predictions and targets use [steps, nodes, labels] ordering. Supervision
    is restricted to computational nodes, preserving configured label order
    and normalization. Auxiliary volume and geometric representations are
    not required by the loss.
    """
    loss_config = config.get("loss", {})
    window_config = config.get("window", {})

    # preds_node shape: [num_steps, num_nodes, num_labels]
    num_steps = preds_node.shape[0]

    # --- Apply Mask ---
    # We only care about computational nodes
    preds_node = preds_node[:, comp_mask, :]
    targets_node = targets_node[:, comp_mask, :]

    if preds_node.numel() == 0 or targets_node.numel() == 0:
        # Handle empty batch or case with no comp nodes
        return {"total_loss": torch.tensor(0.0, device=preds_node.device, requires_grad=True)}

    # --- Get label ordering from config ---
    label_vars = window_config.get("label_vars", ["wd", "vx", "vy"])
    n_labels = len(label_vars)

    # --- Map label names to indices for slicing ---
    label_indices = {var: idx for idx, var in enumerate(label_vars)}

    # --- Load normalization stats and convert threshold (if flood-aware is enabled) ---
    threshold_normalized = None
    threshold_real = None
    normalization_stats = None
    flood_aware_config = loss_config.get("flood_aware", {})
    flood_aware_enabled = flood_aware_config.get("enabled", False)

    if flood_aware_enabled:
        # Get threshold in real units from config
        # Ensure threshold_real is a float (YAML might load it as string)
        threshold_real = float(flood_aware_config.get("threshold_real", 0.0))

        # Check if threshold_normalized was pre-computed (performance optimization)
        # This avoids loading stats file from disk on every batch
        if "threshold_normalized" in flood_aware_config:
            # Ensure threshold_normalized is a float (YAML might load it as string)
            threshold_normalized = float(flood_aware_config["threshold_normalized"])
            # Check if normalization stats are also pre-stored (for threshold conversion and classification loss)
            if "normalization_stats" in flood_aware_config:
                normalization_stats = flood_aware_config["normalization_stats"]
        else:
            # Fallback: Load normalization stats (for threshold conversion and classification loss)
            try:
                from gnn4cf_graph_builder import load_stats_file
                paths_cfg = config.get("paths", {})
                stats_path = paths_cfg.get(
                    "normalization_stats_path",
                    os.path.join(
                        paths_cfg.get("output_dir", "output_data"),
                        paths_cfg.get("normalization_stats_filename", "normalization_stats.json"),
                    ),
                )

                if os.path.exists(stats_path):
                    normalization_stats = load_stats_file(stats_path)
                    if "wd" in normalization_stats:
                        threshold_normalized = convert_real_threshold_to_normalized(
                            threshold_real, normalization_stats, var_name="wd"
                        )
                    else:
                        print(f"    Warning: 'wd' not found in normalization stats. Using threshold_real={threshold_real}m directly.")
                        threshold_normalized = threshold_real
                else:
                    print(f"    Warning: Normalization stats not found at {stats_path}")
                    print(f"     Using threshold_real={threshold_real}m directly (assuming no normalization)")
                    threshold_normalized = threshold_real
            except Exception as e:
                print(f"    Warning: Could not load normalization stats: {e}")
                print(f"     Using threshold_real={threshold_real}m directly")
                threshold_normalized = threshold_real

    # --- 1. Main Loss Calculation (config-driven) ---
    loss_type = loss_config.get("loss_type", "hybrid")
    lambda_rel = loss_config.get("lambda_relative", 0.5)
    lambda_abs = loss_config.get("lambda_abs", 0.5)

    def compute_loss(pred, target, loss_type_str):
        """Compute loss based on loss_type."""
        if loss_type_str == "rmse":
            return torch.sqrt(F.mse_loss(pred, target) + 1e-8)
        elif loss_type_str == "mse":
            return F.mse_loss(pred, target)
        elif loss_type_str == "mae":
            return F.l1_loss(pred, target)
        elif loss_type_str == "relative_mse":
            return torch.sum((pred - target) ** 2) / (torch.sum(target ** 2) + 1e-8)
        elif loss_type_str == "hybrid":
            rel_loss = torch.sum((pred - target) ** 2) / (torch.sum(target ** 2) + 1e-8)
            abs_loss = F.l1_loss(pred, target, reduction="mean")
            return lambda_rel * rel_loss + lambda_abs * abs_loss
        else:
            # Default to hybrid if unknown
            rel_loss = torch.sum((pred - target) ** 2) / (torch.sum(target ** 2) + 1e-8)
            abs_loss = F.l1_loss(pred, target, reduction="mean")
            return lambda_rel * rel_loss + lambda_abs * abs_loss

    # --- Slice predictions and targets by label order and compute losses ---
    variable_losses = {}
    weighted_cfg = flood_aware_config.get("weighted", {})
    weighted_enabled = flood_aware_enabled and weighted_cfg.get("enabled", False)
    apply_weighted_to = weighted_cfg.get("apply_to", [])

    for var in label_vars:
        var_idx = label_indices[var]
        pred_var = preds_node[..., var_idx:var_idx + 1]
        target_var = targets_node[..., var_idx:var_idx + 1]

        # Check if weighted loss should be applied to this variable
        if weighted_enabled and var in apply_weighted_to and threshold_normalized is not None:
            # Use weighted regression loss with bounded exponential weighting (Option B)
            weight_non_flooded = weighted_cfg.get("weight_non_flooded", 1.0)
            # Accept alpha or its supported exponential_coefficient alias.
            alpha = weighted_cfg.get("alpha", None)
            if alpha is None:
                # Use the supported coefficient alias when alpha is absent.
                exponential_coefficient = weighted_cfg.get("exponential_coefficient", None)
                alpha = exponential_coefficient  # Use as alpha if provided
            max_weight = weighted_cfg.get("max_weight", None)

            # Prepare bounded exponential weighting parameters
            # alpha is the rate parameter, max_weight is required when alpha is provided
            alpha_param = alpha if alpha is not None and alpha > 0 else None

            var_loss = compute_weighted_regression_loss(
                pred_var, target_var, threshold_normalized,
                weight_non_flooded, loss_type,
                lambda_rel, lambda_abs,
                alpha=alpha_param,
                max_weight=max_weight
            )
        else:
            # Standard loss
            var_loss = compute_loss(pred_var, target_var, loss_type)

        variable_losses[var] = var_loss

    # --- Apply per-variable weights ---
    weighted_variable_losses = {}
    for var in label_vars:
        weight_key = f"weight_{var}"  # "weight_wd", "weight_vx", "weight_vy"
        var_weight = loss_config.get(weight_key, 1.0)  # Default 1.0 if not specified
        weighted_variable_losses[var] = var_weight * variable_losses[var]

    # --- Sum of weighted variable losses, averaged over steps ---
    main_loss = sum(weighted_variable_losses.values()) / num_steps

    # --- 2. Optional shallow-depth auxiliary loss ---
    # This term is separate from the main regression loss. It only looks at WD
    # targets in a configurable shallow band, e.g. 0.01 m <= depth < 0.5 m.
    shallow_depth_result = None
    shallow_depth_loss = preds_node.sum() * 0.0
    lambda_shallow = 0.0
    if _shallow_depth_enabled(config):
        if "wd" not in label_indices:
            raise ValueError(
                "loss.shallow_depth_loss.enabled=true requires 'wd' in "
                "config['window']['label_vars'] because the auxiliary loss is "
                "defined for water depth."
            )
        wd_idx = label_indices["wd"]
        shallow_depth_result = compute_shallow_depth_loss(
            preds_node[..., wd_idx:wd_idx + 1],
            targets_node[..., wd_idx:wd_idx + 1],
            config,
        )
        if shallow_depth_result is not None:
            shallow_depth_loss = shallow_depth_result["shallow_depth_loss"]
            lambda_shallow = _shallow_depth_lambda(config)

    # --- 4. Classification Loss (if enabled) ---
    classification_loss = torch.tensor(0.0, device=preds_node.device, requires_grad=True)
    classification_accuracy = torch.tensor(0.0, device=preds_node.device)
    classification_precision = torch.tensor(0.0, device=preds_node.device)
    classification_recall = torch.tensor(0.0, device=preds_node.device)
    classification_f1 = torch.tensor(0.0, device=preds_node.device)

    classification_cfg = flood_aware_config.get("classification", {})
    classification_enabled = flood_aware_enabled and classification_cfg.get("enabled", False)
    apply_classification_to = classification_cfg.get("apply_to", [])

    if classification_enabled and threshold_normalized is not None:
        # Apply classification loss to specified variables
        classification_losses = []
        classification_metric_buckets = {
            "classification_accuracy": [],
            "classification_precision": [],
            "classification_recall": [],
            "classification_f1": [],
        }

        for var in apply_classification_to:
            if var in label_vars:
                var_idx = label_indices[var]
                pred_var = preds_node[..., var_idx:var_idx + 1]
                target_var = targets_node[..., var_idx:var_idx + 1]

                temperature = classification_cfg.get("temperature", 1.0)
                var_cls_loss, var_cls_metrics = compute_classification_loss(
                    pred_var, target_var, threshold_normalized, temperature
                )
                classification_losses.append(var_cls_loss)
                for metric_name, metric_value in var_cls_metrics.items():
                    classification_metric_buckets[metric_name].append(metric_value)

        if classification_losses:
            classification_loss = sum(classification_losses) / len(classification_losses)
            classification_accuracy = (
                sum(classification_metric_buckets["classification_accuracy"])
                / len(classification_metric_buckets["classification_accuracy"])
            )
            classification_precision = (
                sum(classification_metric_buckets["classification_precision"])
                / len(classification_metric_buckets["classification_precision"])
            )
            classification_recall = (
                sum(classification_metric_buckets["classification_recall"])
                / len(classification_metric_buckets["classification_recall"])
            )
            classification_f1 = (
                sum(classification_metric_buckets["classification_f1"])
                / len(classification_metric_buckets["classification_f1"])
            )

    lambda_classification = classification_cfg.get("lambda_classification", 0.5) if classification_enabled else 0.0

    # --- 5. Total Loss ---
    total_loss = (
        main_loss
        + lambda_shallow * shallow_depth_loss
        + lambda_classification * classification_loss
    )

    # --- 6. Metrics (RMSE) ---
    with torch.no_grad():
        rmse_metrics = {}
        for var in label_vars:
            var_idx = label_indices[var]
            pred_var = preds_node[..., var_idx:var_idx + 1]
            target_var = targets_node[..., var_idx:var_idx + 1]
            rmse_metrics[f"rmse_{var}"] = torch.sqrt(F.mse_loss(pred_var, target_var) + 1e-8)
        rmse_total = torch.sqrt(F.mse_loss(preds_node, targets_node) + 1e-8)

    # --- Build return dict with dynamic keys ---
    return_dict = {
        "total_loss": total_loss,
        "main_loss": main_loss,
    }

    # Add per-variable losses (use weighted losses for consistency)
    for var in label_vars:
        return_dict[f"loss_{var}"] = weighted_variable_losses[var]

    # Add per-variable RMSE metrics
    for var in label_vars:
        return_dict[f"rmse_{var}"] = rmse_metrics[f"rmse_{var}"]

    return_dict["rmse_total"] = rmse_total

    # Add classification metrics (if enabled)
    if classification_enabled:
        return_dict["classification_loss"] = classification_loss
        return_dict["classification_accuracy"] = classification_accuracy
        return_dict["classification_precision"] = classification_precision
        return_dict["classification_recall"] = classification_recall
        return_dict["classification_f1"] = classification_f1

    # Add shallow-depth auxiliary metrics only when the term is enabled.
    if shallow_depth_result is not None:
        return_dict.update(shallow_depth_result)

    return return_dict


# =============================================================================
# 3. Temporal node-feature helpers
# =============================================================================

def _get_dynamic_dims(x_full, n_static_features, past_steps):
    """Helper to calculate dynamic feature dimensions from the x tensor."""
    N = x_full.shape[0]
    n_dynamic_flat = x_full.shape[1] - n_static_features

    if n_dynamic_flat <= 0 or n_dynamic_flat % past_steps != 0:
        raise ValueError(
            f"Dynamic features ({n_dynamic_flat}) not divisible by past_steps ({past_steps}).\n"
            f"Check config: past_steps vs. dynamic feature list."
        )
    return n_dynamic_flat // past_steps


def prepare_model_input(x_full, n_static_features, past_steps, model_history_steps):
    """
    Prepares model inputs for the "Real Loss" branch.
    It slices the full `past_steps` window to get the *last* `model_history_steps`.

    Example (p=1, h=2, T_past=3): Slices (t-2, t-1, t) -> (t-1, t)
    Example (p=2, h=2, T_past=4): Slices (t-3, t-2, t-1, t) -> (t-1, t)
    Example (p=3, h=3, T_past=6): Slices (..., t-2, t-1, t) -> (t-2, t-1, t)
    """
    n_dynamic_vars = _get_dynamic_dims(x_full, n_static_features, past_steps)
    N = x_full.shape[0]

    x_static = x_full[:, :n_static_features]
    x_dynamic_flat = x_full[:, n_static_features:]

    # Reshape: [N, T_past * F_dyn] -> [N, T_past, F_dyn]
    x_dynamic = x_dynamic_flat.reshape(N, past_steps, n_dynamic_vars)

    # Slice the time dimension to get the last `h` steps
    x_dynamic_trimmed = x_dynamic[:, -model_history_steps:, :]

    # Flatten back and recombine
    x_dynamic_trimmed_flat = x_dynamic_trimmed.reshape(N, -1)
    return torch.cat([x_static, x_dynamic_trimmed_flat], dim=1)


def prepare_model_input_stability(x_full, n_static_features, past_steps, model_history_steps):
    """
    Prepares model inputs for the "Stability Loss" seed prediction.
    It slices the full `past_steps` window to get the *first* `model_history_steps`.

    Example (p=1, h=2, T_past=3): Slices (t-2, t-1, t) -> (t-2, t-1)
    Example (p=2, h=2, T_past=4): Slices (t-3, t-2, t-1, t) -> (t-3, t-2)
    Example (p=3, h=3, T_past=6): Slices (t-5, ..., t) -> (t-5, t-4, t-3)
    """
    n_dynamic_vars = _get_dynamic_dims(x_full, n_static_features, past_steps)
    N = x_full.shape[0]

    x_static = x_full[:, :n_static_features]
    x_dynamic_flat = x_full[:, n_static_features:]

    # Reshape: [N, T_past * F_dyn] -> [N, T_past, F_dyn]
    x_dynamic = x_dynamic_flat.reshape(N, past_steps, n_dynamic_vars)

    # Slice the time dimension to get the first `h` steps
    x_dynamic_trimmed = x_dynamic[:, :model_history_steps, :]

    # Flatten back and recombine
    x_dynamic_trimmed_flat = x_dynamic_trimmed.reshape(N, -1)
    return torch.cat([x_static, x_dynamic_trimmed_flat], dim=1)


def update_graph_with_seed(x_full, seed_pred_chunk, n_static_features,
                           past_steps, n_state_vars, n_driver_vars, predictor_step):
    """
    Updates the 'full' graph tensor by replacing the data in the
    *last* `predictor_step` time steps with the noisy seed prediction.

    Example (p=1, h=2, T_past=3): Replaces (t) with (t_pred, t_drivers)
    Example (p=2, h=2, T_past=4): Replaces (t-1, t) with (t-1_pred, t-1_drivers)
                                 and (t_pred, t_drivers)
    """
    n_dyn_total = n_state_vars + n_driver_vars
    N = x_full.shape[0]

    x_static = x_full[:, :n_static_features]
    x_dynamic_flat = x_full[:, n_static_features:]

    # Reshape: [N, T_past * F_dyn] -> [N, T_past, F_dyn]
    x_dynamic = x_dynamic_flat.reshape(N, past_steps, n_dyn_total)

    # Keep all steps *except* the last `p` ones
    # (p=1, T_past=3) -> keeps [0, 1] (t-2, t-1)
    # (p=2, T_past=4) -> keeps [0, 1] (t-3, t-2)
    history_part = x_dynamic[:, :-predictor_step, :]

    # Get the *true drivers* from the steps we are replacing
    drivers_part = x_dynamic[:, -predictor_step:, n_state_vars:]  # Shape: [N, p, F_driver]

    # Reshape the seed prediction (which is just state)
    seed_state_part = seed_pred_chunk.reshape(N, predictor_step, n_state_vars)  # Shape: [N, p, F_state]

    # Combine the *predicted* state with the *true* drivers
    new_seed_chunk = torch.cat([seed_state_part, drivers_part], dim=2)  # Shape: [N, p, F_dyn]

    # Recombine and flatten
    x_dynamic_new = torch.cat([history_part, new_seed_chunk], dim=1)
    x_dynamic_new_flat = x_dynamic_new.reshape(N, -1)

    return torch.cat([x_static, x_dynamic_new_flat], dim=1)


def use_prediction_pushforward(x_current, pred_chunk, driver_chunk,
                               n_static_features, past_steps,
                               n_state_vars, n_driver_vars, predictor_step):
    """
    The core pushforward function. Shifts the graph state by `predictor_step`.
    - Drops the oldest `p` steps.
    - Appends a new chunk of `(predicted_state, true_drivers)`.
    """
    n_dyn_total = n_state_vars + n_driver_vars
    N = x_current.shape[0]

    x_static = x_current[:, :n_static_features]
    x_dynamic_flat = x_current[:, n_static_features:]

    # Reshape: [N, T_past * F_dyn] -> [N, T_past, F_dyn]
    x_dynamic = x_dynamic_flat.reshape(N, past_steps, n_dyn_total)

    # Drop the oldest `p` steps
    shifted_history = x_dynamic[:, predictor_step:, :]  # Shape: [N, T_past - p, F_dyn]

    # Reshape inputs to [N, p, F_vars]
    pred_state_block = pred_chunk.reshape(N, predictor_step, n_state_vars)
    driver_block = driver_chunk.reshape(N, predictor_step, n_driver_vars)

    # Combine new state and drivers
    new_chunk = torch.cat([pred_state_block, driver_block], dim=2)  # Shape: [N, p, F_dyn]

    # Append new chunk and flatten
    x_dynamic_new = torch.cat([shifted_history, new_chunk], dim=1)
    x_dynamic_new_flat = x_dynamic_new.reshape(N, -1)

    return torch.cat([x_static, x_dynamic_new_flat], dim=1)


# =============================================================================
# 5. Rollout Inference Functions
# =============================================================================

def prepare_model_input_for_rollout(x_full, n_static_features, past_steps, predictor_step):
    """
    Rollout: slice h=max(2, predictor_step) slices from the current window.

    Always uses the LAST h slices (standard autoregressive behavior).
    This ensures the model sees the most recent information, including its own predictions
    in subsequent rollout steps.

    After pushforward, the window structure is:
    - Oldest p slices are dropped
    - New p slices (predictions) are appended
    - The last h slices of the shifted window contain the most recent information
      (including predictions), which is appropriate for the next prediction.

    This *must not* include the (unknown) true state at the prediction target time.
    Assumes x_full holds exactly past_steps = h + predictor_step.

    Returns tensor with [static, h * F_dyn] flattened.

    Example (p=1, h=2, T_past=3):
        Initial window: [t-2, t-1, t] -> Input: [t-1, t] (last h slices)
        After pushforward: [t-1, t, t+1_pred] -> Input: [t, t+1_pred] (last h slices, includes prediction)
    Example (p=2, h=2, T_past=4):
        Initial window: [t-3, t-2, t-1, t] -> Input: [t-1, t] (last h=2 slices)
        After pushforward: [t-1, t, t+1_pred, t+2_pred] -> Input: [t+1_pred, t+2_pred] (last h=2 slices)
    """
    h = max(2, predictor_step)
    n_dynamic_vars = _get_dynamic_dims(x_full, n_static_features, past_steps)  # F_dyn
    N = x_full.shape[0]

    x_static = x_full[:, :n_static_features]
    x_dyn = x_full[:, n_static_features:].reshape(N, past_steps, n_dynamic_vars)

    # Always use last h slices (standard autoregressive behavior)
    x_dyn_h = x_dyn[:, -h:, :]          # shape [N, h, F_dyn]

    x_dyn_h_flat = x_dyn_h.reshape(N, -1)

    return torch.cat([x_static, x_dyn_h_flat], dim=1)


@torch.no_grad()
def rollout_autoregressive(
    model, initial_x_full, edge_index, edge_attr, node_type, edge_type,
    future_drivers, n_static_features, past_steps, n_state_vars, n_driver_vars,
    predictor_step,
    probe_nodes=None, label_vars=("wd", "vx", "vy"),
    driver_vars=("acc_rainfall", "sea_level", "sea_level_trend"),
    debug=False, y_truth=None, debug_save_path=None
):
    """
    Autoregressive rollout:
      - Start from initial window x_full holding exactly past_steps = h + p
      - For the FIRST step: use FIRST h slices of initial window (to preserve early test information)
      - For SUBSEQUENT steps: use FIRST h slices of rolled-forward window (contiguous history)
      - Repeatedly:
         1) build input from h slices (first h for rollout, preserving early context),
         2) predict next p state-slices,
         3) pushforward window using (pred_state, true_driver_chunk),
         4) accumulate predictions until all future drivers are consumed.

    Args:
        model: The GNN model
        initial_x_full: [N, n_static + past_steps*(n_state+n_driver)] initial window
        edge_index, edge_attr, node_type, edge_type: Graph structure
        future_drivers: [N, future_steps_rollout * n_driver_vars] true driver values (future portion only)
        n_static_features, past_steps, n_state_vars, n_driver_vars, predictor_step: Config params
        probe_nodes: List of node indices for debug printing
        label_vars: Tuple of label variable names
        driver_vars: Tuple of driver variable names
        debug: Enable debug printing
        y_truth: Optional [N, future_steps_rollout, n_label_vars] ground truth for no-leakage check
        debug_save_path: Optional path to save debug data as JSON file (only if debug=True)

    Returns:
        preds: [T_future_rollout, N, n_label_vars] predictions
    """
    device = initial_x_full.device
    h = max(2, predictor_step)
    N = initial_x_full.shape[0]
    n_label_vars = len(label_vars)

    # Derive future_steps from driver tensor shape (not from config)
    future_steps_rollout = future_drivers.shape[1] // n_driver_vars
    assert future_drivers.shape[1] % n_driver_vars == 0, \
        f"future_drivers dim ({future_drivers.shape[1]}) not divisible by n_driver_vars ({n_driver_vars})"
    assert future_steps_rollout > 0, \
        f"Invalid future_drivers length: {future_drivers.shape[1]} (must be > 0)"

    # Calculate number of complete chunks (handle remainder by truncating)
    num_chunks = future_steps_rollout // predictor_step
    assert num_chunks > 0, \
        f"Cannot create chunks: future_steps_rollout={future_steps_rollout}, predictor_step={predictor_step}"

    # Truncate future_drivers to exact multiple of (num_chunks * predictor_step * n_driver_vars)
    # This handles cases where future_steps_rollout is not divisible by predictor_step
    future_steps_usable = num_chunks * predictor_step
    if future_steps_usable < future_steps_rollout:
        remainder_steps = future_steps_rollout - future_steps_usable
        print(f"  Warning: future_steps_rollout ({future_steps_rollout}) not divisible by predictor_step ({predictor_step})")
        print(f"  Truncating {remainder_steps} step(s) to fit {num_chunks} complete chunks ({future_steps_usable} steps)")
        future_drivers = future_drivers[:, :future_steps_usable * n_driver_vars]
        future_steps_rollout = future_steps_usable
        # Also truncate y_truth if provided to match
        if y_truth is not None and y_truth.shape[1] > future_steps_rollout:
            y_truth = y_truth[:, :future_steps_rollout, :]

    # Shape checks
    assert initial_x_full.shape[1] == n_static_features + past_steps * (n_state_vars + n_driver_vars), \
        f"initial_x_full dim mismatch: got {initial_x_full.shape[1]}, expected {n_static_features + past_steps * (n_state_vars + n_driver_vars)}"

    x_curr = initial_x_full.clone()
    pred_chunks = []

    # Reshape drivers into chunks: [N, future_steps_rollout * n_driver] -> [num_chunks, N, p * n_driver]
    # Each drivers_chunks[i] corresponds to the exact time indices for the i-th prediction chunk
    drivers_chunks = future_drivers.reshape(N, num_chunks, predictor_step * n_driver_vars).permute(1, 0, 2)

    # No-leakage check: verify first input doesn't include true state at prediction target
    if debug and y_truth is not None and num_chunks > 0:
        # We use last h slices, so the last slice in input is at index past_steps-1
        # The prediction target starts at index past_steps (which is the first future step)
        x_in_first = prepare_model_input_for_rollout(x_curr, n_static_features, past_steps, predictor_step)
        x_dyn_first = x_in_first[:, n_static_features:].reshape(N, h, n_state_vars + n_driver_vars)
        last_state_slice = x_dyn_first[:, -1, :n_state_vars]  # [N, n_state_vars] at index h-1
        gt_t0 = y_truth[:, 0, :] if y_truth.shape[1] > 0 else None  # [N, n_label_vars] at prediction target

        if gt_t0 is not None:
            # Check that last_state_slice (at h-1) != gt_t0 (at h) for at least one node
            max_diff = (last_state_slice - gt_t0[:, :n_state_vars]).abs().max().item()
            if max_diff < 1e-6:
                print(f"[WARNING] Potential leakage detected: last input slice matches GT at prediction target (diff={max_diff:.2e})")
            else:
                print(f"[OK] No leakage: last input slice (index h-1) differs from GT at prediction target (max_diff={max_diff:.6f})")

    # Store last prediction from previous chunk for continuity check
    last_pred_from_prev_chunk = None

    # Initialize debug data structure for JSON export
    debug_data = None
    if debug and debug_save_path:
        debug_data = {
            "rollout_info": {
                "num_nodes": int(N),
                "num_chunks": int(num_chunks),
                "predictor_step": int(predictor_step),
                "past_steps": int(past_steps),
                "model_history_steps": int(h),
                "future_steps_rollout": int(future_steps_rollout),
                "n_label_vars": int(n_label_vars),
                "n_state_vars": int(n_state_vars),
                "n_driver_vars": int(n_driver_vars),
                "label_vars": list(label_vars),
                "driver_vars": list(driver_vars),
                "probe_nodes": list(probe_nodes) if probe_nodes else []
            },
            "chunks": []
        }

    for i in range(num_chunks):
        # Initialize chunk data for JSON export (if debug mode enabled)
        chunk_data = None
        if debug_data:
            chunk_data = {
                "chunk_index": i,
                "residual_extraction": {},
                "cascading_within_chunk": {},
                "pushforward": {}
            }

        # 1) Build input: use last h slices from window (model_history_steps)
        #    - This follows the unified rule: model input = last h slices from past_steps window
        #    - For past_steps=3, h=2: takes [t=1, t=2] from [t=0, t=1, t=2]
        #    - After pushforward: takes last h slices from shifted window (maintains window size)
        x_in = prepare_model_input_for_rollout(x_curr, n_static_features, past_steps, predictor_step)

        # ===== DEBUG: Check residual extraction BEFORE prediction =====
        if debug and probe_nodes:
            # Extract what the model will use as residual (last known value from x_in)
            x_in_dyn = x_in[:, n_static_features:].reshape(N, h, n_state_vars + n_driver_vars)
            # The model extracts residual from the LAST slice (index h-1) of x_in
            # This corresponds to the last timestep in the input window
            residual_from_x_in = x_in_dyn[:, -1, :n_state_vars]  # [N, n_state_vars] = last slice state

            print(f"\n{'='*80}")
            print(f"[RESIDUAL DEBUG] Chunk {i}: Checking residual extraction")
            print(f"{'='*80}")

            for nidx in probe_nodes:
                if nidx < N:
                    # Extract residual that model will use (from last slice of x_in)
                    residual_wd = residual_from_x_in[nidx, 0].item()  # Assuming wd is first label var

                    # Collect data for JSON
                    node_residual_data = {
                        "residual_from_x_in": {label_vars[0]: float(residual_wd)},
                        "last_slice_state": [float(x) for x in x_in_dyn[nidx, -1, :n_state_vars].cpu().numpy()],
                        "last_slice_driver": [float(x) for x in x_in_dyn[nidx, -1, n_state_vars:].cpu().numpy()]
                    }

                    # For chunk i > 0, this should match the last prediction from chunk i-1
                    if i > 0 and last_pred_from_prev_chunk is not None:
                        last_pred_wd = last_pred_from_prev_chunk[nidx, 0].item()
                        diff = abs(residual_wd - last_pred_wd)
                        match_status = "MATCH" if diff < 1e-5 else f"MISMATCH (diff={diff:.6f})"
                        print(f"  Node {nidx}: residual_from_x_in[wd]={residual_wd:.6f}, "
                              f"last_pred_from_chunk_{i-1}[wd]={last_pred_wd:.6f} {match_status}")

                        node_residual_data["last_pred_from_prev_chunk"] = {label_vars[0]: float(last_pred_wd)}
                        node_residual_data["continuity_check"] = {
                            "matches": bool(diff < 1e-5),
                            "difference": float(diff)
                        }
                    else:
                        print(f"  Node {nidx}: residual_from_x_in[wd]={residual_wd:.6f} (chunk 0, no previous prediction)")
                        node_residual_data["continuity_check"] = {"matches": None, "difference": None}

                    if chunk_data:
                        chunk_data["residual_extraction"][f"node_{nidx}"] = node_residual_data

                    # Also show the full last slice for context
                    last_slice_full = x_in_dyn[nidx, -1, :]
                    print(f"    Last slice in x_in: state={last_slice_full[:n_state_vars].cpu().numpy()}, "
                          f"driver={last_slice_full[n_state_vars:].cpu().numpy()}")

        # 2) Predict next p state(s)
        y_pred_chunk = model(x_in, edge_index, edge_attr, node_type, edge_type)  # [N, p*n_label]

        # ===== DEBUG: Check cascading within chunk =====
        if debug and probe_nodes:
            y_pred_reshaped = y_pred_chunk.reshape(N, predictor_step, n_label_vars)  # [N, p, n_label_vars]

            print(f"\n[RESIDUAL DEBUG] Chunk {i}: Checking cascading within chunk")
            print(f"  Model predicts deltas, then adds residuals in cascading fashion:")
            print(f"    Step 1: y(t+1) = delta_1 + y(t)")
            print(f"    Step 2: y(t+2) = delta_2 + y(t+1)")
            print(f"    Step 3: y(t+3) = delta_3 + y(t+2)")
            print(f"    ...")

            for nidx in probe_nodes:
                if nidx < N:
                    # Get residual that was used (from last slice of x_in)
                    residual_wd = residual_from_x_in[nidx, 0].item()

                    # Check cascading: each step should build on previous
                    print(f"\n  Node {nidx} cascading check:")
                    print(f"    Residual (y(t)): {residual_wd:.6f}")

                    # Collect cascading data for JSON
                    cascading_steps = []

                    # Reconstruct what the model did internally
                    # The model adds residual to first step, then uses that for next step
                    prev_value = residual_wd
                    for step_idx in range(predictor_step):
                        pred_wd = y_pred_reshaped[nidx, step_idx, 0].item()
                        delta_implied = pred_wd - prev_value
                        print(f"    Step {step_idx+1}: pred={pred_wd:.6f}, "
                              f"delta_implied={delta_implied:.6f}, "
                              f"prev_value={prev_value:.6f}")

                        # Verify: pred should equal prev_value + delta
                        expected = prev_value + delta_implied
                        matches = abs(pred_wd - expected) <= 1e-5
                        if not matches:
                            print(f"        WARNING: Cascading mismatch! "
                                  f"pred={pred_wd:.6f} != prev+delta={expected:.6f}")

                        cascading_steps.append({
                            "step": int(step_idx + 1),
                            "prediction": {label_vars[0]: float(pred_wd)},
                            "delta_implied": {label_vars[0]: float(delta_implied)},
                            "prev_value": {label_vars[0]: float(prev_value)},
                            "matches_cascading": bool(matches)
                        })

                        prev_value = pred_wd  # For next step

                    if chunk_data:
                        chunk_data["cascading_within_chunk"][f"node_{nidx}"] = {
                            "residual": {label_vars[0]: float(residual_wd)},
                            "steps": cascading_steps
                        }

        # Debug: Verify prediction shape matches expectations
        if debug and i == 0:
            expected_shape = (N, predictor_step * n_label_vars)
            if y_pred_chunk.shape != expected_shape:
                print(f"[ERROR] Prediction shape mismatch! Got {y_pred_chunk.shape}, expected {expected_shape}")
            else:
                print(f"[DEBUG] Prediction shape OK: {y_pred_chunk.shape} = [N={N}, p*n_label={predictor_step * n_label_vars}]")

        pred_chunks.append(y_pred_chunk)

        # Store last prediction from this chunk for next chunk's continuity check
        y_pred_reshaped = y_pred_chunk.reshape(N, predictor_step, n_label_vars)
        last_pred_from_prev_chunk = y_pred_reshaped[:, -1, :]  # [N, n_label_vars] = last step in chunk

        # (Optional) debug prints for probe nodes
        if debug and probe_nodes:
            y_pred_view = y_pred_chunk.reshape(N, predictor_step, n_label_vars)
            for nidx in probe_nodes:
                if nidx < N:
                    vals = ", ".join([f"{label_vars[k]}={y_pred_view[nidx, 0, k].item():.5f}"
                                     for k in range(min(n_label_vars, y_pred_view.shape[2]))])
                    print(f"[rollout] step {i} node {nidx} pred -> [{vals}]")

                    # Print the ACTUAL slices fed to the model (from x_in, not x_curr)
                    x_in_dyn = x_in[:, n_static_features:].reshape(N, h, n_state_vars + n_driver_vars)
                    # Always uses last h slices (standard autoregressive behavior)
                    slice_label = "last"
                    print(f"  [rollout] step {i} node {nidx} {slice_label} h slices (ACTUAL input to model):")
                    for t_idx in range(h):
                        state_vals = ", ".join([f"{label_vars[k]}={x_in_dyn[nidx, t_idx, k].item():.5f}"
                                               for k in range(min(n_state_vars, len(label_vars)))])
                        driver_vals = ", ".join([f"{driver_vars[k]}={x_in_dyn[nidx, t_idx, n_state_vars + k].item():.5f}"
                                                for k in range(min(n_driver_vars, len(driver_vars)))])
                        print(f"    slice_{t_idx}: states=[{state_vals}], drivers=[{driver_vals}]")

        # 3) Update window with predicted state + true drivers for the next step
        if i < num_chunks - 1:
            # ===== DEBUG: Check window state BEFORE pushforward =====
            if debug and probe_nodes:
                x_dyn_before = x_curr[:, n_static_features:].reshape(N, past_steps, n_state_vars + n_driver_vars)
                print(f"\n[PUSHFORWARD DEBUG] Chunk {i}: Window state BEFORE pushforward")
                for nidx in probe_nodes:
                    if nidx < N:
                        last_slice_before = x_dyn_before[nidx, -1, :n_state_vars]  # Last slice state
                        y_pred_reshaped = y_pred_chunk.reshape(N, predictor_step, n_label_vars)
                        first_pred = y_pred_reshaped[nidx, 0, :n_state_vars]  # First pred in chunk
                        last_pred = y_pred_reshaped[nidx, -1, :n_state_vars]  # Last pred in chunk
                        print(f"  Node {nidx}:")
                        print(f"    Last slice in window (before): wd={last_slice_before[0].item():.6f}")
                        print(f"    First pred in chunk (to insert): wd={first_pred[0].item():.6f}")
                        print(f"    Last pred in chunk (will be new last): wd={last_pred[0].item():.6f}")

            x_curr = use_prediction_pushforward(
                x_curr,
                y_pred_chunk.detach(),          # predicted states
                drivers_chunks[i],              # true drivers for this forecast step
                n_static_features, past_steps,
                n_state_vars, n_driver_vars, predictor_step
            )

            # ===== DEBUG: Check window state AFTER pushforward =====
            if debug:
                static_sum_before = initial_x_full[:, :n_static_features].sum().item()
                static_sum_after = x_curr[:, :n_static_features].sum().item()
                if abs(static_sum_before - static_sum_after) > 1e-6:
                    print(f"[WARNING] Static features changed after step {i}: diff={abs(static_sum_before - static_sum_after):.2e}")

                if probe_nodes:
                    x_dyn_after = x_curr[:, n_static_features:].reshape(N, past_steps, n_state_vars + n_driver_vars)
                    y_pred_reshaped = y_pred_chunk.reshape(N, predictor_step, n_label_vars)

                    print(f"\n[PUSHFORWARD DEBUG] Chunk {i}: Window state AFTER pushforward")
                    print(f"  Verifying that last prediction from chunk {i} is correctly inserted into window")
                    print(f"  This last prediction should be used as residual for chunk {i+1}")

                    for nidx in probe_nodes:
                        if nidx < N:
                            # Last slice in window after pushforward
                            last_slice_after = x_dyn_after[nidx, -1, :n_state_vars]  # [n_state_vars]

                            # Last prediction from chunk i (should match last_slice_after)
                            last_pred_in_chunk = y_pred_reshaped[nidx, -1, :n_state_vars]  # [n_state_vars]

                            # Check match
                            diff_wd = abs(last_slice_after[0].item() - last_pred_in_chunk[0].item())
                            match_status = "MATCH" if diff_wd < 1e-5 else f"MISMATCH (diff={diff_wd:.6f})"

                            print(f"  Node {nidx}:")
                            print(f"    Last slice in window (after): wd={last_slice_after[0].item():.6f}")
                            print(f"    Last pred from chunk {i}: wd={last_pred_in_chunk[0].item():.6f} {match_status}")

                            # Collect pushforward data for JSON
                            window_slices = []
                            for slice_idx in range(max(0, past_steps-2), past_steps):
                                slice_state = x_dyn_after[nidx, slice_idx, :n_state_vars]
                                window_slices.append({
                                    "slice_index": int(slice_idx),
                                    "state": {label_vars[0]: float(slice_state[0].item())}
                                })

                            pushforward_data = {
                                "last_slice_after": {label_vars[0]: float(last_slice_after[0].item())},
                                "last_pred_in_chunk": {label_vars[0]: float(last_pred_in_chunk[0].item())},
                                "match_check": {
                                    "matches": bool(diff_wd < 1e-5),
                                    "difference": float(diff_wd)
                                },
                                "window_slices": window_slices
                            }

                            if chunk_data:
                                chunk_data["pushforward"][f"node_{nidx}"] = pushforward_data

                            # Also check what will be extracted as residual for next chunk
                            # The model will use prepare_model_input_for_rollout which takes last h slices
                            # So the residual for chunk i+1 will come from the last slice of x_curr
                            # which should be the last prediction from chunk i
                            print(f"    -> This value will be used as residual for chunk {i+1}")

                            # Show full window structure for context
                            print(f"    Window structure (showing last 2 slices):")
                            for slice_idx in range(max(0, past_steps-2), past_steps):
                                slice_state = x_dyn_after[nidx, slice_idx, :n_state_vars]
                                print(f"      slice_{slice_idx}: wd={slice_state[0].item():.6f}")

        # Add chunk data to debug_data if collecting
        if chunk_data and debug_data:
            debug_data["chunks"].append(chunk_data)

    # Concatenate all predictions: [N, future_steps_rollout * n_label_vars] -> [future_steps_rollout, N, n_label_vars]
    preds = torch.cat(pred_chunks, dim=1).reshape(N, future_steps_rollout, n_label_vars).permute(1, 0, 2)

    # ===== Save debug data to JSON file =====
    if debug_data and debug_save_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(debug_save_path) if os.path.dirname(debug_save_path) else ".", exist_ok=True)

            # Convert numpy arrays and tensors to native Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    return str(obj)

            debug_data_serializable = convert_to_serializable(debug_data)

            # Save to JSON file
            with open(debug_save_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data_serializable, f, indent=2, ensure_ascii=False)

            print(f"\n{'='*80}")
            print(f"[DEBUG] Residual cascading debug data saved to: {debug_save_path}")
            print(f"{'='*80}")
        except Exception as e:
            print(f"\n[WARNING] Failed to save debug data to JSON: {e}")

    return preds


# =============================================================================
# 6. Debug Function: Verify Multi-Step Output Ordering
# =============================================================================

def debug_output_ordering(y_pred, y_true, var_names, predictor_step, node_idx=0, K=None):
    """
    Debug function to verify the ordering of multi-step labels and decoder outputs.

    Prints:
    - y_pred[0, :K] and y_true[0, :K] with labeled indices
    - A mapping table of (index -> (var_name, step))

    Args:
        y_pred: [N, n_label_vars * predictor_step] model predictions
        y_true: [N, n_label_vars * predictor_step] ground truth labels
        var_names: List of variable names (e.g., ["wd", "vx", "vy"])
        predictor_step: Number of prediction steps (p)
        node_idx: Node index to inspect (default: 0)
        K: Number of indices to print (default: min(12, total))

    Returns:
        dict: Mapping of index -> (var_name, step) for step-major ordering
        dict: Mapping of index -> (var_name, step) for var-major ordering
    """
    n_label_vars = len(var_names)
    total_size = n_label_vars * predictor_step

    if K is None:
        K = min(12, total_size)

    # Extract first K values for the specified node
    y_pred_k = y_pred[node_idx, :K].detach().cpu().numpy()
    y_true_k = y_true[node_idx, :K].detach().cpu().numpy()

    print("\n" + "="*80)
    print("DEBUG: Multi-Step Output Ordering Verification")
    print("="*80)
    print(f"Node index: {node_idx}")
    print(f"Total output size: {total_size} = {n_label_vars} vars x {predictor_step} steps")
    print(f"Variable names: {var_names}")
    print(f"Predictor steps: {predictor_step}")
    print("\n" + "-"*80)
    print("First K values (index 0 to K-1):")
    print("-"*80)
    print(f"{'Index':<8} {'y_pred':<15} {'y_true':<15} {'Step-Major':<25} {'Var-Major':<25}")
    print("-"*80)

    # Build mapping tables
    step_major_map = {}
    var_major_map = {}

    for idx in range(K):
        # Step-major: [var0_t1, var1_t1, ..., var0_t2, var1_t2, ...]
        step = idx // n_label_vars
        var_idx = idx % n_label_vars
        var_name = var_names[var_idx]
        step_major_map[idx] = (var_name, step + 1)

        # Var-major: [var0_t1, var0_t2, ..., var1_t1, var1_t2, ...]
        var_idx_vm = idx // predictor_step
        step_vm = idx % predictor_step
        var_name_vm = var_names[var_idx_vm] if var_idx_vm < n_label_vars else "?"
        var_major_map[idx] = (var_name_vm, step_vm + 1)

        step_major_str = f"{var_name}_t{step+1}"
        var_major_str = f"{var_name_vm}_t{step_vm+1}" if var_idx_vm < n_label_vars else "?"

        print(f"{idx:<8} {y_pred_k[idx]:<15.6f} {y_true_k[idx]:<15.6f} {step_major_str:<25} {var_major_str:<25}")

    print("-"*80)
    print("\nMapping Table (Step-Major Ordering):")
    print("  Index -> (var_name, step)")
    for idx in range(min(K, total_size)):
        var_name, step = step_major_map[idx]
        print(f"    {idx:3d} -> ({var_name:>3s}, t{step})")

    print("\nMapping Table (Var-Major Ordering):")
    print("  Index -> (var_name, step)")
    for idx in range(min(K, total_size)):
        var_name, step = var_major_map[idx]
        if var_name != "?":
            print(f"    {idx:3d} -> ({var_name:>3s}, t{step})")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("Based on the code analysis:")
    print("  1. Data construction (gnn4cf_graph_builder.py snapshot construction):")
    print("     - dynamic_labels stacked as [T, N, n_label_vars] (var-major in last dim)")
    print("     - After permute(1,0,2).reshape: [N, future_steps * n_label_vars]")
    print("     - This creates STEP-MAJOR: [wd_t1, vx_t1, vy_t1, wd_t2, vx_t2, vy_t2, ...]")
    print("  2. Loss function (gnn4cf_training_utils.py loss function):")
    print("     - y_truth_unrolled = data.y.reshape(N, future_steps, n_label_vars)")
    print("     - This assumes STEP-MAJOR ordering")
    print("  3. Model decoder output:")
    print("     - Output shape: [N, n_label_vars * predictor_step]")
    print("     - Comment says: [wd_t1, vx_t1, vy_t1, wd_t2, vx_t2, vy_t2] (STEP-MAJOR)")
    print("  4. Residual tiling (line 855):")
    print("     - node_residual.unsqueeze(1).repeat(1, predictor_step, 1).reshape(...) creates STEP-MAJOR")
    print("     - This is CORRECT and matches the step-major output ordering")
    print("="*80)

    return step_major_map, var_major_map

# =============================================================================
# Branch objectives, monitoring metrics, and training loops
# =============================================================================

def _zero_tensor_like(reference_tensor: torch.Tensor) -> torch.Tensor:
    """Return a scalar zero tensor on the same device/dtype as `reference_tensor`."""
    return torch.zeros((), device=reference_tensor.device, dtype=reference_tensor.dtype)


def _metric_item(value) -> float:
    """Safely convert a Python scalar or 0-d tensor to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _classification_term_tensor(loss_dict, config):
    """
    Return the weighted classification contribution for one branch.

    Weighted regression is already inside `main_loss`; this helper adds the
    optional flood-aware classification term on top of that regression loss.
    """
    main_loss = loss_dict["main_loss"]
    classification_cfg = (
        config.get("loss", {})
        .get("flood_aware", {})
        .get("classification", {})
    )

    if not classification_cfg.get("enabled", False):
        return _zero_tensor_like(main_loss)

    classification_loss = loss_dict.get("classification_loss")
    if classification_loss is None:
        return _zero_tensor_like(main_loss)

    lambda_classification = float(
        classification_cfg.get("lambda_classification", 0.5)
    )
    return lambda_classification * classification_loss


def _shallow_depth_term_tensor(loss_dict, config):
    """
    Return the weighted shallow-depth contribution for one branch.

    This auxiliary term is intentionally added beside the existing main
    regression and classification objectives, so the original loss remains
    available as the baseline when `loss.shallow_depth_loss.enabled` is false.
    """
    main_loss = loss_dict["main_loss"]
    if not _shallow_depth_enabled(config):
        return _zero_tensor_like(main_loss)

    shallow_depth_loss = loss_dict.get("shallow_depth_loss")
    if shallow_depth_loss is None:
        return _zero_tensor_like(main_loss)

    return _shallow_depth_lambda(config) * shallow_depth_loss


def _build_branch_objective(loss_dict, config):
    """
    Build the branch objective used for real/stability weighting.

    Branch objective = regression main loss
                       + optional shallow-depth term
                       + optional weighted classification term.
    The manuscript configuration disables classification, leaving only the
    water-depth prediction and shallow-depth terms in each branch.
    """
    main_loss_tensor = loss_dict["main_loss"]
    shallow_depth_term_tensor = _shallow_depth_term_tensor(loss_dict, config)
    classification_term_tensor = _classification_term_tensor(loss_dict, config)
    branch_total_tensor = (
        main_loss_tensor
        + shallow_depth_term_tensor
        + classification_term_tensor
    )

    return {
        "main_loss_tensor": main_loss_tensor,
        "shallow_depth_term_tensor": shallow_depth_term_tensor,
        "classification_term_tensor": classification_term_tensor,
        "branch_total_tensor": branch_total_tensor,
    }


def _make_empty_branch_loss_dict(reference_tensor, label_vars):
    """Construct a zero-valued branch loss dictionary for the skipped branch path."""
    zero_tensor = _zero_tensor_like(reference_tensor)
    loss_dict = {
        "total_loss": zero_tensor,
        "main_loss": zero_tensor,
    }
    for var in label_vars:
        loss_dict[f"loss_{var}"] = zero_tensor
        loss_dict[f"rmse_{var}"] = zero_tensor
    return loss_dict


def _compute_flood_metrics_for_branch(
    preds_real,
    preds_stability,
    y_truth_unrolled,
    label_vars,
    threshold_normalized,
    comp_mask,
):
    """
    Compute flood metrics from the rollout branch when available.

    The stability/autoregressive branch is the preferred signal because it
    reflects true rollout behavior. If that branch is not present, fall back to
    the real branch.
    """
    if "wd" not in label_vars or threshold_normalized is None:
        return None

    pred_source = preds_stability if preds_stability is not None else preds_real
    wd_idx = label_vars.index("wd")
    pred_wd = pred_source[:, :, wd_idx:wd_idx + 1]
    target_wd = y_truth_unrolled[:, :, wd_idx:wd_idx + 1]
    return compute_flood_metrics(pred_wd, target_wd, threshold_normalized, comp_mask)


def _get_runtime_normalization_stats(config):
    """Fetch runtime normalization stats injected by the trainer, if available."""
    runtime_cfg = config.get("_runtime", {}) if isinstance(config, dict) else {}
    return runtime_cfg.get("normalization_stats", {}) or {}


def _resolve_threshold_normalized_for_metrics(config):
    """Resolve the flood threshold for monitoring, even when flood-aware loss is disabled."""
    if not isinstance(config, dict):
        return None

    flood_aware_cfg = config.get("loss", {}).get("flood_aware", {})
    if not flood_aware_cfg:
        return None

    threshold_normalized = flood_aware_cfg.get("threshold_normalized", None)
    if threshold_normalized is not None:
        return float(threshold_normalized)

    threshold_real = flood_aware_cfg.get("threshold_real", None)
    if threshold_real is None:
        return None

    threshold_real = float(threshold_real)
    normalization_stats = _get_runtime_normalization_stats(config)
    if "wd" in normalization_stats:
        return float(
            convert_real_threshold_to_normalized(
                threshold_real, normalization_stats, var_name="wd"
            )
        )
    return threshold_real


def _compute_wd_performance_metrics(
    pred_source,
    y_truth_unrolled,
    label_vars,
    comp_mask,
    config,
    prefix,
    threshold_normalized=None,
):
    """
    Compute WD RMSE/MAE in both normalized space and meters when stats exist.

    Metrics are restricted to computational nodes so they match the loss mask.
    """
    if pred_source is None or "wd" not in label_vars:
        return {}

    wd_idx = label_vars.index("wd")
    pred_wd = pred_source[:, comp_mask, wd_idx:wd_idx + 1]
    target_wd = y_truth_unrolled[:, comp_mask, wd_idx:wd_idx + 1]
    if pred_wd.numel() == 0 or target_wd.numel() == 0:
        return {}

    diff_norm = pred_wd - target_wd
    metrics = {
        f"{prefix}_rmse_wd_normalized": torch.sqrt(torch.mean(diff_norm ** 2)).item(),
        f"{prefix}_mae_wd_normalized": torch.mean(torch.abs(diff_norm)).item(),
    }

    normalization_stats = _get_runtime_normalization_stats(config)
    wet_mask = None
    if threshold_normalized is not None:
        wet_mask = target_wd >= threshold_normalized

    if "wd" in normalization_stats:
        pred_wd_m = denormalize_tensor(pred_wd, normalization_stats, var_name="wd")
        target_wd_m = denormalize_tensor(target_wd, normalization_stats, var_name="wd")
        diff_m = pred_wd_m - target_wd_m
        metrics.update(
            {
                f"{prefix}_rmse_wd_m": torch.sqrt(torch.mean(diff_m ** 2)).item(),
                f"{prefix}_mae_wd_m": torch.mean(torch.abs(diff_m)).item(),
            }
        )

    if wet_mask is not None and wet_mask.any():
        wet_diff_norm = diff_norm[wet_mask]
        metrics.update(
            {
                f"{prefix}_rmse_wd_wet_normalized": torch.sqrt(torch.mean(wet_diff_norm ** 2)).item(),
                f"{prefix}_mae_wd_wet_normalized": torch.mean(torch.abs(wet_diff_norm)).item(),
            }
        )
        if "wd" in normalization_stats:
            wet_diff_m = diff_m[wet_mask]
            metrics.update(
                {
                    f"{prefix}_rmse_wd_wet_m": torch.sqrt(torch.mean(wet_diff_m ** 2)).item(),
                    f"{prefix}_mae_wd_wet_m": torch.mean(torch.abs(wet_diff_m)).item(),
                }
            )

    return metrics


def train_loop(model, dataloader, optimizer, scheduler, device, config, feature_counts, scheduler_info=None):
    """
    Combined training loop ("Real" + "Stability") generalized for `predictor_step`.

    The optimization target combines the configured loss terms:
    - branch objective = regression main loss + optional shallow-depth/classification
    - batch objective = w_real * real_branch + w_stab * stability_branch

    Each branch uses the same configured regression and auxiliary objectives.
    """
    model.train()

    cfg_window = config["window"]
    predictor_step = cfg_window["predictor_step"]
    past_steps = cfg_window["past_steps"]
    future_steps = cfg_window["future_steps"]

    n_label_vars = feature_counts["n_label_vars"]
    n_state_vars = feature_counts["n_state_vars"]
    n_driver_vars = feature_counts["n_driver_vars"]
    n_static_features = feature_counts["n_static_node"]

    model_history_steps = max(2, predictor_step)
    expected_past_steps = model_history_steps + predictor_step
    if past_steps != expected_past_steps:
        raise ValueError(
            f"Config Error: past_steps ({past_steps}) is incorrect.\n"
            f"With predictor_step = {predictor_step}, model_history_steps = {model_history_steps}.\n"
            f"past_steps MUST be model_history_steps + predictor_step = {expected_past_steps}."
        )

    label_vars = cfg_window.get("label_vars", ["wd", "vx", "vy"])
    loss_config = config.get("loss", {})
    loss_weights = loss_config.get("loss_weights", {})
    w_real = float(loss_weights.get("real", 1.0))
    w_stab = float(loss_weights.get("stability", 1.0))
    skip_stability = w_stab <= 0.0
    shallow_depth_enabled = _shallow_depth_enabled(config)

    batch_metrics = {
        "real_total": 0.0,
        "stability_total": 0.0,
        "real_main_loss": 0.0,
        "stability_main_loss": 0.0,
        "real_classification_term": 0.0,
        "stability_classification_term": 0.0,
        "total_loss": 0.0,
        "real_classification_loss": 0.0,
        "real_classification_accuracy": 0.0,
        "real_classification_precision": 0.0,
        "real_classification_recall": 0.0,
        "real_classification_f1": 0.0,
        "stability_classification_loss": 0.0,
        "stability_classification_accuracy": 0.0,
        "stability_classification_precision": 0.0,
        "stability_classification_recall": 0.0,
        "stability_classification_f1": 0.0,
    }
    if shallow_depth_enabled:
        for branch_name in ["real", "stability"]:
            batch_metrics[f"{branch_name}_shallow_depth_loss"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_term"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_active_fraction"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_active_count"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_rmse_m"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_mae_m"] = 0.0
    for var in label_vars:
        batch_metrics[f"real_loss_{var}"] = 0.0
        batch_metrics[f"stability_loss_{var}"] = 0.0
        batch_metrics[f"total_loss_{var}"] = 0.0
    if "wd" in label_vars:
        batch_metrics["real_rmse_wd"] = 0.0
        batch_metrics["stability_rmse_wd"] = 0.0
        batch_metrics["real_rmse_wd_normalized"] = 0.0
        batch_metrics["real_mae_wd_normalized"] = 0.0
        batch_metrics["stability_rmse_wd_normalized"] = 0.0
        batch_metrics["stability_mae_wd_normalized"] = 0.0
        batch_metrics["real_rmse_wd_m"] = 0.0
        batch_metrics["real_mae_wd_m"] = 0.0
        batch_metrics["real_rmse_wd_wet_normalized"] = 0.0
        batch_metrics["real_mae_wd_wet_normalized"] = 0.0
        batch_metrics["real_rmse_wd_wet_m"] = 0.0
        batch_metrics["real_mae_wd_wet_m"] = 0.0
        batch_metrics["stability_rmse_wd_m"] = 0.0
        batch_metrics["stability_mae_wd_m"] = 0.0
        batch_metrics["stability_rmse_wd_wet_normalized"] = 0.0
        batch_metrics["stability_mae_wd_wet_normalized"] = 0.0
        batch_metrics["stability_rmse_wd_wet_m"] = 0.0
        batch_metrics["stability_mae_wd_wet_m"] = 0.0

    batch_metrics["flooded_nodes_pct"] = 0.0
    batch_metrics["flood_precision"] = 0.0
    batch_metrics["flood_recall"] = 0.0
    batch_metrics["flood_f1"] = 0.0
    batch_metrics["mean_flood_depth_normalized"] = 0.0
    batch_metrics["max_flood_depth_normalized"] = 0.0
    batch_metrics["mean_depth_all_normalized"] = 0.0

    threshold_normalized = _resolve_threshold_normalized_for_metrics(config)

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for data in pbar:
        data = data.to(device)
        N = data.num_nodes
        optimizer.zero_grad()

        if torch.isnan(data.x).any():
            print("WARNING: NaN values in training node inputs.")
            continue
        if torch.isnan(data.edge_attr).any():
            print("WARNING: NaN values in training edge inputs.")
            continue

        y_truth_unrolled = data.y.reshape(N, future_steps, n_label_vars).permute(1, 0, 2)
        y_truth_chunks = data.y.reshape(N, -1, n_label_vars * predictor_step).permute(1, 0, 2)
        drivers_truth_chunks = data.future_drivers.reshape(N, -1, n_driver_vars * predictor_step).permute(1, 0, 2)

        x_full = data.x.clone()
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_type = data.node_type
        edge_type = data.edge_type

        comp_mask = data.node_type == 0

        preds_real_list = []
        x_real = x_full.clone()

        num_chunks = future_steps // predictor_step
        for i in range(num_chunks):
            x_model_in = prepare_model_input(
                x_real, n_static_features, past_steps, model_history_steps
            )

            y_pred_chunk = model(x_model_in, edge_index, edge_attr, node_type, edge_type)

            if torch.isnan(y_pred_chunk).any():
                print("WARNING: NaN values in model predictions.")
                break

            if not hasattr(train_loop, "_debug_called") and i == 0:
                debug_output_ordering(
                    y_pred_chunk,
                    y_truth_chunks[0],
                    label_vars,
                    predictor_step,
                    node_idx=0,
                    K=min(12, predictor_step * len(label_vars)),
                )
                train_loop._debug_called = True

            preds_real_list.append(y_pred_chunk)

            if i < num_chunks - 1:
                x_real = use_prediction_pushforward(
                    x_real,
                    y_truth_chunks[i],
                    drivers_truth_chunks[i],
                    n_static_features,
                    past_steps,
                    n_state_vars,
                    n_driver_vars,
                    predictor_step,
                )

        preds_real = torch.cat(preds_real_list, dim=1).reshape(N, future_steps, n_label_vars).permute(1, 0, 2)
        loss_dict_real = compound_gnn_loss(
            preds_real,
            y_truth_unrolled,
            comp_mask,
            config,
        )
        real_terms = _build_branch_objective(loss_dict_real, config)
        real_total_tensor = real_terms["branch_total_tensor"]
        real_total = _metric_item(real_total_tensor)

        preds_stability = None
        if skip_stability:
            loss_dict_stability = _make_empty_branch_loss_dict(real_total_tensor, label_vars)
            stability_terms = {
                "main_loss_tensor": _zero_tensor_like(real_total_tensor),
                "shallow_depth_term_tensor": _zero_tensor_like(real_total_tensor),
                "classification_term_tensor": _zero_tensor_like(real_total_tensor),
                "branch_total_tensor": _zero_tensor_like(real_total_tensor),
            }
            stability_total_tensor = stability_terms["branch_total_tensor"]
            stability_total = 0.0
        else:
            preds_stability_list = []

            x_model_in_stability = prepare_model_input_stability(
                x_full, n_static_features, past_steps, model_history_steps
            )

            y_pred_chunk_0 = model(
                x_model_in_stability, edge_index, edge_attr, node_type, edge_type
            ).detach()

            x_stability_full = update_graph_with_seed(
                x_full,
                y_pred_chunk_0,
                n_static_features,
                past_steps,
                n_state_vars,
                n_driver_vars,
                predictor_step,
            )

            for i in range(num_chunks):
                x_model_in = prepare_model_input(
                    x_stability_full, n_static_features, past_steps, model_history_steps
                )

                y_pred_chunk = model(x_model_in, edge_index, edge_attr, node_type, edge_type)
                preds_stability_list.append(y_pred_chunk)

                if i < num_chunks - 1:
                    x_stability_full = use_prediction_pushforward(
                        x_stability_full,
                        y_pred_chunk.detach(),
                        drivers_truth_chunks[i],
                        n_static_features,
                        past_steps,
                        n_state_vars,
                        n_driver_vars,
                        predictor_step,
                    )

            preds_stability = torch.cat(preds_stability_list, dim=1).reshape(N, future_steps, n_label_vars).permute(1, 0, 2)
            loss_dict_stability = compound_gnn_loss(
                preds_stability,
                y_truth_unrolled,
                comp_mask,
                config,
            )
            stability_terms = _build_branch_objective(loss_dict_stability, config)
            stability_total_tensor = stability_terms["branch_total_tensor"]
            stability_total = _metric_item(stability_total_tensor)

        total_loss_batch = (
            w_real * real_total_tensor
            + w_stab * stability_total_tensor
        )

        if torch.isnan(total_loss_batch):
            print("WARNING: Batch loss is NaN.")
            print(
                f"  real_total: {real_total}, stability_total: {stability_total}"
            )
            continue

        total_loss_batch.backward()
        # Clip gradients for numerical stability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler_info is None or not scheduler_info.get("needs_validation_metric", False):
            scheduler.step()

        batch_metrics["real_total"] += real_total
        batch_metrics["stability_total"] += stability_total
        batch_metrics["real_main_loss"] += _metric_item(real_terms["main_loss_tensor"])
        batch_metrics["stability_main_loss"] += _metric_item(stability_terms["main_loss_tensor"])
        if shallow_depth_enabled:
            batch_metrics["real_shallow_depth_term"] += _metric_item(
                real_terms["shallow_depth_term_tensor"]
            )
            batch_metrics["stability_shallow_depth_term"] += _metric_item(
                stability_terms["shallow_depth_term_tensor"]
            )
        batch_metrics["real_classification_term"] += _metric_item(real_terms["classification_term_tensor"])
        batch_metrics["stability_classification_term"] += _metric_item(stability_terms["classification_term_tensor"])
        batch_metrics["total_loss"] += _metric_item(total_loss_batch)

        for var in label_vars:
            if f"loss_{var}" in loss_dict_real:
                real_loss_val = _metric_item(loss_dict_real[f"loss_{var}"])
                batch_metrics[f"real_loss_{var}"] += real_loss_val
            if f"loss_{var}" in loss_dict_stability:
                stab_loss_val = _metric_item(loss_dict_stability[f"loss_{var}"])
                batch_metrics[f"stability_loss_{var}"] += stab_loss_val
            if f"loss_{var}" in loss_dict_real and f"loss_{var}" in loss_dict_stability:
                batch_metrics[f"total_loss_{var}"] += (
                    _metric_item(loss_dict_real[f"loss_{var}"])
                    + _metric_item(loss_dict_stability[f"loss_{var}"])
                )

        if "wd" in label_vars:
            if "rmse_wd" in loss_dict_real:
                batch_metrics["real_rmse_wd"] += _metric_item(loss_dict_real["rmse_wd"])
            if "rmse_wd" in loss_dict_stability:
                batch_metrics["stability_rmse_wd"] += _metric_item(loss_dict_stability["rmse_wd"])

        if "classification_loss" in loss_dict_real:
            batch_metrics["real_classification_loss"] += _metric_item(loss_dict_real["classification_loss"])
            batch_metrics["real_classification_accuracy"] += _metric_item(
                loss_dict_real.get("classification_accuracy", 0.0)
            )
            batch_metrics["real_classification_precision"] += _metric_item(
                loss_dict_real.get("classification_precision", 0.0)
            )
            batch_metrics["real_classification_recall"] += _metric_item(
                loss_dict_real.get("classification_recall", 0.0)
            )
            batch_metrics["real_classification_f1"] += _metric_item(
                loss_dict_real.get("classification_f1", 0.0)
            )

        if "classification_loss" in loss_dict_stability:
            batch_metrics["stability_classification_loss"] += _metric_item(loss_dict_stability["classification_loss"])
            batch_metrics["stability_classification_accuracy"] += _metric_item(
                loss_dict_stability.get("classification_accuracy", 0.0)
            )
            batch_metrics["stability_classification_precision"] += _metric_item(
                loss_dict_stability.get("classification_precision", 0.0)
            )
            batch_metrics["stability_classification_recall"] += _metric_item(
                loss_dict_stability.get("classification_recall", 0.0)
            )
            batch_metrics["stability_classification_f1"] += _metric_item(
                loss_dict_stability.get("classification_f1", 0.0)
            )

        if shallow_depth_enabled:
            for branch_name, loss_dict in [
                ("real", loss_dict_real),
                ("stability", loss_dict_stability),
            ]:
                if "shallow_depth_loss" not in loss_dict:
                    continue
                for metric_name in [
                    "shallow_depth_loss",
                    "shallow_depth_active_fraction",
                    "shallow_depth_active_count",
                    "shallow_depth_rmse_m",
                    "shallow_depth_mae_m",
                ]:
                    batch_metrics[f"{branch_name}_{metric_name}"] += _metric_item(
                        loss_dict.get(metric_name, 0.0)
                    )

        if "wd" in label_vars:
            real_perf_metrics = _compute_wd_performance_metrics(
                preds_real,
                y_truth_unrolled,
                label_vars,
                comp_mask,
                config,
                prefix="real",
                threshold_normalized=threshold_normalized,
            )
            for key, value in real_perf_metrics.items():
                batch_metrics[key] += value

            stability_perf_metrics = _compute_wd_performance_metrics(
                preds_stability,
                y_truth_unrolled,
                label_vars,
                comp_mask,
                config,
                prefix="stability",
                threshold_normalized=threshold_normalized,
            )
            for key, value in stability_perf_metrics.items():
                batch_metrics[key] += value

        flood_metrics = _compute_flood_metrics_for_branch(
            preds_real=preds_real,
            preds_stability=preds_stability,
            y_truth_unrolled=y_truth_unrolled,
            label_vars=label_vars,
            threshold_normalized=threshold_normalized,
            comp_mask=comp_mask,
        )
        if flood_metrics is not None:
            batch_metrics["flooded_nodes_pct"] += flood_metrics["flooded_nodes_pct"]
            batch_metrics["flood_precision"] += flood_metrics["flood_precision"]
            batch_metrics["flood_recall"] += flood_metrics["flood_recall"]
            batch_metrics["flood_f1"] += flood_metrics["flood_f1"]
            batch_metrics["mean_flood_depth_normalized"] += flood_metrics["mean_flood_depth_normalized"]
            batch_metrics["max_flood_depth_normalized"] += flood_metrics["max_flood_depth_normalized"]
            batch_metrics["mean_depth_all_normalized"] += flood_metrics["mean_depth_all_normalized"]

    num_batches = max(1, len(dataloader))
    for key in batch_metrics:
        batch_metrics[key] /= num_batches

    return batch_metrics


@torch.no_grad()
def validate_loop(model, dataloader, device, config, feature_counts):
    """
    Validation loop generalized for `predictor_step`.

    Validation uses the branch that best reflects deployment:
    - stability/autoregressive branch only when stability is enabled
    - real/teacher-forcing branch only when stability is disabled

    This reduces validation cost while keeping checkpoint selection focused on
    the rollout behavior that matters at inference time.
    """
    model.eval()

    cfg_window = config["window"]
    predictor_step = cfg_window["predictor_step"]
    past_steps = cfg_window["past_steps"]
    future_steps = cfg_window["future_steps"]

    n_label_vars = feature_counts["n_label_vars"]
    n_state_vars = feature_counts["n_state_vars"]
    n_driver_vars = feature_counts["n_driver_vars"]
    n_static_features = feature_counts["n_static_node"]

    model_history_steps = max(2, predictor_step)
    expected_past_steps = model_history_steps + predictor_step
    if past_steps != expected_past_steps:
        raise ValueError(
            f"Config Error: past_steps ({past_steps}) is incorrect.\n"
            f"With predictor_step = {predictor_step}, model_history_steps = {model_history_steps}.\n"
            f"past_steps MUST be model_history_steps + predictor_step = {expected_past_steps}."
        )

    label_vars = cfg_window.get("label_vars", ["wd", "vx", "vy"])
    loss_config = config.get("loss", {})
    loss_weights = loss_config.get("loss_weights", {})
    w_real = float(loss_weights.get("real", 1.0))
    w_stab = float(loss_weights.get("stability", 1.0))
    skip_stability = w_stab <= 0.0
    shallow_depth_enabled = _shallow_depth_enabled(config)

    batch_metrics = {
        "real_total": 0.0,
        "stability_total": 0.0,
        "real_main_loss": 0.0,
        "stability_main_loss": 0.0,
        "real_classification_term": 0.0,
        "stability_classification_term": 0.0,
        "total_loss": 0.0,
        "real_classification_loss": 0.0,
        "real_classification_accuracy": 0.0,
        "real_classification_precision": 0.0,
        "real_classification_recall": 0.0,
        "real_classification_f1": 0.0,
        "stability_classification_loss": 0.0,
        "stability_classification_accuracy": 0.0,
        "stability_classification_precision": 0.0,
        "stability_classification_recall": 0.0,
        "stability_classification_f1": 0.0,
    }
    if shallow_depth_enabled:
        for branch_name in ["real", "stability"]:
            batch_metrics[f"{branch_name}_shallow_depth_loss"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_term"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_active_fraction"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_active_count"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_rmse_m"] = 0.0
            batch_metrics[f"{branch_name}_shallow_depth_mae_m"] = 0.0
    for var in label_vars:
        batch_metrics[f"real_loss_{var}"] = 0.0
        batch_metrics[f"stability_loss_{var}"] = 0.0
        batch_metrics[f"total_loss_{var}"] = 0.0
    if "wd" in label_vars:
        batch_metrics["real_rmse_wd"] = 0.0
        batch_metrics["stability_rmse_wd"] = 0.0
        batch_metrics["real_rmse_wd_normalized"] = 0.0
        batch_metrics["real_mae_wd_normalized"] = 0.0
        batch_metrics["stability_rmse_wd_normalized"] = 0.0
        batch_metrics["stability_mae_wd_normalized"] = 0.0
        batch_metrics["real_rmse_wd_m"] = 0.0
        batch_metrics["real_mae_wd_m"] = 0.0
        batch_metrics["real_rmse_wd_wet_normalized"] = 0.0
        batch_metrics["real_mae_wd_wet_normalized"] = 0.0
        batch_metrics["real_rmse_wd_wet_m"] = 0.0
        batch_metrics["real_mae_wd_wet_m"] = 0.0
        batch_metrics["stability_rmse_wd_m"] = 0.0
        batch_metrics["stability_mae_wd_m"] = 0.0
        batch_metrics["stability_rmse_wd_wet_normalized"] = 0.0
        batch_metrics["stability_mae_wd_wet_normalized"] = 0.0
        batch_metrics["stability_rmse_wd_wet_m"] = 0.0
        batch_metrics["stability_mae_wd_wet_m"] = 0.0

    batch_metrics["flooded_nodes_pct"] = 0.0
    batch_metrics["flood_precision"] = 0.0
    batch_metrics["flood_recall"] = 0.0
    batch_metrics["flood_f1"] = 0.0
    batch_metrics["mean_flood_depth_normalized"] = 0.0
    batch_metrics["max_flood_depth_normalized"] = 0.0
    batch_metrics["mean_depth_all_normalized"] = 0.0

    threshold_normalized = _resolve_threshold_normalized_for_metrics(config)

    use_stability_validation = not skip_stability

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for data in pbar:
        data = data.to(device)
        N = data.num_nodes

        y_truth_unrolled = data.y.reshape(N, future_steps, n_label_vars).permute(1, 0, 2)
        drivers_truth_chunks = data.future_drivers.reshape(N, -1, n_driver_vars * predictor_step).permute(1, 0, 2)

        x_full = data.x.clone()
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_type = data.node_type
        edge_type = data.edge_type

        comp_mask = data.node_type == 0

        num_chunks = future_steps // predictor_step
        preds_real = None
        preds_stability = None

        if use_stability_validation:
            loss_dict_real = _make_empty_branch_loss_dict(
                torch.tensor(0.0, device=device, dtype=data.y.dtype),
                label_vars,
            )
            real_terms = {
                "main_loss_tensor": _zero_tensor_like(loss_dict_real["main_loss"]),
                "shallow_depth_term_tensor": _zero_tensor_like(loss_dict_real["main_loss"]),
                "classification_term_tensor": _zero_tensor_like(loss_dict_real["main_loss"]),
                "branch_total_tensor": _zero_tensor_like(loss_dict_real["main_loss"]),
            }
            real_total_tensor = real_terms["branch_total_tensor"]
            real_total = 0.0

            preds_stability_list = []
            x_model_in_stability = prepare_model_input_stability(
                x_full, n_static_features, past_steps, model_history_steps
            )
            y_pred_chunk_0 = model(
                x_model_in_stability, edge_index, edge_attr, node_type, edge_type
            )
            x_stability_full = update_graph_with_seed(
                x_full,
                y_pred_chunk_0,
                n_static_features,
                past_steps,
                n_state_vars,
                n_driver_vars,
                predictor_step,
            )

            for i in range(num_chunks):
                x_model_in = prepare_model_input(
                    x_stability_full, n_static_features, past_steps, model_history_steps
                )
                y_pred_chunk = model(x_model_in, edge_index, edge_attr, node_type, edge_type)
                preds_stability_list.append(y_pred_chunk)

                if i < num_chunks - 1:
                    x_stability_full = use_prediction_pushforward(
                        x_stability_full,
                        y_pred_chunk,
                        drivers_truth_chunks[i],
                        n_static_features,
                        past_steps,
                        n_state_vars,
                        n_driver_vars,
                        predictor_step,
                    )

            preds_stability = torch.cat(preds_stability_list, dim=1).reshape(N, future_steps, n_label_vars).permute(1, 0, 2)
            loss_dict_stability = compound_gnn_loss(
                preds_stability,
                y_truth_unrolled,
                comp_mask,
                config,
            )
            stability_terms = _build_branch_objective(loss_dict_stability, config)
            stability_total_tensor = stability_terms["branch_total_tensor"]
            stability_total = _metric_item(stability_total_tensor)
            total_loss_val = _metric_item(
                w_stab * stability_total_tensor
            )

            batch_metrics["stability_total"] += stability_total
            batch_metrics["stability_main_loss"] += _metric_item(stability_terms["main_loss_tensor"])
            if shallow_depth_enabled:
                batch_metrics["stability_shallow_depth_term"] += _metric_item(
                    stability_terms["shallow_depth_term_tensor"]
                )
            batch_metrics["stability_classification_term"] += _metric_item(
                stability_terms["classification_term_tensor"]
            )
            batch_metrics["total_loss"] += total_loss_val

            for var in label_vars:
                if f"loss_{var}" in loss_dict_stability:
                    stab_loss_val = _metric_item(loss_dict_stability[f"loss_{var}"])
                    batch_metrics[f"stability_loss_{var}"] += stab_loss_val
                    batch_metrics[f"total_loss_{var}"] += stab_loss_val

            if "wd" in label_vars and "rmse_wd" in loss_dict_stability:
                batch_metrics["stability_rmse_wd"] += _metric_item(loss_dict_stability["rmse_wd"])

            if "classification_loss" in loss_dict_stability:
                batch_metrics["stability_classification_loss"] += _metric_item(
                    loss_dict_stability["classification_loss"]
                )
                batch_metrics["stability_classification_accuracy"] += _metric_item(
                    loss_dict_stability.get("classification_accuracy", 0.0)
                )
                batch_metrics["stability_classification_precision"] += _metric_item(
                    loss_dict_stability.get("classification_precision", 0.0)
                )
                batch_metrics["stability_classification_recall"] += _metric_item(
                    loss_dict_stability.get("classification_recall", 0.0)
                )
                batch_metrics["stability_classification_f1"] += _metric_item(
                    loss_dict_stability.get("classification_f1", 0.0)
                )

            if shallow_depth_enabled and "shallow_depth_loss" in loss_dict_stability:
                for metric_name in [
                    "shallow_depth_loss",
                    "shallow_depth_active_fraction",
                    "shallow_depth_active_count",
                    "shallow_depth_rmse_m",
                    "shallow_depth_mae_m",
                ]:
                    batch_metrics[f"stability_{metric_name}"] += _metric_item(
                        loss_dict_stability.get(metric_name, 0.0)
                    )

            if "wd" in label_vars:
                stability_perf_metrics = _compute_wd_performance_metrics(
                    preds_stability,
                    y_truth_unrolled,
                    label_vars,
                    comp_mask,
                    config,
                    prefix="stability",
                    threshold_normalized=threshold_normalized,
                )
                for key, value in stability_perf_metrics.items():
                    batch_metrics[key] += value
        else:
            preds_real_list = []
            x_real = x_full.clone()
            y_truth_chunks = data.y.reshape(N, -1, n_label_vars * predictor_step).permute(1, 0, 2)

            for i in range(num_chunks):
                x_model_in = prepare_model_input(
                    x_real, n_static_features, past_steps, model_history_steps
                )
                y_pred_chunk = model(x_model_in, edge_index, edge_attr, node_type, edge_type)
                preds_real_list.append(y_pred_chunk)

                if i < num_chunks - 1:
                    x_real = use_prediction_pushforward(
                        x_real,
                        y_truth_chunks[i],
                        drivers_truth_chunks[i],
                        n_static_features,
                        past_steps,
                        n_state_vars,
                        n_driver_vars,
                        predictor_step,
                    )

            preds_real = torch.cat(preds_real_list, dim=1).reshape(N, future_steps, n_label_vars).permute(1, 0, 2)
            loss_dict_real = compound_gnn_loss(
                preds_real,
                y_truth_unrolled,
                comp_mask,
                config,
            )
            real_terms = _build_branch_objective(loss_dict_real, config)
            real_total_tensor = real_terms["branch_total_tensor"]
            real_total = _metric_item(real_total_tensor)

            loss_dict_stability = _make_empty_branch_loss_dict(real_total_tensor, label_vars)
            stability_terms = {
                "main_loss_tensor": _zero_tensor_like(real_total_tensor),
                "shallow_depth_term_tensor": _zero_tensor_like(real_total_tensor),
                "classification_term_tensor": _zero_tensor_like(real_total_tensor),
                "branch_total_tensor": _zero_tensor_like(real_total_tensor),
            }
            stability_total_tensor = stability_terms["branch_total_tensor"]
            stability_total = 0.0

            total_loss_val = _metric_item(
                w_real * real_total_tensor
            )

            batch_metrics["real_total"] += real_total
            batch_metrics["real_main_loss"] += _metric_item(real_terms["main_loss_tensor"])
            if shallow_depth_enabled:
                batch_metrics["real_shallow_depth_term"] += _metric_item(
                    real_terms["shallow_depth_term_tensor"]
                )
            batch_metrics["real_classification_term"] += _metric_item(
                real_terms["classification_term_tensor"]
            )
            batch_metrics["total_loss"] += total_loss_val

            for var in label_vars:
                if f"loss_{var}" in loss_dict_real:
                    real_loss_val = _metric_item(loss_dict_real[f"loss_{var}"])
                    batch_metrics[f"real_loss_{var}"] += real_loss_val
                    batch_metrics[f"total_loss_{var}"] += real_loss_val

            if "wd" in label_vars and "rmse_wd" in loss_dict_real:
                batch_metrics["real_rmse_wd"] += _metric_item(loss_dict_real["rmse_wd"])

            if "classification_loss" in loss_dict_real:
                batch_metrics["real_classification_loss"] += _metric_item(
                    loss_dict_real["classification_loss"]
                )
                batch_metrics["real_classification_accuracy"] += _metric_item(
                    loss_dict_real.get("classification_accuracy", 0.0)
                )
                batch_metrics["real_classification_precision"] += _metric_item(
                    loss_dict_real.get("classification_precision", 0.0)
                )
                batch_metrics["real_classification_recall"] += _metric_item(
                    loss_dict_real.get("classification_recall", 0.0)
                )
                batch_metrics["real_classification_f1"] += _metric_item(
                    loss_dict_real.get("classification_f1", 0.0)
                )

            if shallow_depth_enabled and "shallow_depth_loss" in loss_dict_real:
                for metric_name in [
                    "shallow_depth_loss",
                    "shallow_depth_active_fraction",
                    "shallow_depth_active_count",
                    "shallow_depth_rmse_m",
                    "shallow_depth_mae_m",
                ]:
                    batch_metrics[f"real_{metric_name}"] += _metric_item(
                        loss_dict_real.get(metric_name, 0.0)
                    )

            if "wd" in label_vars:
                real_perf_metrics = _compute_wd_performance_metrics(
                    preds_real,
                    y_truth_unrolled,
                    label_vars,
                    comp_mask,
                    config,
                    prefix="real",
                    threshold_normalized=threshold_normalized,
                )
                for key, value in real_perf_metrics.items():
                    batch_metrics[key] += value

        flood_metrics = _compute_flood_metrics_for_branch(
            preds_real=preds_real,
            preds_stability=preds_stability,
            y_truth_unrolled=y_truth_unrolled,
            label_vars=label_vars,
            threshold_normalized=threshold_normalized,
            comp_mask=comp_mask,
        )
        if flood_metrics is not None:
            batch_metrics["flooded_nodes_pct"] += flood_metrics["flooded_nodes_pct"]
            batch_metrics["flood_precision"] += flood_metrics["flood_precision"]
            batch_metrics["flood_recall"] += flood_metrics["flood_recall"]
            batch_metrics["flood_f1"] += flood_metrics["flood_f1"]
            batch_metrics["mean_flood_depth_normalized"] += flood_metrics["mean_flood_depth_normalized"]
            batch_metrics["max_flood_depth_normalized"] += flood_metrics["max_flood_depth_normalized"]
            batch_metrics["mean_depth_all_normalized"] += flood_metrics["mean_depth_all_normalized"]

    num_batches = max(1, len(dataloader))
    for key in batch_metrics:
        batch_metrics[key] /= num_batches

    return batch_metrics
