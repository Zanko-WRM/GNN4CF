# -*- coding: utf-8 -*-

"""
Build typed graph snapshots from HEC-RAS compound-flood simulations.

Read hydraulic outputs, mesh connectivity, scalar terrain statistics, and GIS
attributes, then assemble temporally ordered input and target windows.
Computational nodes carry interior hydraulic state and rainfall drivers;
boundary ghost nodes carry coastal forcing. The HDF dataset module adds
representative nodes and virtual boundary-interior connections.
"""

import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import h5py
import glob
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# =========================================================
# UTILITIES: Min–max + stats IO
# =========================================================
def min_max_normalize(data, min_val=None, max_val=None, eps=1e-8):
    """
    Performs vectorized min-max normalization on NumPy arrays or pandas objects.

    Args:
        data (np.ndarray | pd.Series | pd.DataFrame): The input data to normalize.
        min_val (float, optional): A pre-calculated minimum value. If None, it's computed from the data.
        max_val (float, optional): A pre-calculated maximum value. If None, it's computed from the data.
        eps (float, optional): A small epsilon to avoid division by zero.

    Returns:
        tuple[np.ndarray | pd.Series | pd.DataFrame, dict]: A tuple containing:
            - The normalized data.
            - A dictionary with the min and max values used for scaling.
    """
    arr = data.to_numpy(dtype=float) if isinstance(data, (pd.Series, pd.DataFrame)) else np.asarray(data, dtype=float)
    if min_val is None: min_val = np.nanmin(arr)
    if max_val is None: max_val = np.nanmax(arr)
    denom = (max_val - min_val)
    norm = np.zeros_like(arr, dtype=float) if not np.isfinite(denom) or denom < eps else (arr - min_val) / (denom + eps)
    if isinstance(data, pd.Series):  norm = pd.Series(norm, index=data.index, name=data.name)
    if isinstance(data, pd.DataFrame): norm = pd.DataFrame(norm, index=data.index, columns=data.columns)
    stats = {
        "method": "minmax",
        "min": float(min_val),
        "max": float(max_val),
        "original_min": float(np.nanmin(arr)),
        "original_max": float(np.nanmax(arr))
    }
    return norm, stats


def zscore_normalize(data, mean=None, std=None, eps=1e-8):
    """
    Performs z-score normalization (standardization) on NumPy arrays or pandas objects.

    Args:
        data (np.ndarray | pd.Series | pd.DataFrame): The input data to normalize.
        mean (float, optional): A pre-calculated mean value. If None, it's computed from the data.
        std (float, optional): A pre-calculated standard deviation. If None, it's computed from the data.
        eps (float, optional): A small epsilon to avoid division by zero.

    Returns:
        tuple[np.ndarray | pd.Series | pd.DataFrame, dict]: A tuple containing:
            - The normalized data.
            - A dictionary with the mean and std values used for scaling.
    """
    arr = data.to_numpy(dtype=float) if isinstance(data, (pd.Series, pd.DataFrame)) else np.asarray(data, dtype=float)
    if mean is None: mean = np.nanmean(arr)
    if std is None: std = np.nanstd(arr)
    norm = np.zeros_like(arr, dtype=float) if not np.isfinite(std) or std < eps else (arr - mean) / (std + eps)
    if isinstance(data, pd.Series): norm = pd.Series(norm, index=data.index, name=data.name)
    if isinstance(data, pd.DataFrame): norm = pd.DataFrame(norm, index=data.index, columns=data.columns)
    stats = {
        "method": "zscore",
        "mean": float(mean),
        "std": float(std),
        "original_min": float(np.nanmin(arr)),
        "original_max": float(np.nanmax(arr)),
        "original_mean": float(np.nanmean(arr)),
        "original_std": float(np.nanstd(arr))
    }
    return norm, stats


def log1p_zscore_normalize(data, log1p_mean=None, log1p_std=None, eps=1e-8):
    """
    Performs log1p transformation followed by z-score normalization.
    Handles zeros gracefully and is ideal for skewed, zero-inflated data.

    Args:
        data (np.ndarray | pd.Series | pd.DataFrame): The input data to normalize (must be >= 0).
        log1p_mean (float, optional): Pre-calculated mean of log(1+x). If None, computed from data.
        log1p_std (float, optional): Pre-calculated std of log(1+x). If None, computed from data.
        eps (float, optional): A small epsilon to avoid division by zero.

    Returns:
        tuple[np.ndarray | pd.Series | pd.DataFrame, dict]: A tuple containing:
            - The normalized data.
            - A dictionary with statistics including method, log1p_mean, log1p_std, and original stats.
    """
    arr = data.to_numpy(dtype=float) if isinstance(data, (pd.Series, pd.DataFrame)) else np.asarray(data, dtype=float)

    # Ensure non-negative (log1p requires x >= 0)
    arr = np.maximum(arr, 0.0)

    # Step 1: Apply log1p transformation
    log1p_arr = np.log1p(arr)  # log(1+x), handles zeros correctly

    # Step 2: Compute or use provided statistics
    if log1p_mean is None: log1p_mean = np.nanmean(log1p_arr)
    if log1p_std is None: log1p_std = np.nanstd(log1p_arr)

    # Step 3: Apply z-score to log1p values
    norm = np.zeros_like(log1p_arr, dtype=float) if not np.isfinite(log1p_std) or log1p_std < eps else (log1p_arr - log1p_mean) / (log1p_std + eps)

    if isinstance(data, pd.Series): norm = pd.Series(norm, index=data.index, name=data.name)
    if isinstance(data, pd.DataFrame): norm = pd.DataFrame(norm, index=data.index, columns=data.columns)

    stats = {
        "method": "log1p_zscore",
        "log1p_mean": float(log1p_mean),
        "log1p_std": float(log1p_std),
        "original_min": float(np.nanmin(arr)),
        "original_max": float(np.nanmax(arr)),
        "original_mean": float(np.nanmean(arr)),
        "original_std": float(np.nanstd(arr))
    }
    return norm, stats


def save_stats_file(stats, file_path):
    """
    Saves the combined statistics dictionary to a JSON file.

    Args:
        stats (dict): The dictionary containing normalization statistics.
        file_path (str): The path to the output JSON file.
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved combined normalization stats to: {file_path}")


def load_stats_file(file_path):
    """
    Loads a JSON statistics file if it exists, otherwise returns an empty dict.

    Args:
        file_path (str): Path to the JSON statistics file.

    Returns:
        dict: The loaded statistics dictionary.
    """
    if not os.path.exists(file_path):
        print(f"Statistics file not found at {file_path}. A new one will be created.")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def denormalize(data, stats):
    """
    Universal denormalize function that dispatches based on normalization method in stats.

    Args:
        data (np.ndarray | pd.Series | pd.DataFrame): Normalized data.
        stats (dict): Statistics dictionary containing 'method' and method-specific parameters.
                     For backward compatibility, can also accept (min_val, max_val) tuple.

    Returns:
        np.ndarray | pd.Series | pd.DataFrame: The denormalized data in original scale.
    """
    # Backward compatibility: if stats is a tuple, treat as (min_val, max_val)
    if isinstance(stats, tuple):
        min_val, max_val = stats
        arr = data.to_numpy(dtype=float) if isinstance(data, (pd.Series, pd.DataFrame)) else np.asarray(data, dtype=float)
        denorm = arr * (max_val - min_val) + min_val
        if isinstance(data, pd.Series): denorm = pd.Series(denorm, index=data.index, name=data.name)
        if isinstance(data, pd.DataFrame): denorm = pd.DataFrame(denorm, index=data.index, columns=data.columns)
        return denorm

    method = stats.get("method", "minmax")  # Default to minmax for backward compatibility
    arr = data.to_numpy(dtype=float) if isinstance(data, (pd.Series, pd.DataFrame)) else np.asarray(data, dtype=float)

    if method == "minmax":
        min_val = stats["min"]
        max_val = stats["max"]
        denorm = arr * (max_val - min_val) + min_val
    elif method == "zscore":
        mean = stats["mean"]
        std = stats["std"]
        denorm = arr * std + mean
    elif method == "log1p_zscore":
        log1p_mean = stats["log1p_mean"]
        log1p_std = stats["log1p_std"]
        # Step 1: Reverse z-score
        log1p_data = arr * log1p_std + log1p_mean
        # Step 2: Reverse log1p: exp(x) - 1
        denorm = np.expm1(log1p_data)
    elif method == "no_norm":
        denorm = arr  # No denormalization needed
    else:
        raise ValueError(f"Unknown normalization method: {method}. Stats must contain 'method' field.")

    if isinstance(data, pd.Series): denorm = pd.Series(denorm, index=data.index, name=data.name)
    if isinstance(data, pd.DataFrame): denorm = pd.DataFrame(denorm, index=data.index, columns=data.columns)
    return denorm


def denormalize_legacy(data, min_val, max_val):
    """
    Restore min-max-normalized values using explicit scale bounds.

    Args:
        data (np.ndarray | pd.Series | pd.DataFrame): Normalized data (0 to 1).
        min_val (float): The original minimum value used for normalization.
        max_val (float): The original maximum value used for normalization.

    Returns:
        np.ndarray | pd.Series | pd.DataFrame: The denormalized data.
    """
    return denormalize(data, {"method": "minmax", "min": min_val, "max": max_val})


# =========================================================
# Config & HDF Loaders
# =========================================================
def load_config(path=None):
    """Load the shared repository configuration or an explicitly supplied path."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "config_gnn4cf_final.yml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hdf_roots(cfg):
    """Constructs HDF group paths from the config."""
    area = cfg["hecras"]["area_name"]
    g_root = cfg["hecras"].get("geometry_root", "Geometry/2D Flow Areas")
    r_root = cfg["hecras"].get(
        "results_root",
        "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas",
    )
    return g_root, r_root, f"{g_root}/{area}", f"{r_root}/{area}"


def load_hdf_required(hdf_path, cfg):
    """
    Loads all required static and dynamic arrays from a HEC-RAS HDF file.

    Args:
        hdf_path (str): Path to the HEC-RAS HDF output file.
        cfg (dict): The global configuration dictionary.

    Returns:
        dict: A dictionary of NumPy arrays and DataFrames from the HDF file.
    """
    g_root, r_root, g_area, r_area = _hdf_roots(cfg)
    out = {"_source_hdf": hdf_path, "_area": cfg["hecras"]["area_name"]}
    with h5py.File(hdf_path, "r") as f:
        def get(p):
            return f[p][:] if p in f else None

        # --- Static Geometry ---
        out["cells_center_xy"] = get(f"{g_area}/Cells Center Coordinate")
        out["cell_points"] = get(f"{g_root}/Cell Points")
        out["faces_cell_idx"] = get(f"{g_area}/Faces Cell Indexes")
        out["elevations"] = get(f"{g_area}/Cells Minimum Elevation")
        out["area_cell"] = get(f"{g_area}/Cells Surface Area")
        out["manning_n"] = get(f"{g_area}/Cells Center Manning's n")
        out["infiltration"] = get(f"{g_area}/Infiltration/Abstraction Ratio")
        if out["infiltration"] is None and out.get("elevations") is not None:
            out["infiltration"] = np.zeros_like(out["elevations"], dtype=float)

        # --- Dynamic Results ---
        out["wd"] = get(f"{r_area}/Cell Invert Depth")
        out["vx"] = get(f"{r_area}/Cell Velocity - Velocity X")
        out["vy"] = get(f"{r_area}/Cell Velocity - Velocity Y")
        out["cell_v"] = get(f"{r_area}/Cell Volume")

        # --- Dynamic Drivers ---
        pr = get(f"{r_area}/Cell Precipitation Rate")
        T_fallback = out["wd"].shape[0] if isinstance(out.get("wd"), np.ndarray) and out["wd"] is not None else 0
        if pr is None:
            pr = np.zeros((T_fallback,), dtype=float)
        elif pr.ndim == 2 and pr.shape[1] >= 1:
            pr = pr[:, 0]
        else:
            pr = pr.reshape(pr.shape[0], )
        out["pr"] = pr  # Precipitation Rate

        # Accumulated Precipitation Depth
        cul_pr_path = f"{r_area}/Cell Cumulative Precipitation Depth"
        cul_pr = get(cul_pr_path)
        if cul_pr is None and out.get("wd") is not None:
            # If not found, create a zero array with the same shape as other cell-based results
            cul_pr = np.zeros_like(out["wd"], dtype=float)
        out["cul_pr"] = cul_pr   # Acc_Precipitation


        # Read stage data - support both single and multiple boundary formats
        bc_path = f"{r_area}/Boundary Conditions"
        bc_stage_data = {}  # {bc_id: stage_array} - for multiple boundaries

        # First, try to read from BC-specific datasets (BC1 - Stage per Face, BC2 - Stage per Face, etc.)
        # The HDF structure has datasets at the same level as BC directories:
        # Boundary Conditions/BC1 - Stage per Face (dataset)
        # Boundary Conditions/BC1 (directory/group)
        if bc_path in f:
            available_items = list(f[bc_path].keys())

            # Look ONLY for BC-specific stage datasets matching pattern "BC* - Stage per Face"
            # Ignore directories (BC1, BC2, BC3) and other datasets (BC* - Flow per Face, etc.)
            stage_datasets = [item for item in available_items
                            if item.endswith(" - Stage per Face") and item.startswith("BC")]

            for stage_dataset_name in stage_datasets:
                try:
                    # Extract BC ID from dataset name (e.g., "BC1 - Stage per Face" -> 1)
                    bc_name = stage_dataset_name.split(" - ")[0]  # "BC1"
                    bc_num_str = bc_name[2:]  # "1"
                    if bc_num_str.isdigit():
                        bc_id = int(bc_num_str)
                        # Skip BC0 (doesn't exist in our model)
                        if bc_id == 0:
                            continue
                        stage_path = f"{bc_path}/{stage_dataset_name}"
                        stage_data = get(stage_path)
                        if stage_data is not None:
                            bc_stage_data[bc_id] = stage_data
                            print(f"  ✓ Loaded stage data for {bc_name}: shape {stage_data.shape}")
                except (ValueError, KeyError, AttributeError):
                    continue

        # Determine which format to use
        if len(bc_stage_data) > 1:
            # Multiple boundaries: use BC-specific data
            out["bc_stage_data"] = bc_stage_data
            # Create placeholder (will be filled from bc_stage_data in gather_and_compute_node_timeseries)
            n_faces = out["faces_cell_idx"].shape[0] if out.get("faces_cell_idx") is not None else 0
            out["stage_faces"] = np.zeros((T_fallback, n_faces), dtype=float)  # Placeholder
        else:
            # Single boundary: use old format path
            stage = get(f"{r_area}/Boundary Conditions/BC Line - Stage per Face")
            if stage is None:
                n_faces = out["faces_cell_idx"].shape[0] if out.get("faces_cell_idx") is not None else 0
                stage = np.zeros((T_fallback, n_faces), dtype=float)
            out["stage_faces"] = stage  # Single boundary format
            out["bc_stage_data"] = {}  # Empty for single boundary

        ds_path = "Geometry/Boundary Condition Lines/External Faces"
        if ds_path in f:
            arr = f[ds_path][:]
            df = pd.DataFrame(arr)
            rename = {"BC Line ID": "bc_line_id", "Face Index": "face_index"}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            for c in ["bc_line_id", "face_index"]:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            out["external_faces"] = df
        else:
            out["external_faces"] = pd.DataFrame(columns=["bc_line_id", "face_index"])

    if out.get("cells_center_xy") is not None:
        out["n_all"] = int(out["cells_center_xy"].shape[0])
    if out.get("cell_points") is not None:
        out["n_comp"] = int(out["cell_points"].shape[0])
    elif out.get("cells_center_xy") is not None:
        out["n_comp"] = int(out["cells_center_xy"].shape[0])

    return out


# =========================================================
# Warmup Time Removal from HDF Data
# =========================================================
def trim_warmup_from_hdf_data(hdf_data, warmup_steps, enabled=True):
    """
    Removes warmup time steps from all dynamic arrays in hdf_data.

    This function trims the first 'warmup_steps' from all arrays that have
    a time dimension (first axis). Static geometry data is left unchanged.

    Args:
        hdf_data (dict): Dictionary returned from load_hdf_required()
        warmup_steps (int): Number of time steps to remove from the beginning
        enabled (bool): If False, returns hdf_data unchanged (no trimming)

    Returns:
        dict: Modified hdf_data with warmup steps removed from dynamic arrays
    """
    if not enabled or warmup_steps <= 0:
        if not enabled:
            print(f"  [WARMUP] Trimming disabled (enabled=False)")
        elif warmup_steps <= 0:
            print(f"  [WARMUP] Trimming skipped (warmup_steps={warmup_steps} <= 0)")
        return hdf_data

    print(f"  [WARMUP] Starting trim: removing first {warmup_steps} time steps from all dynamic arrays")

    # List of keys that contain time-series data (first dimension is time)
    time_series_keys = [
        "wd", "vx", "vy", "cell_v",  # State variables
        "pr", "cul_pr",              # Precipitation data
        "stage_faces"                # Boundary stage data
    ]

    # Track original time dimension for validation
    original_T = None

    # Trim regular time-series arrays
    for key in time_series_keys:
        if key not in hdf_data:
            continue

        arr = hdf_data[key]
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            continue

        # Get time dimension (first axis)
        if arr.ndim == 0:
            continue  # Skip scalars

        T = arr.shape[0]

        # Set original_T from first valid array
        if original_T is None:
            original_T = T

        # Validate consistency
        if T != original_T:
            print(f"  WARNING: {key} has T={T}, expected T={original_T}. Skipping trim for this array.")
            continue

        # Check if we can trim
        if T <= warmup_steps:
            print(f"  WARNING: {key} has only {T} time steps, cannot remove {warmup_steps} warmup steps. Skipping.")
            continue

        # Trim the array
        if arr.ndim == 1:
            hdf_data[key] = arr[warmup_steps:]
        else:
            hdf_data[key] = arr[warmup_steps:, ...]

    # Handle bc_stage_data (dictionary of arrays)
    if "bc_stage_data" in hdf_data and isinstance(hdf_data["bc_stage_data"], dict):
        bc_stage_data = hdf_data["bc_stage_data"]
        for bc_id, stage_array in bc_stage_data.items():
            if not isinstance(stage_array, np.ndarray) or stage_array.size == 0:
                continue

            if stage_array.ndim < 1:
                continue

            T = stage_array.shape[0]

            # Validate consistency
            if original_T is not None and T != original_T:
                print(f"  WARNING: bc_stage_data[{bc_id}] has T={T}, expected T={original_T}. Skipping trim.")
                continue

            # Check if we can trim
            if T <= warmup_steps:
                print(f"  WARNING: bc_stage_data[{bc_id}] has only {T} time steps, cannot remove {warmup_steps} warmup steps. Skipping.")
                continue

            # Trim the array
            if stage_array.ndim == 1:
                bc_stage_data[bc_id] = stage_array[warmup_steps:]
            else:
                bc_stage_data[bc_id] = stage_array[warmup_steps:, ...]

        hdf_data["bc_stage_data"] = bc_stage_data

    # Report trimming
    if original_T is not None:
        new_T = original_T - warmup_steps
        print(f"  [WARMUP] ✓ Trimmed {warmup_steps} warmup time steps: {original_T} -> {new_T} time steps")
        print(f"  [WARMUP] Arrays trimmed: {[k for k in time_series_keys if k in hdf_data]}")
        if "bc_stage_data" in hdf_data and isinstance(hdf_data["bc_stage_data"], dict) and len(hdf_data["bc_stage_data"]) > 0:
            print(f"  [WARMUP] BC stage data trimmed: BC{list(hdf_data['bc_stage_data'].keys())}")
    else:
        print(f"  [WARMUP] ⚠ WARNING: No time-series arrays found to trim!")

    return hdf_data


# =========================================================
# Precomputed CSV & External Feature Loaders
# =========================================================
def load_cell_attributes(cfg):
    """Read cell IDs and the scalar elevation attributes zmin, zmax and relief."""
    path = cfg["paths"].get("cell_attributes_csv")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, usecols=["cell_id", "zmin", "zmax", "relief"])


def load_face_attributes(cfg):
    """Read scalar face attributes and mesh-connectivity information.

    Retain source column order and the existing face, cell, ghost and boundary IDs.
    """
    path = cfg["paths"].get("face_attributes_csv")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    scalar_columns = {
        "face_id", "left_id", "right_id", "left_ghost", "right_ghost",
        "is_boundary", "boundary_side", "bc_node", "length", "nx", "ny",
        "d_n", "n_face", "zmin", "zmax", "A_max", "Lw_max", "R_max",
    }
    df = pd.read_csv(path, usecols=lambda name: name in scalar_columns)
    for col in ["face_id", "left_id", "right_id", "left_ghost", "right_ghost", "bc_node"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def load_and_merge_external_features(static_df, feature_files, join_key="node_id"):
    """
    Loads additional features from external CSV files and merges them into a DataFrame.

    Args:
        static_df (pd.DataFrame): The base DataFrame to merge features into.
        feature_files (list): A list of dictionaries from the config, each specifying a file path.
        join_key (str, optional): The column name in the base DataFrame to join on.

    Returns:
        pd.DataFrame: The DataFrame with external features merged.
    """
    if not feature_files:
        return static_df
    merged_df = static_df.copy()
    for file_info in feature_files:
        path = file_info['path']
        join_on = file_info['join_on']
        print(f"Loading external features from {os.path.basename(path)}...")
        try:
            external_df = pd.read_csv(path, sep=r'\s+')
            if join_on != join_key and join_on in external_df.columns:
                external_df = external_df.rename(columns={join_on: join_key})
            merged_df = pd.merge(merged_df, external_df, on=join_key, how="left")
        except FileNotFoundError:
            print(f"  WARNING: File not found: {path}")
        except Exception as e:
            print(f"  WARNING: Could not process file {path}. Error: {e}")
    return merged_df


# =========================================================
# NORMALIZATION LOGIC
# =========================================================


def normalize_global_static_features(df, stats, cfg):
    """
    Applies feature-specific normalization to SCALAR static features based on config.

    It calculates stats for features if they are not already in the stats dict.
    ID columns and features listed in 'exclude_static_features' are skipped.
    Aspect values of -1 denote flat areas and are excluded from its statistics.
    Args:
        df (pd.DataFrame): The DataFrame of static node or edge features.
        stats (dict): The master dictionary of normalization statistics.
        cfg (dict): The global configuration dictionary.

    Returns:
        tuple[pd.DataFrame, dict]: A tuple containing:
            - The DataFrame with normalized columns.
            - The updated statistics dictionary.
    """
    norm_cfg = cfg.get("normalization", {})
    exclude = norm_cfg.get("exclude_static_features", [])
    feature_methods = norm_cfg.get("feature_methods", {}).get("static", {})
    default_method = norm_cfg.get("default_methods", {}).get("static", "minmax")

    id_cols = [c for c in df.columns if 'id' in c or 'src' in c or 'dst' in c]

    # Select scalar numeric attributes, excluding IDs and configured exceptions.
    cols_to_norm = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in id_cols + exclude
    ]

    for col in cols_to_norm:
        # Get normalization method from config, fallback to default
        method = feature_methods.get(col, default_method)

        if method == "no_norm":
            continue  # Skip normalization for this feature

        # Special handling for aspect: Convert -1 (flat areas) to NaN before normalization
        # This ensures -1 values are excluded from statistics calculation
        n_flat = 0
        if col == "aspect":
            # Convert -1 (flat areas) to NaN so they're excluded from stats
            n_flat = (df[col] == -1).sum()
            if n_flat > 0:
                df[col] = df[col].replace(-1, np.nan)
                print(f"  Converting {n_flat} flat areas (aspect=-1) to NaN for normalization")

        # Check if stats already exist (from previous run) and method matches
        if col in stats and stats[col].get("method") == method:
            # Use existing stats
            col_stats = stats[col]
        else:
            # Compute new stats (NaN values are automatically excluded by np.nanmin/nanmax/nanmean/nanstd)
            if method == "minmax":
                _, col_stats = min_max_normalize(df[col])
            elif method == "zscore":
                _, col_stats = zscore_normalize(df[col])
            elif method == "log1p_zscore":
                _, col_stats = log1p_zscore_normalize(df[col])
            else:
                print(f"  WARNING: Unknown normalization method '{method}' for static feature '{col}'. Using default '{default_method}'.")
                method = default_method
                if method == "minmax":
                    _, col_stats = min_max_normalize(df[col])
                elif method == "zscore":
                    _, col_stats = zscore_normalize(df[col])
                elif method == "log1p_zscore":
                    _, col_stats = log1p_zscore_normalize(df[col])

            stats[col] = col_stats

        # Apply normalization using saved stats
        col_stats = stats[col]
        if col_stats["method"] == "minmax":
            df[col], _ = min_max_normalize(df[col], min_val=col_stats["min"], max_val=col_stats["max"])
        elif col_stats["method"] == "zscore":
            df[col], _ = zscore_normalize(df[col], mean=col_stats["mean"], std=col_stats["std"])
        elif col_stats["method"] == "log1p_zscore":
            df[col], _ = log1p_zscore_normalize(df[col], log1p_mean=col_stats["log1p_mean"], log1p_std=col_stats["log1p_std"])

        # For aspect: After normalization, fill NaN (flat areas) with a sentinel value
        # Using -1 in normalized space (which will be outside [0,1] range) as a marker
        # Alternatively, we could use 0 or keep NaN (model should handle NaN)
        if col == "aspect":
            # Option 1: Fill NaN with -1 (outside normalized range, acts as sentinel)
            # Option 2: Fill NaN with 0 (treat flat as "no direction")
            # Option 3: Keep NaN (requires model to handle NaN)
            # Using Option 1: -1 as sentinel (will be outside [0,1] range)
            df[col] = df[col].fillna(-1.0)
            print(f"  Filled {n_flat} flat areas with normalized sentinel value -1.0")

    return df, stats


# =========================================================
# Geometry Builders
# =========================================================
def build_nodes(hdf_data, face_attributes_df, **kwargs):
    """
    Builds the node DataFrame, including computational cells and boundary ghost cells.

    It also creates a mapping between the original HEC-RAS node IDs and the dense
    row indices used in the graph tensors.

    Args:
        hdf_data (dict): Dictionary of data loaded from the HEC-RAS HDF file.
        face_attributes_df (pd.DataFrame): DataFrame loaded from the face attribute CSV.

    Returns:
        tuple[pd.DataFrame, dict, dict]: A tuple containing:
            - nodes_df: DataFrame with columns ['node_id', 'node_type', 'x', 'y'].
            - index_map: Dictionary for mapping between node_id and row index.
            - masks: Dictionary of boolean masks for identifying node types.
    """
    centers = hdf_data.get("cells_center_xy")
    if centers is None: raise ValueError("build_nodes: 'cells_center_xy' missing.")
    n_comp = int(hdf_data.get("n_comp", 0))
    if n_comp <= 0: raise ValueError("build_nodes: invalid 'n_comp'.")
    n_all = int(hdf_data.get("n_all", centers.shape[0]))
    ext = hdf_data.get("external_faces", pd.DataFrame())
    ext_face_ids = set()
    if "face_index" in ext.columns and len(ext):
        ext_face_ids = set(pd.to_numeric(ext["face_index"], errors="coerce").fillna(-1).astype(int).tolist())

    fc = face_attributes_df.copy()
    bnd_on_ext = fc[fc["is_boundary"].astype(bool) & fc["face_id"].isin(ext_face_ids)]
    bc_line_ghosts = set(int(g) for g in bnd_on_ext["bc_node"].astype(int).tolist() if n_comp <= g < n_all)

    ghosts_to_include = list(range(n_comp, n_all))
    rows = []
    for cid in range(n_comp): rows.append((cid, "comp", float(centers[cid, 0]), float(centers[cid, 1])))
    for gid in ghosts_to_include:
        ntype = "bghost" if gid in bc_line_ghosts else "non_bc_ghost"
        rows.append((gid, ntype, float(centers[gid, 0]), float(centers[gid, 1])))
    nodes_df = pd.DataFrame(rows, columns=["node_id", "node_type", "x", "y"])

    node_id_to_row = {nid: i for i, nid in enumerate(nodes_df["node_id"].tolist())}
    index_map = {"node_id_to_row": node_id_to_row, "row_to_node_id": nodes_df["node_id"].tolist()}

    t = nodes_df["node_type"].values
    masks = {"mask_comp": (t == "comp").astype(np.uint8), "mask_label": (t == "comp").astype(np.uint8)}
    return nodes_df, index_map, masks


def build_edges(face_attributes_df, n_comp, **kwargs):
    """
    Builds the edge DataFrame from face data, creating source-destination pairs.

    It distinguishes between internal edges (connecting two computational cells)
    and boundary edges (connecting a cell to a ghost node).

    Args:
        face_attributes_df (pd.DataFrame): DataFrame loaded from the face attribute CSV.
        n_comp (int): The number of computational cells.

    Returns:
        pd.DataFrame: DataFrame with columns ['src', 'dst', 'face_id', 'edge_type'].
    """
    edges = []
    internal = face_attributes_df[
        (face_attributes_df["left_id"] >= 0) & (face_attributes_df["left_id"] < n_comp) &
        (face_attributes_df["right_id"] >= 0) & (face_attributes_df["right_id"] < n_comp)
        ]
    for _, r in internal.iterrows():
        edges.append((int(r["left_id"]), int(r["right_id"]), int(r["face_id"]), "internal"))

    if kwargs.get("include_boundary_edges", True):
        for _, r in face_attributes_df.iterrows():
            L, R = int(r.get("left_id", -1)), int(r.get("right_id", -1))
            LG, RG = int(r.get("left_ghost", -1)), int(r.get("right_ghost", -1))
            fid = int(r.get("face_id", -1))

            # Boundary edge: comp (left) -> ghost (right)
            if (0 <= L < n_comp) and (RG >= n_comp):
                # Forward edge: comp -> ghost (I→B direction)
                edges.append((L, RG, fid, "boundary"))
                # Reverse edge: ghost -> comp (B→I direction)
                edges.append((RG, L, fid, "boundary"))

            # Boundary edge: comp (right) -> ghost (left)
            if (0 <= R < n_comp) and (LG >= n_comp):
                # Forward edge: comp -> ghost (I→B direction)
                edges.append((R, LG, fid, "boundary"))
                # Reverse edge: ghost -> comp (B→I direction)
                edges.append((LG, R, fid, "boundary"))

    return pd.DataFrame(edges, columns=["src", "dst", "face_id", "edge_type"]).drop_duplicates()


# =========================================================
# Static Feature Generation & Selection
# =========================================================
def attach_static_node_features(nodes_df, hdf_data, cell_attributes_df):
    """Attach scalar HDF properties and active elevation statistics to nodes.

    Computational nodes retain their cell area, roughness, infiltration, zmin,
    zmax, and relief. Ghost nodes keep their coordinates and zero-valued cell
    properties.
    """
    attributes_by_id = {int(r["cell_id"]): r for _, r in cell_attributes_df.iterrows()}
    area = hdf_data.get("area_cell")
    ncell = hdf_data.get("manning_n")
    infil = hdf_data.get("infiltration")
    out_rows = []
    for _, r in nodes_df.iterrows():
        nid, ntype = int(r["node_id"]), r["node_type"]
        if ntype == "comp":
            rec = attributes_by_id.get(nid, {})
            base_vals = {
                "area_cell": float(area[nid]) if area is not None and nid < len(area) else 0.0,
                "manning_n": float(ncell[nid]) if ncell is not None and nid < len(ncell) else 0.0,
                "infiltration": float(infil[nid]) if infil is not None and nid < len(infil) else 0.0,
                "zmin": float(rec.get("zmin", np.nan)),
                "zmax": float(rec.get("zmax", np.nan)),
                "relief": float(rec.get("relief", np.nan)),
            }
        else:
            base_vals = {k: 0.0 for k in [
                "area_cell", "manning_n", "infiltration", "zmin", "zmax", "relief"
            ]}
        out_rows.append({"node_id": nid, "x": r["x"], "y": r["y"], **base_vals})
    return pd.DataFrame(out_rows)

def attach_static_edge_features(edges_df, face_attributes_df, nodes_df):
    """Attach scalar face geometry and directed relative coordinates to edges.

    Connectivity and source face attributes are kept in their original order.
    Feature selection chooses d_n, dx, and dy after global normalization.
    """
    fc = face_attributes_df.set_index("face_id", drop=False)
    out_rows = []
    for _, e in edges_df.iterrows():
        fid = int(e["face_id"])
        rec = fc.loc[fid]
        out_rows.append({
            "src": int(e["src"]), "dst": int(e["dst"]),
            "face_id": fid, **rec.to_dict()
        })
    edges_static_df = pd.DataFrame(out_rows)
    node_coords = nodes_df.set_index("node_id")[["x", "y"]]
    merged_df = edges_static_df.merge(
        node_coords, left_on="src", right_index=True
    ).merge(
        node_coords, left_on="dst", right_index=True, suffixes=("_src", "_dst")
    )
    merged_df["dx"] = merged_df["x_dst"] - merged_df["x_src"]
    merged_df["dy"] = merged_df["y_dst"] - merged_df["y_src"]
    return merged_df


def select_final_features(full_df, cfg, id_cols, type):
    """Select scalar feature blocks, then filter them in source column order.

    Selectors never reorder columns. This preserves the checkpoint-sensitive
    ordering of coordinates, cell properties, terrain statistics, and GIS
    attributes. Curve and polynomial inputs are not supported.
    """
    feature_list = cfg.get("features", {}).get(f"static_{type}_features", [])
    feature_selectors = cfg.get("feature_selectors", {}).get(type, {})
    obsolete = {"hypsometry_curves", "hypsometry_polyfit", "face_curves"}
    if obsolete.intersection(feature_list) or cfg.get("features", {}).get("include_max_values", False):
        raise ValueError("GNN4CF uses scalar features; curve, polynomial, and maximum inputs are unsupported.")
    known_columns = set(id_cols)
    known_columns.update([
        "x", "y", "area_cell", "manning_n", "infiltration",
        "zmin", "zmean", "zmax", "relief"
    ])
    discovered_external_cols = [col for col in full_df.columns if col not in known_columns]
    block_columns = {}
    for feature in feature_list:
        if feature == "terrain_stats" and type == "node":
            block_columns[feature] = ["zmin", "zmax", "relief"]
        elif feature == "node_coordinates" and type == "node":
            block_columns[feature] = ["x", "y"]
        elif feature == "external_gis_features" and type == "node":
            block_columns[feature] = discovered_external_cols or cfg.get("features", {}).get("external_features", [])
        elif feature == "geometry_stats" and type == "edge":
            block_columns[feature] = ["length", "nx", "ny", "d_n"]
        elif feature == "relative_coordinates" and type == "edge":
            block_columns[feature] = ["dx", "dy"]
        else:
            block_columns[feature] = [feature]
    final_features = []
    for block_name, block_cols in block_columns.items():
        selector = feature_selectors.get(block_name)
        if selector is None:
            final_features.extend(block_cols)
        elif "include" in selector:
            final_features.extend(col for col in block_cols if col in selector["include"])
        elif "include_prefix" in selector:
            final_features.extend(
                col for col in block_cols
                if any(col.startswith(prefix) for prefix in selector["include_prefix"])
            )
        else:
            final_features.extend(block_cols)
    columns_to_keep = id_cols + [col for col in final_features if col in full_df.columns]
    unique_columns = list(dict.fromkeys(columns_to_keep))
    missing = set(final_features) - set(full_df.columns)
    if missing:
        print(f"  WARNING: Configured {type} features not found and skipped: {sorted(list(missing))}")
    print(f"Selecting {len(unique_columns) - len(id_cols)} static {type} features.")
    return full_df[unique_columns]


# =========================================================
# Dynamic Features
# =========================================================
def gather_and_compute_node_timeseries(hdf_data, nodes_df, face_attributes_df, cfg):
    """
    Assembles all raw and computed time-series features as specified in the config.

    This function is the central point for creating the full suite of dynamic
    features. It fetches raw data from the HDF, maps it to the graph nodes,
    and then computes accumulation and sea-level trend features.

    Args:
        hdf_data (dict): Data loaded from the HEC-RAS HDF for the current event.
        nodes_df (pd.DataFrame): The main node DataFrame.
        face_attributes_df (pd.DataFrame): The face attribute DataFrame.
        cfg (dict): The global configuration dictionary.

    Returns:
        dict: A dictionary where keys are feature names (matching the config)
              and values are NumPy arrays of shape (T, N).
    """
    # --- 1. Get dimensions and masks ---
    wd_arr = hdf_data.get("wd")
    T = wd_arr.shape[0] if isinstance(wd_arr, np.ndarray) and wd_arr.size > 0 else 0
    N, n_comp = len(nodes_df), int(hdf_data.get("n_comp", 0))
    if T == 0: return {}
    comp_mask = (nodes_df["node_type"] == "comp").to_numpy()

    # --- 2. Initialize the output dictionary ---
    ts_data = {}

    # --- 3. Place raw HEC-RAS results for computational cells ---
    for feature_name, hdf_key in [
        ("wd", "wd"),
        ("vx", "vx"),
        ("vy", "vy"),
        ("cell_v", "cell_v"),
        ("rainfall_rate", "pr"),
        ("acc_rainfall", "cul_pr")
    ]:
        arr_in = hdf_data.get(hdf_key)
        arr_out = np.zeros((T, N), dtype=np.float32)

        # Ensure the input array is valid before processing
        if isinstance(arr_in, np.ndarray) and arr_in.shape[0] >= T:
            # Check the shape to handle the difference between global and per-cell data
            if arr_in.ndim == 1:  # This is for global data like rainfall_rate (shape: (T,))
                arr_out[:T, comp_mask] = arr_in[:T].reshape(-1, 1)
            elif arr_in.ndim == 2 and arr_in.shape[1] >= comp_mask.sum():  # For per-cell data
                arr_out[:T, comp_mask] = arr_in[:T, :comp_mask.sum()]

        ts_data[feature_name] = arr_out


    # Sea Level
    sea_level = np.zeros((T, N), dtype=np.float32)
    ext = hdf_data.get("external_faces")
    bc_stage_data = hdf_data.get("bc_stage_data", {})  # BC-specific stage data

    if isinstance(ext, pd.DataFrame) and len(ext) > 0:
        fc = face_attributes_df.set_index("face_id")
        f2g = fc["bc_node"].to_dict()
        n2r = {nid: i for i, nid in enumerate(nodes_df["node_id"])}

        # Check if we have multiple boundaries (BC-specific data)
        if len(bc_stage_data) > 1 and "bc_line_id" in ext.columns:
            # Multiple boundaries: Read from BC-specific arrays (BC1/BC1 - Stage per Face, etc.)
            # Group faces by BC to map them correctly
            for bc_id, bc_stage_array in bc_stage_data.items():
                # HEC-RAS uses 0-indexed BC Line IDs (0, 1, 2) but we read BC-specific datasets as 1-indexed (BC1, BC2, BC3)
                # Try both: bc_id (1-indexed) and bc_id - 1 (0-indexed)
                bc_faces_1idx = ext[ext["bc_line_id"] == bc_id].copy()
                bc_faces_0idx = ext[ext["bc_line_id"] == (bc_id - 1)].copy()

                # Use whichever has matching face count
                n_faces_in_stage = bc_stage_array.shape[1] if bc_stage_array.ndim == 2 else 0

                if len(bc_faces_0idx) == n_faces_in_stage:
                    bc_faces = bc_faces_0idx
                elif len(bc_faces_1idx) == n_faces_in_stage:
                    bc_faces = bc_faces_1idx
                else:
                    # Fallback: use 1-indexed if it has any faces, otherwise 0-indexed
                    bc_faces = bc_faces_1idx if len(bc_faces_1idx) > 0 else bc_faces_0idx

                if len(bc_faces) == 0:
                    continue

                # bc_stage_array shape: [T, n_faces_for_this_bc]
                if bc_stage_array.ndim != 2 or bc_stage_array.shape[0] < T:
                    continue

                n_faces_in_bc = bc_stage_array.shape[1]
                if len(bc_faces) != n_faces_in_bc:
                    print(f"    WARNING: BC{bc_id} has {len(bc_faces)} faces in external_faces but {n_faces_in_bc} columns in stage array")

                # Map each face to its boundary node
                # Stage-array columns follow the same order as
                # faces appear in the External Faces dataset when filtered by bc_line_id.
                # HEC-RAS typically stores them in the order they appear in the geometry dataset.
                # We preserve the original order from external_faces (don't sort) to match HEC-RAS ordering.

                for face_idx, (k, row) in enumerate(bc_faces.iterrows()):
                    fid = int(row["face_index"])

                    # Get stage value for this face (use face_idx as column index)
                    # This assumes the stage array columns are in the same order as faces
                    # in external_faces when filtered by bc_line_id (preserving HEC-RAS order)
                    if face_idx < n_faces_in_bc:
                        stage_value = bc_stage_array[:T, face_idx]
                    else:
                        # Fallback: use last column if index out of bounds
                        stage_value = bc_stage_array[:T, -1] if n_faces_in_bc > 0 else np.zeros(T)

                    # Map to boundary node
                    gid = f2g.get(fid)
                    if gid is not None and gid >= 0:
                        ridx = n2r.get(gid)
                        if ridx is not None:
                            sea_level[:T, ridx] = stage_value
        else:
            # Single boundary: Use old format (BC Line - Stage per Face)
            stage = hdf_data.get("stage_faces")
            if isinstance(stage, np.ndarray) and stage.ndim == 2 and len(ext) == stage.shape[1] and stage.shape[0] >= T:
                for k, fid in enumerate(ext["face_index"].astype(int)):
                    gid = f2g.get(fid)
                    if gid is not None and gid >= 0:
                        ridx = n2r.get(gid)
                        if ridx is not None:
                            sea_level[:T, ridx] = stage[:T, k]

    # =========================================================
    # VALIDATION: Check sea_level assignment
    # =========================================================
    if isinstance(ext, pd.DataFrame) and sea_level.shape[0] > 0:
        boundary_mask = (nodes_df['node_type'] == 'bghost').to_numpy()
        has_bc_line_id = "bc_line_id" in ext.columns
        bc_stage_data = hdf_data.get("bc_stage_data", {})
        is_multiple_bc = len(bc_stage_data) > 1

        if has_bc_line_id and is_multiple_bc:
            print("  [VALIDATION] Checking sea_level assignment for multiple boundary conditions...")
        elif has_bc_line_id:
            print("  [VALIDATION] Checking sea_level assignment for single boundary condition...")
        else:
            print("  [VALIDATION] Checking sea_level assignment...")

        # Map faces to BC lines and nodes (if bc_line_id available)
        bc_to_nodes = {}  # {bc_line_id: [node_indices]}
        if has_bc_line_id:
            fc = face_attributes_df.set_index("face_id")
            f2g = fc["bc_node"].to_dict()
            n2r = {nid: i for i, nid in enumerate(nodes_df["node_id"])}

            for k, row in ext.iterrows():
                if pd.isna(row.get("bc_line_id")):
                    continue
                bc_id = int(row["bc_line_id"])
                # Note: bc_line_id is 0-indexed (0, 1, 2) from HEC-RAS
                # BC1 corresponds to bc_line_id=0, BC2 to bc_line_id=1, BC3 to bc_line_id=2
                # So we should NOT skip bc_id == 0 - it's a valid BC (BC1)
                fid = int(row["face_index"])
                gid = f2g.get(fid)
                if gid is not None and gid >= 0:
                    ridx = n2r.get(gid)
                    if ridx is not None and boundary_mask[ridx]:
                        if bc_id not in bc_to_nodes:
                            bc_to_nodes[bc_id] = []
                        bc_to_nodes[bc_id].append(ridx)

        # Check each BC line (if multiple) or overall boundary (if single)
        # Reference stage for inactive coastal boundaries in this study.
        # Used only for diagnostics; simulated inputs and predictions are not altered.
        normal_stage_value = 1.092
        variance_threshold = 1e-6  # Minimum variance to consider BC as active

        if is_multiple_bc and len(bc_to_nodes) > 0:
            # Multiple boundaries: check each BC separately
            print(f"    Found {len(bc_to_nodes)} boundary condition line(s): {sorted(bc_to_nodes.keys())}")

            for bc_id in sorted(bc_to_nodes.keys()):
                node_indices = np.unique(bc_to_nodes[bc_id])
                if len(node_indices) == 0:
                    continue

                # Get sea_level values for this BC's nodes
                bc_sea_level = sea_level[:, node_indices]  # [T, n_nodes_for_this_bc]

                # Compute statistics
                bc_mean = np.mean(bc_sea_level)
                bc_std = np.std(bc_sea_level)
                bc_min = np.min(bc_sea_level)
                bc_max = np.max(bc_sea_level)
                bc_variance = np.var(bc_sea_level)

                # Check if values are constant (inactive BC) or variable (active BC)
                is_constant = bc_variance < variance_threshold
                is_near_normal = abs(bc_mean - normal_stage_value) < 0.01  # Within 1cm

                status = "INACTIVE (constant)" if is_constant else "ACTIVE (variable)"
                if is_constant and is_near_normal:
                    status += f" [normal={normal_stage_value:.3f}m]"

                print(f"    BC{bc_id}: {len(node_indices)} nodes, {status}")
                print(f"      Stats: mean={bc_mean:.6f}m, std={bc_std:.6f}m, min={bc_min:.6f}m, max={bc_max:.6f}m, var={bc_variance:.9f}")

                # Additional check: all nodes in same BC should have same values (within tolerance)
                if len(node_indices) > 1:
                    node_means = np.mean(bc_sea_level, axis=0)
                    node_std = np.std(node_means)
                    if node_std > 0.001:  # More than 1mm difference
                        print(f"      WARNING: Nodes in BC{bc_id} have different mean values (std={node_std:.6f}m)")
                    else:
                        print(f"      OK: All nodes in BC{bc_id} have consistent values")

            # Summary for multiple BCs
            active_bcs = [bc_id for bc_id in bc_to_nodes.keys()
                         if np.var(sea_level[:, np.unique(bc_to_nodes[bc_id])]) >= variance_threshold]
            inactive_bcs = [bc_id for bc_id in bc_to_nodes.keys() if bc_id not in active_bcs]

            if len(active_bcs) == 1:
                print(f"    ✓ Validation PASSED: One active BC (BC{active_bcs[0]}), {len(inactive_bcs)} inactive BC(s)")
            elif len(active_bcs) == 0:
                print(f"    ⚠ WARNING: No active BC detected (all BCs appear constant)")
            else:
                print(f"    ⚠ WARNING: Multiple active BCs detected: {active_bcs}")
        else:
            # Single boundary: check overall boundary nodes
            boundary_node_indices = np.where(boundary_mask)[0]
            if len(boundary_node_indices) > 0:
                boundary_sea_level = sea_level[:, boundary_node_indices]  # [T, n_boundary_nodes]

                # Compute overall statistics
                overall_mean = np.mean(boundary_sea_level)
                overall_std = np.std(boundary_sea_level)
                overall_min = np.min(boundary_sea_level)
                overall_max = np.max(boundary_sea_level)
                overall_variance = np.var(boundary_sea_level)

                print(f"    Single boundary: {len(boundary_node_indices)} boundary nodes")
                print(f"      Stats: mean={overall_mean:.6f}m, std={overall_std:.6f}m, min={overall_min:.6f}m, max={overall_max:.6f}m, var={overall_variance:.9f}")

                # Check consistency across nodes
                if len(boundary_node_indices) > 1:
                    node_means = np.mean(boundary_sea_level, axis=0)
                    node_std = np.std(node_means)
                    if node_std > 0.001:  # More than 1mm difference
                        print(f"      WARNING: Boundary nodes have different mean values (std={node_std:.6f}m)")
                    else:
                        print(f"      OK: All boundary nodes have consistent values")

                if overall_variance < variance_threshold:
                    print(f"    ⚠ WARNING: Boundary appears constant (variance={overall_variance:.9f})")
                else:
                    print(f"    ✓ Validation PASSED: Boundary has variable values")
    #########################################################
    #########################################################



    ts_data['sea_level'] = sea_level
    ts_data['sea_level_trend'] = np.concatenate([np.zeros((1, N)), np.sign(np.diff(sea_level, axis=0))], axis=0)

    return ts_data


# =========================================================
# Graph Snapshot Creation and Saving
# =========================================================
def create_graph_snapshots(nodes_static_df, edges_df, edges_static_df, raw_ts, cfg, node_type_tensor, edge_type_tensor):
    """Constructs a sequence of graph snapshots for time-series forecasting.

    Assemble model-ready graph snapshots from selected features. It takes
    the static properties of the graph (node/edge features and connectivity) and
    combines them with a full time-series of dynamic data.

    Using a sliding window approach, it generates a list of `torch_geometric.data.Data`
    objects. Each object, or "snapshot," represents one complete training sample.
    It contains the static graph features concatenated with a window of dynamic
    features from the past (`past_steps`) as input `x`.

    The snapshot also includes several target and auxiliary tensors:
    - `y`: The primary prediction targets (e.g., `wd`) for a future window.
    - `y_aux`: Optional secondary ground truths (e.g., `cell_v`) for the same
      future window, retained for HDF/cache compatibility.
    - `future_drivers`: The known external forces (e.g., `rainfall_rate`) for
      the future window, essential for multi-step "rollout" predictions.

    Args:
        nodes_static_df (pd.DataFrame): A DataFrame containing the selected,
            and purely numerical static features for every node in the graph.
        edges_df (pd.DataFrame): A DataFrame defining the graph's connectivity,
            containing 'src' and 'dst' columns that map to `node_id`.
        edges_static_df (pd.DataFrame): A DataFrame containing the selected,
            and purely numerical static features for every edge in the graph.
        raw_ts (dict): A dictionary of NumPy arrays, where each key is a dynamic
            variable name and the value is an array of shape `(T, N)`.
        cfg (dict): The global configuration dictionary, used to access windowing
            parameters and dynamic feature lists.
        node_type_tensor (torch.Tensor): A 1D tensor of shape `(N,)` that maps
            each node to an integer type.
        edge_type_tensor (torch.Tensor): A 1D tensor of shape `(E,)` that maps
            each edge to an integer type.

    Returns:
        list[Data]: A list of `torch_geometric.data.Data` objects. Each object
            is a self-contained graph snapshot ready for model training.
    """
    print("Creating graph snapshots...")
    features_cfg = cfg.get("features", {})
    window_cfg = cfg.get("window", {})
    past_steps = window_cfg["past_steps"]
    future_steps = window_cfg["future_steps"]

    # Read all variable lists from the config file, providing empty lists as defaults
    label_vars = window_cfg.get("label_vars", [])
    aux_vars = window_cfg.get("aux_vars", [])
    state_vars = features_cfg.get("dynamic_input_state_variables", [])
    driver_vars = features_cfg.get("dynamic_input_drivers", [])
    input_vars = state_vars + driver_vars

    # Determine the number of valid snapshots that can be created
    # Use first available key from label_vars, or fall back to any key in raw_ts
    first_key = next((k for k in label_vars if k in raw_ts), next(iter(raw_ts)) if raw_ts else None)
    if first_key is None:
        raise ValueError("raw_ts is empty or no valid keys found")
    T, N = raw_ts[first_key].shape
    T_valid = T - past_steps - future_steps + 1
    if T_valid <= 0:
        print("WARNING: Not enough time steps to create snapshots. Returning empty list.")
        return []

    # --- 1. Prepare static graph components as tensors (once) ---
    x_static = torch.tensor(nodes_static_df.drop(columns=['node_id']).values, dtype=torch.float)
    edge_attr_static = torch.tensor(edges_static_df.drop(columns=['src', 'dst', 'face_id']).values, dtype=torch.float)
    edge_index = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)

    # --- 2. Prepare all dynamic data stacks (once) for efficient slicing ---
    dynamic_inputs = np.stack([raw_ts[key] for key in input_vars if key in raw_ts], axis=-1)
    dynamic_labels = np.stack([raw_ts[key] for key in label_vars if key in raw_ts], axis=-1)
    future_drivers_stack = np.stack([raw_ts[key] for key in driver_vars if key in raw_ts], axis=-1)

    dynamic_aux = None
    if aux_vars:
        dynamic_aux = np.stack([raw_ts[key] for key in aux_vars if key in raw_ts], axis=-1)


    # --- 3. Loop through time to create each snapshot ---
    snapshots = []
    for t in tqdm(range(T_valid), desc="  Building snapshots"):
        input_slice = slice(t, t + past_steps)
        label_slice = slice(t + past_steps, t + past_steps + future_steps)

        # Assemble node features (x), labels (y), and future drivers
        x_dynamic = torch.tensor(dynamic_inputs[input_slice, :, :], dtype=torch.float).permute(1, 0, 2).reshape(N, -1)
        x = torch.cat([x_static, x_dynamic], dim=1)
        y = torch.tensor(dynamic_labels[label_slice, :, :], dtype=torch.float).permute(1, 0, 2).reshape(N, -1)
        future_drivers = torch.tensor(future_drivers_stack[label_slice, :, :], dtype=torch.float).permute(1, 0,
                                                                                                          2).reshape(N,
                                                                                                                     -1)

        # Create a dictionary to hold all data for the snapshot
        snapshot_data = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr_static,
            'y': y,
            'future_drivers': future_drivers,
            'time_index': torch.tensor([t], dtype=torch.long),
            'node_type': node_type_tensor,
            'edge_type': edge_type_tensor
        }

        # Conditionally add the auxiliary targets if they exist
        if dynamic_aux is not None:
            y_aux = torch.tensor(dynamic_aux[label_slice, :, :], dtype=torch.float)
            snapshot_data['y_aux'] = y_aux.permute(1, 0, 2).reshape(N, -1)

            # Create the PAST auxiliary data (x_aux) using the *same* stack
            x_aux = torch.tensor(dynamic_aux[input_slice, :, :], dtype=torch.float)
            snapshot_data['x_aux'] = x_aux.permute(1, 0, 2).reshape(N, -1)

        # Create the PyG Data object and append it to the list
        snapshot = Data(**snapshot_data)
        snapshots.append(snapshot)

    return snapshots

def save_snapshots(snapshots, event_name, save_dir):
    """
    Saves a list of graph snapshots for an event to a pickle file.

    Args:
        snapshots (list[Data]): The list of graph snapshots to save.
        event_name (str): The name of the event (used for the filename).
        save_dir (str): The directory to save the file in.
    """
    if not snapshots:
        print(f"No snapshots to save for event '{event_name}'.")
        return
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{event_name}_snapshots.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(snapshots, f)
    print(f"\nSaved {len(snapshots)} snapshots for event '{event_name}' to:\n  {file_path}")


def print_example_snapshot(snapshots):
    """Prints a summary of a single example snapshot from the list."""
    if not snapshots:
        print("\nNo snapshots created to display.")
        return
    print("\n================ EXAMPLE SNAPSHOT (first window) ================")
    snapshot = snapshots[0]
    print(snapshot)
    print(f"\nNode features 'x' shape: {snapshot.x.shape}")
    print(f"Edge index 'edge_index' shape: {snapshot.edge_index.shape}")
    print(f"Edge features 'edge_attr' shape: {snapshot.edge_attr.shape}")
    print(f"Node labels 'y' shape: {snapshot.y.shape}")
    print(f"Time index of this window: {snapshot.time_index.item()}")
    print("=================================================================\n")


def summarize_graph(nodes_df, edges_df):
    """Prints a concise summary of the graph's size."""
    n_comp = (nodes_df["node_type"] == "comp").sum()
    print(f"=== Graph summary ===\nnodes: {len(nodes_df)} (comp={n_comp}), edges: {len(edges_df)}")

# =========================================================
# Interactive feature inspection
# =========================================================
def print_node_features_by_id(node_id, nodes_df, nodes_static_df, raw_ts, cfg, index_map):
    """
    Provides a detailed printout of all features for a specific node ID.

    This function displays the node's static properties and its dynamic features,
    labels, auxiliary data, and future drivers for the final time window,
    helping to validate the entire data assembly process.

    Args:
        node_id (int): The ID of the node to inspect.
        nodes_df (pd.DataFrame): DataFrame with basic node info (id, type, coords).
        nodes_static_df (pd.DataFrame): DataFrame with selected static node features.
        raw_ts (dict): Dictionary of full, normalized time-series arrays.
        cfg (dict): The global configuration dictionary.
        index_map (dict): The mapping from node_id to tensor row index.
    """
    print("\n--- Feature inspection ---")
    print(f"================== NODE ID: {node_id} ==================")
    try:
        # --- Basic Information ---
        node_info = nodes_df[nodes_df["node_id"] == node_id].iloc[0]
        row_idx = index_map["node_id_to_row"][node_id]
        print(f"Node Type: '{node_info['node_type']}' | Tensor Row Index: {row_idx}")

        # --- Static Features ---
        static_features = nodes_static_df[nodes_static_df["node_id"] == node_id].iloc[0]
        print("\n--- Static Features ---")
        for feature, value in static_features.drop("node_id").items():
            print(f"  {feature:<20}: {value:.4f}")

        # --- Dynamic Features (for both first and last training windows) ---
        past_steps = cfg["window"]["past_steps"]
        future_steps = cfg["window"]["future_steps"]

        # Get all feature lists from config
        state_vars = cfg["features"].get("dynamic_input_state_variables", [])
        driver_vars = cfg["features"].get("dynamic_input_drivers", [])
        input_vars = state_vars + driver_vars
        label_vars = cfg["window"].get("label_vars", [])

        # Use first available key from label_vars, or fall back to any key in raw_ts
        first_key = next((k for k in label_vars if k in raw_ts), next(iter(raw_ts)) if raw_ts else None)
        if first_key is None:
            raise ValueError("raw_ts is empty or no valid keys found")
        T = raw_ts[first_key].shape[0]
        start_t_first = 0  # First time window
        start_t_last = T - past_steps - future_steps  # Last time window
        aux_vars = cfg["window"].get("aux_vars", [])

        # Helper function to print window data
        def print_window_data(window_name, start_t, is_first=True):
            print(f"\n--- {window_name} (time steps t={start_t} to t={start_t + past_steps - 1}) ---")

            # --- Input Window ---
            print(f"  Input Window:")
            for t_step in range(past_steps):
                print(f"    t-{past_steps - 1 - t_step}:")
                current_t = start_t + t_step
                for var in input_vars:
                    value = raw_ts.get(var, np.zeros((T, len(nodes_df))))[current_t, row_idx]
                    print(f"      {var:<20}: {value:.4f}")

            # --- Auxiliary Input Window (x_aux) ---
            if aux_vars:
                print(f"  Auxiliary Input Window (x_aux):")
                for t_step in range(past_steps):
                    current_t = start_t + t_step
                    for var in aux_vars:
                        value = raw_ts.get(var, np.zeros((T, len(nodes_df))))[current_t, row_idx]
                        print(f"      {var:<20}: {value:.4f}")

            # --- Label Window ---
            print(f"  Label Window (time steps t={start_t + past_steps} to t={start_t + past_steps + future_steps - 1}):")
            for t_step in range(future_steps):
                current_t = start_t + past_steps + t_step
                for var in label_vars:
                    value = raw_ts.get(var, np.zeros((T, len(nodes_df))))[current_t, row_idx]
                    print(f"      {var:<20}: {value:.4f}")

            # --- Auxiliary Label Window (y_aux) ---
            if aux_vars:
                print(f"  Auxiliary Label Window (y_aux):")
                for t_step in range(future_steps):
                    current_t = start_t + past_steps + t_step
                    for var in aux_vars:
                        value = raw_ts.get(var, np.zeros((T, len(nodes_df))))[current_t, row_idx]
                        print(f"      {var:<20}: {value:.4f}")

            # --- Future Drivers Window ---
            if driver_vars:
                print(f"  Future Drivers Window:")
                for t_step in range(future_steps):
                    current_t = start_t + past_steps + t_step
                    for var in driver_vars:
                        value = raw_ts.get(var, np.zeros((T, len(nodes_df))))[current_t, row_idx]
                        print(f"      {var:<20}: {value:.4f}")

        # Print first time window
        print_window_data("Dynamic Features (First Time Window)", start_t_first, is_first=True)

        # Print last time window (only if different from first)
        if start_t_last != start_t_first:
            print_window_data("Dynamic Features (Last Time Window)", start_t_last, is_first=False)

    except (IndexError, KeyError) as e:
        print(f"ERROR: Node with ID {node_id} not found or data missing. Details: {e}")
    print("===================================================")

def print_edge_features_by_id(face_id, edges_df, edges_static_df, nodes_df, raw_ts, cfg, index_map):
    """
    Provides a detailed printout of all features for a specific edge (face) ID.

    This function displays the edge's static properties and provides the dynamic
    context by showing the features of its connected source and destination nodes.

    Args:
        face_id (int): The face_id of the edge to inspect.
        edges_df (pd.DataFrame): DataFrame with basic edge info (src, dst, type).
        edges_static_df (pd.DataFrame): DataFrame with selected static edge features.
        nodes_df (pd.DataFrame): DataFrame with basic node info.
        raw_ts (dict): Dictionary of full, normalized time-series arrays.
        cfg (dict): The global configuration dictionary.
        index_map (dict): The mapping from node_id to tensor row index.
    """
    print("\n--- Feature inspection ---")
    print(f"================== EDGE (FACE ID: {face_id}) ==================")
    try:
        # --- Basic Information ---
        edge_info = edges_df[edges_df["face_id"] == face_id].iloc[0]
        src_id, dst_id = int(edge_info["src"]), int(edge_info["dst"])
        print(f"Edge Type: '{edge_info['edge_type']}'")
        print(f"Connects: Node {src_id} -> Node {dst_id}")

        # --- Static Features ---
        static_features = edges_static_df[edges_static_df["face_id"] == face_id].iloc[0]
        print("\n--- Static Edge Features ---")
        for feature, value in static_features.drop(["src", "dst", "face_id"]).items():
            print(f"  {feature:<20}: {value:.4f}")

        # --- Dynamic Context from Nodes ---
        print("\n--- Dynamic Context (Last Time Window) ---")
        past_steps = cfg["window"]["past_steps"]
        label_vars = cfg["window"].get("label_vars", [])

        # Use first available key from label_vars, or fall back to any key in raw_ts
        first_key = next((k for k in label_vars if k in raw_ts), next(iter(raw_ts)) if raw_ts else None)
        if first_key is None:
            raise ValueError("raw_ts is empty or no valid keys found")
        T = raw_ts[first_key].shape[0]
        start_t = T - past_steps - cfg["window"]["future_steps"]
        state_vars = cfg["features"]["dynamic_input_state_variables"] # Show key states

        for node_label, node_id in [("Source", src_id), ("Destination", dst_id)]:
            node_type = nodes_df[nodes_df.node_id == node_id].iloc[0]['node_type']
            print(f"\n  {node_label} Node {node_id} (type: '{node_type}') at t={start_t} to t={start_t + past_steps - 1}:")
            row_idx = index_map["node_id_to_row"][node_id]
            for var in state_vars:
                values = raw_ts[var][start_t : start_t + past_steps, row_idx]
                values_str = ', '.join([f'{v:.4f}' for v in values])
                print(f"    {var:<20}: [{values_str}]")

    except (IndexError, KeyError):
        print(f"ERROR: Edge with Face ID {face_id} not found in the processed data.")
    print("=========================================================")


# =========================================================
# Snapshot data validation
# =========================================================
def check_snapshots_for_nan(cfg):
    """
    Loads all saved .pkl files from the snapshot directory and checks
    every single snapshot for NaN or Inf values in 'x' and 'edge_attr'.
    """
    print("\n--- Running Snapshot Integrity Check ---")
    output_dir = cfg["paths"].get("output_dir", "output_data")
    snapshot_dir = cfg["paths"].get("snapshot_dir", os.path.join(output_dir, "graph_snapshots"))

    pkl_files = sorted(glob.glob(os.path.join(snapshot_dir, "*.pkl")))
    if not pkl_files:
        print(f"⚠️  No .pkl files found in {snapshot_dir} to check.")
        return False

    total_files = len(pkl_files)
    total_snapshots_checked = 0
    bad_files_found = []

    # Use tqdm for a progress bar
    for pkl_file in tqdm(pkl_files, desc="Checking .pkl files"):
        file_basename = os.path.basename(pkl_file)
        try:
            with open(pkl_file, "rb") as f:
                event_snapshots = pickle.load(f)

            total_snapshots_checked += len(event_snapshots)

            # Check each snapshot in the file
            for i, snapshot in enumerate(event_snapshots):
                x_has_nan = torch.isnan(snapshot.x).any()
                edge_has_nan = torch.isnan(snapshot.edge_attr).any()

                x_has_inf = torch.isinf(snapshot.x).any()
                edge_has_inf = torch.isinf(snapshot.edge_attr).any()

                # If we find any problem, log it and break from the inner loop
                if x_has_nan or edge_has_nan or x_has_inf or edge_has_inf:
                    reason = []
                    if x_has_nan: reason.append("'x' has NaN")
                    if x_has_inf: reason.append("'x' has Inf")
                    if edge_has_nan: reason.append("'edge_attr' has NaN")
                    if edge_has_inf: reason.append("'edge_attr' has Inf")

                    bad_files_found.append((file_basename, i, ", ".join(reason)))
                    break  # Stop checking this file, we know it's bad

        except Exception as e:
            print(f"\nError loading {file_basename}: {e}")
            bad_files_found.append((file_basename, -1, f"Failed to load: {e}"))

    # --- Report Results ---
    print(f"\nChecked {total_snapshots_checked} snapshots across {total_files} files.")

    if not bad_files_found:
        print("✅ SUCCESS: All snapshots are clean (no NaN or Inf values found).")
        print("----------------------------------------\n")
        return True
    else:
        print(f"🔥 ERROR: Found {len(bad_files_found)} file(s) with corrupt data.")
        for (filename, snap_idx, reason_str) in bad_files_found:
            if snap_idx == -1:
                print(f"  - {filename}: {reason_str}")
            else:
                print(f"  - {filename} (at snapshot index {snap_idx}): {reason_str}")

        print("\n  Verify missing-value handling in static feature assembly.")
        print("  Check attach_static_node_features and attach_static_edge_features.")
        print("  Regenerate affected snapshots after correcting the source data.")
        print("----------------------------------------\n")
        return False

# =========================================================
# __main__ Execution Block
# =========================================================
if __name__ == "__main__":
    cfg = load_config()
    norm_cfg = cfg.get("normalization", {})
    norm_enabled = norm_cfg.get("enabled", False)
    output_dir = cfg["paths"]["output_dir"]
    stats_path = os.path.join(output_dir, "normalization_stats.json")
    all_stats = load_stats_file(stats_path)

    print("--- Processing Static Data (once for all events) ---")
    cell_attributes_df = load_cell_attributes(cfg)
    face_attributes_df = load_face_attributes(cfg)

    hdf_dir = cfg["paths"]["copy_each_hdf_to"]
    try:
        sample_hdf_path = glob.glob(os.path.join(hdf_dir, "*.hdf"))[0]
    except IndexError:
        raise FileNotFoundError(f"No .hdf files found in '{hdf_dir}'")

    print(f"Using sample HDF for geometry: {os.path.basename(sample_hdf_path)}")
    geom_data = load_hdf_required(sample_hdf_path, cfg)
    nodes_df, index_map, masks = build_nodes(geom_data, face_attributes_df)
    edges_df = build_edges(face_attributes_df, n_comp=geom_data["n_comp"])

    nodes_static_df = attach_static_node_features(nodes_df, geom_data, cell_attributes_df)
    edges_static_df = attach_static_edge_features(edges_df, face_attributes_df, nodes_df)
    nodes_static_df = load_and_merge_external_features(nodes_static_df, cfg["paths"].get("external_node_features", []))
    # Fill ALL NaNs (from cell attributes and external features) with 0.0
    nodes_static_df = nodes_static_df.fillna(0.0)


    if norm_enabled:
        print("Applying global normalization to static features...")
        nodes_static_df, all_stats = normalize_global_static_features(nodes_static_df, all_stats, cfg)
        edges_static_df, all_stats = normalize_global_static_features(edges_static_df, all_stats, cfg)

    final_nodes_static_df = select_final_features(nodes_static_df, cfg, ["node_id"], "node")
    final_edges_static_df = select_final_features(edges_static_df, cfg, ["src", "dst", "face_id"], "edge")
    summarize_graph(nodes_df, edges_df)

    node_type_map = {'comp': 0, 'bghost': 1, 'non_bc_ghost': 2}
    node_type_tensor = torch.tensor(nodes_df['node_type'].map(node_type_map).values, dtype=torch.long)
    edge_type_map = {'internal': 0, 'boundary': 1}
    edge_type_tensor = torch.tensor(edges_df['edge_type'].map(edge_type_map).values, dtype=torch.long)

    hdf_files = sorted(glob.glob(os.path.join(hdf_dir, "*.hdf")))
    print(f"\n--- Found {len(hdf_files)} events to process in '{hdf_dir}' ---")

    for hdf_path in hdf_files:
        event_name = os.path.splitext(os.path.basename(hdf_path))[0]
        print(f"\n--- Processing Event: {event_name} ---")
        hdf_data = load_hdf_required(hdf_path, cfg)

        # --- Trim warmup time steps if enabled ---
        window_cfg = cfg.get("window", {})
        warmup_steps = window_cfg.get("warmup_steps", 0)
        warmup_enabled = window_cfg.get("warmup_enabled", False)
        print(f"  [WARMUP] warmup_enabled={warmup_enabled}, warmup_steps={warmup_steps}")
        hdf_data = trim_warmup_from_hdf_data(hdf_data, warmup_steps, enabled=warmup_enabled)

        # Assemble hydraulic state and forcing time series.
        raw_ts = gather_and_compute_node_timeseries(hdf_data, nodes_df, face_attributes_df, cfg)

        if norm_enabled:
            norm_cfg = cfg.get("normalization", {})
            feature_methods = norm_cfg.get("feature_methods", {}).get("dynamic", {})
            default_method = norm_cfg.get("default_methods", {}).get("dynamic", "zscore")

            # Get boundary ghost node mask for sea_level normalization
            # sea_level is only applied to boundary ghost nodes (bghost).
            # Comp and non_bc_ghost nodes should remain zero (not normalized).
            boundary_mask = (nodes_df['node_type'] == 'bghost').to_numpy()  # Boolean mask for boundary nodes

            for var in raw_ts.keys():
                # Get normalization method from config, fallback to default
                method = feature_methods.get(var, default_method)

                if method == "no_norm":
                    continue  # Skip normalization for this feature

                # Special handling for sea_level: normalize ONLY boundary ghost nodes
                if var == 'sea_level':
                    # Get or compute stats from boundary nodes only
                    if var in all_stats and all_stats[var].get("method") == method:
                        var_stats = all_stats[var]
                    else:
                        # Compute stats from boundary nodes only (if not in all_stats or method changed)
                        sea_level_boundary = raw_ts[var][:, boundary_mask]  # [T, n_boundary_nodes]

                        if method == "minmax":
                            _, var_stats = min_max_normalize(sea_level_boundary)
                        elif method == "zscore":
                            _, var_stats = zscore_normalize(sea_level_boundary)
                        elif method == "log1p_zscore":
                            _, var_stats = log1p_zscore_normalize(sea_level_boundary)
                        else:
                            print(f"  WARNING: Unknown method '{method}' for sea_level. Using zscore.")
                            method = "zscore"
                            _, var_stats = zscore_normalize(sea_level_boundary)

                        all_stats[var] = var_stats
                        # Format stats based on normalization method
                        if method == "minmax":
                            min_val = var_stats.get('min', 'N/A')
                            max_val = var_stats.get('max', 'N/A')
                            min_str = f"{min_val:.6f}" if isinstance(min_val, (int, float)) else str(min_val)
                            max_str = f"{max_val:.6f}" if isinstance(max_val, (int, float)) else str(max_val)
                            print(f"  Computed sea_level stats from boundary nodes only (method={method}): "
                                  f"min={min_str}, max={max_str}")
                        elif method == "zscore":
                            mean_val = var_stats.get('mean', 'N/A')
                            std_val = var_stats.get('std', 'N/A')
                            mean_str = f"{mean_val:.6f}" if isinstance(mean_val, (int, float)) else str(mean_val)
                            std_str = f"{std_val:.6f}" if isinstance(std_val, (int, float)) else str(std_val)
                            print(f"  Computed sea_level stats from boundary nodes only (method={method}): "
                                  f"mean={mean_str}, std={std_str}")
                        elif method == "log1p_zscore":
                            log1p_mean_val = var_stats.get('log1p_mean', 'N/A')
                            log1p_std_val = var_stats.get('log1p_std', 'N/A')
                            mean_str = f"{log1p_mean_val:.6f}" if isinstance(log1p_mean_val, (int, float)) else str(log1p_mean_val)
                            std_str = f"{log1p_std_val:.6f}" if isinstance(log1p_std_val, (int, float)) else str(log1p_std_val)
                            print(f"  Computed sea_level stats from boundary nodes only (method={method}): "
                                  f"log1p_mean={mean_str}, log1p_std={std_str}")
                        else:
                            print(f"  Computed sea_level stats from boundary nodes only (method={method})")

                    # Normalize ONLY boundary nodes, keep zeros for comp and non_bc_ghost nodes
                    sea_level_normalized = raw_ts[var].copy()  # Start with copy (includes zeros)
                    sea_level_boundary = raw_ts[var][:, boundary_mask]  # Extract boundary values [T, n_boundary_nodes]

                    if var_stats["method"] == "minmax":
                        sea_level_boundary_norm, _ = min_max_normalize(sea_level_boundary,
                                                                       min_val=var_stats["min"],
                                                                       max_val=var_stats["max"])
                    elif var_stats["method"] == "zscore":
                        sea_level_boundary_norm, _ = zscore_normalize(sea_level_boundary,
                                                                      mean=var_stats["mean"],
                                                                      std=var_stats["std"])
                    elif var_stats["method"] == "log1p_zscore":
                        sea_level_boundary_norm, _ = log1p_zscore_normalize(sea_level_boundary,
                                                                             log1p_mean=var_stats["log1p_mean"],
                                                                             log1p_std=var_stats["log1p_std"])

                    sea_level_normalized[:, boundary_mask] = sea_level_boundary_norm  # Replace only boundary values
                    raw_ts[var] = sea_level_normalized

                else:
                    # Normalize all other variables normally (all nodes)
                    if var in all_stats and all_stats[var].get("method") == method:
                        var_stats = all_stats[var]
                    else:
                        # Compute new stats
                        if method == "minmax":
                            _, var_stats = min_max_normalize(raw_ts[var])
                        elif method == "zscore":
                            _, var_stats = zscore_normalize(raw_ts[var])
                        elif method == "log1p_zscore":
                            _, var_stats = log1p_zscore_normalize(raw_ts[var])
                        else:
                            print(f"  WARNING: Unknown method '{method}' for {var}. Using default '{default_method}'.")
                            method = default_method
                            if method == "minmax":
                                _, var_stats = min_max_normalize(raw_ts[var])
                            elif method == "zscore":
                                _, var_stats = zscore_normalize(raw_ts[var])
                            elif method == "log1p_zscore":
                                _, var_stats = log1p_zscore_normalize(raw_ts[var])

                        all_stats[var] = var_stats

                    # Apply normalization using saved stats
                    var_stats = all_stats[var]
                    if var_stats["method"] == "minmax":
                        raw_ts[var], _ = min_max_normalize(raw_ts[var],
                                                            min_val=var_stats["min"],
                                                            max_val=var_stats["max"])
                    elif var_stats["method"] == "zscore":
                        raw_ts[var], _ = zscore_normalize(raw_ts[var],
                                                          mean=var_stats["mean"],
                                                          std=var_stats["std"])
                    elif var_stats["method"] == "log1p_zscore":
                        raw_ts[var], _ = log1p_zscore_normalize(raw_ts[var],
                                                                log1p_mean=var_stats["log1p_mean"],
                                                                log1p_std=var_stats["log1p_std"])

        snapshots = create_graph_snapshots(
            final_nodes_static_df, edges_df, final_edges_static_df, raw_ts,
            cfg, node_type_tensor, edge_type_tensor
        )

        snapshot_dir = os.path.join(output_dir, "graph_snapshots")
        save_snapshots(snapshots, event_name, snapshot_dir)

        if 'example_printed' not in locals():
            print_example_snapshot(snapshots)
            example_printed = True

    if norm_enabled:
        save_stats_file(all_stats, stats_path)

    print("\n--- All events processed successfully! ---")

    # Validate saved graph snapshots.
    # Run the integrity check on the files we just created
    check_snapshots_for_nan(cfg)
    # -----------------------------------------

    # =========================================================
    # Interactive node and edge inspection
    # =========================================================
    # This loop uses the data from the VERY LAST processed event for inspection.
    print("\n--- Starting Interactive Feature Inspector ---")
    print("Enter 'node <id>', 'face <id>', or 'exit' to quit.")

    while True:
        try:
            user_input = input("> ").strip().lower()
            if user_input == 'exit':
                break

            parts = user_input.split()
            if len(parts) != 2:
                print("Invalid format. Use 'node 123' or 'face 456'.")
                continue

            entity_type, entity_id_str = parts
            entity_id = int(entity_id_str)

            if entity_type == 'node':
                print_node_features_by_id(
                    node_id=entity_id,
                    nodes_df=nodes_df,
                    nodes_static_df=final_nodes_static_df,
                    raw_ts=raw_ts,
                    cfg=cfg,
                    index_map=index_map
                )
            elif entity_type == 'face':
                print_edge_features_by_id(
                    face_id=entity_id,
                    edges_df=edges_df,
                    edges_static_df=final_edges_static_df,
                    nodes_df=nodes_df,
                    raw_ts=raw_ts,
                    cfg=cfg,
                    index_map=index_map
                )
            else:
                print(f"Unknown entity type '{entity_type}'. Use 'node' or 'face'.")

        except ValueError:
            print("Invalid ID. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
