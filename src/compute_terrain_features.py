# -*- coding: utf-8 -*-
"""
Compute and sample external terrain attributes for GNN4CF.

Use WhiteboxTools to derive slope (degrees), aspect, mean curvature, D8 flow
direction, and flow accumulation (cells) from the configured DEM. Sample the
attribute rasters at HEC-RAS computational cell points and write the configured
cell-ID table. Temporary rasters are removed after sampling.

Run with --config configs/config_gnn4cf_final.yml from the repository root.
The DEM and HDF coordinates must use the same projected coordinate system.
"""

import argparse
import os
import sys
import yaml
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
# =========================================================
# 1. HDF HELPER FUNCTIONS
# (Simplified: only loads what is needed for cell points)
# =========================================================

def load_config(path):
    """Loads the YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hdf_roots(cfg):
    """Constructs HDF group paths from the config."""
    area = cfg["hecras"]["area_name"]
    g_root = cfg["hecras"].get("geometry_root", "Geometry/2D Flow Areas")
    g_area = f"{g_root}/{area}"
    return g_root, g_area


def load_geom_from_hdf(hdf_path, cfg):
    """Read cell_points coordinates directly from HEC-RAS HDF."""
    g_root, g_area = hdf_roots(cfg)
    d = {}
    with h5py.File(hdf_path, "r") as f:
        # True computational centers (no ghosts):
        d["cell_points"] = f[f"{g_root}/Cell Points"][:]
    return d


def get_ncells_from_hdf(geom):
    """Get cell count from 'cell_points'."""
    if "cell_points" in geom and geom["cell_points"] is not None:
        return int(geom["cell_points"].shape[0])
    raise ValueError("Could not find 'Cell Points' in HDF file.")


def find_latest_hdf(path_or_dir):
    """Finds the latest HDF file in a directory, preferring plan files."""
    if os.path.isfile(path_or_dir):
        return path_or_dir
    # Prefer plan files (.p*.hdf) as they are results
    cands = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir)
             if f.lower().endswith(".hdf") and ".p" in f.lower()]
    if not cands:
        # Fallback to any .hdf if no plan files
        cands = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir)
                 if f.lower().endswith(".hdf")]
    if not cands:
        raise FileNotFoundError(f"No .hdf files found in {path_or_dir}")
    return max(cands, key=os.path.getmtime)


# =========================================================
# 3. MAIN EXECUTION
# =========================================================
def main(config_path):
    print("--- Starting Terrain Attribute Calculation (WhiteboxTools Point Sample) ---")

    # Import WhiteboxTools lazily so we can show a clean error if missing
    try:
        from whitebox import WhiteboxTools
    except Exception as e:
        print("ERROR: Whitebox Python package is not available.")
        print("Installation:")
        print("  pip install whitebox")
        print('  python -c "import whitebox; whitebox.download_wbt()"  # one-time download of the tool')
        sys.exit(1)

    # Instantiate once
    wbt = WhiteboxTools()
    wbt.verbose = False

    # --- Load Config and Paths ---
    cfg = load_config(config_path)
    dem_path = cfg["paths"]["dem_path"]

    hdf_dir = cfg["paths"]["copy_each_hdf_to"]
    if not os.path.isdir(hdf_dir):
        hdf_dir = cfg["paths"]["results_dir"]
    hdf_path = find_latest_hdf(hdf_dir)

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    external_files = cfg["paths"].get("external_node_features", [])
    if not external_files:
        print("ERROR: No 'external_node_features' listed in the configuration.")
        return

    output_path = external_files[0]['path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Input DEM: {dem_path}")
    print(f"Input HDF: {hdf_path}")
    print(f"Output TXT: {output_path}")

    # --- 1. Load Cell Points from HDF ---
    geom = load_geom_from_hdf(hdf_path, cfg)
    cell_points_xy = geom["cell_points"]
    n_cells = get_ncells_from_hdf(geom)
    print(f"Loaded {n_cells} cell center coordinates.")

    # --- 2. Check DEM CRS and Get Pixel Indices ---
    with rasterio.open(dem_path) as dem_ds:
        # Report coordinate system and units.
        print("\n--- DEM Coordinate System Info ---")
        if dem_ds.crs:
            print(f"  Horizontal CRS: {dem_ds.crs.to_string()}")
            if dem_ds.crs.is_projected:
                print(f"  Horizontal Units: {dem_ds.crs.linear_units}")
            elif dem_ds.crs.is_geographic:
                print(f"  Horizontal Units: {dem_ds.crs.angular_units}")
        else:
            print("  Horizontal CRS: Not specified (None)")

        # Try to get vertical units, default to "unknown"
        try:
            v_units = dem_ds.crs.vertical_units or "unknown"
        except AttributeError:
            v_units = "unknown"  # Fallback if .crs is None or doesn't have vertical_units

        print(f"  Vertical Units: {v_units} (Note: often 'unknown', check DEM metadata)")
        print("------------------------------------\n")


        if dem_ds.crs is None or (hasattr(dem_ds.crs, "is_projected") and not dem_ds.crs.is_projected):
            print("\n" + "=" * 60)
            print("  WARNING: DEM CRS is not projected (or missing).")
            print("  Consider reprojecting to a projected CRS (e.g., UTM) for correct slope/area units.")
            print("=" * 60 + "\n")

        xs = cell_points_xy[:, 0]
        ys = cell_points_xy[:, 1]

        # Convert cell coordinates to raster row/column indices.
        rows, cols = rowcol(
            dem_ds.transform,
            xs.astype(float),
            ys.astype(float),
            op=np.floor  # Select the pixel containing each point.
        )

        # to numpy arrays and clamp to bounds
        row_indices = np.asarray(rows, dtype=np.int64)
        col_indices = np.asarray(cols, dtype=np.int64)
        row_indices = np.clip(row_indices, 0, dem_ds.height - 1)
        col_indices = np.clip(col_indices, 0, dem_ds.width - 1)

    # --- 3. Calculate Terrain Attributes using WhiteboxTools ---
    slope_file     = os.path.join(output_dir, "_temp_slope.tif")
    aspect_file    = os.path.join(output_dir, "_temp_aspect.tif")
    curv_file      = os.path.join(output_dir, "_temp_curvature.tif")
    flow_dir_file  = os.path.join(output_dir, "_temp_flow_dir.tif")
    flow_acc_file  = os.path.join(output_dir, "_temp_flow_acc.tif")
    temp_files = [slope_file, aspect_file, curv_file, flow_dir_file, flow_acc_file]

    results = {"cell_id": np.arange(n_cells)}

    try:
        print("  2/4 Calculating Slope...")
        wbt.slope(dem=dem_path, output=slope_file, zfactor=None, units='degrees')

        print("  2/4 Calculating Aspect...")
        wbt.aspect(dem=dem_path, output=aspect_file)

        print("  2/4 Calculating Curvature (Mean)...")
        wbt.mean_curvature(dem=dem_path, output=curv_file, zfactor=None)

        print("  2/4 Calculating Flow Direction (D8)...")
        wbt.d8_pointer(dem=dem_path, output=flow_dir_file)

        print("  3/4 Calculating Flow Accumulation...")
        wbt.d8_flow_accumulation(i=dem_path, output=flow_acc_file, out_type='cells')

        # --- 4. Sample Attribute Rasters at Cell Points ---
        print("  4/4 Sampling attributes at cell points...")

        def sample_raster_at_points(raster_file, rows, cols):
            if not os.path.exists(raster_file):
                print(f"  WARNING: Temp file {raster_file} not found. Skipping.")
                return np.full(rows.shape, np.nan, dtype=float)
            with rasterio.open(raster_file) as src:
                arr = src.read(1)
                return arr[rows, cols].astype(float)

        results["slope"]             = sample_raster_at_points(slope_file, row_indices, col_indices)
        results["aspect"]            = sample_raster_at_points(aspect_file, row_indices, col_indices)
        results["curvature"]         = sample_raster_at_points(curv_file, row_indices, col_indices)
        results["flow_direction"]    = sample_raster_at_points(flow_dir_file, row_indices, col_indices)
        results["flow_accumulation"] = sample_raster_at_points(flow_acc_file, row_indices, col_indices)

    finally:
        print("Cleaning up temporary raster files...")
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: could not remove temp file {f}: {e}")

    # --- 6. Save to Text File ---
    df = pd.DataFrame(results)
    all_cols = ["cell_id", "slope", "aspect", "flow_direction", "curvature", "flow_accumulation"]
    df = df.reindex(columns=all_cols)

    # Preserve missing sampled values as NaNs in the output table.

    df.to_csv(output_path, sep=' ', index=False, float_format='%.6f')
    print("\n--- Success! ---")
    print(f"Saved {len(df)} cell attributes to: {output_path}")
    print("\n--- Output Preview (first 5 rows) ---")
    print(df.head(5).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "config_gnn4cf_final.yml",
        ),
        help="Path to the shared GNN4CF YAML configuration.",
    )
    args = parser.parse_args()
    # Check for required libraries
    try:
        from whitebox import WhiteboxTools
    except ImportError as e:
        print(f"ERROR: Missing required library: {e.name}")
        print("Please install the dependency:")

        print("pip install whitebox")
        print("Then run: python -c \"import whitebox; whitebox.download_wbt()\"")

        sys.exit(1)

    main(args.config)
