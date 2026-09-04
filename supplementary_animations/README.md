# Reviewer Supplementary Animations

This directory records the supplemental rollout animations prepared for manuscript review. The GIF files are stored outside the repository for review because each animation is approximately 45-56 MB.

## Animation Set

| File | Purpose | Wet-depth threshold | Status |
| --- | --- | --- | --- |
| `single_boundary_all_models_BC1_S402_R402_WDge10cm.gif` | Single active boundary example comparing M1, M2, M3, M4 (GNN4CF), and the single-processor model | 0.10 m | Prepared locally; upload link pending |
| `single_boundary_all_models_BC1_S402_R402_WDge20cm.gif` | Same single-boundary event using the 0.20 m wet-depth threshold | 0.20 m | Prepared locally; upload link pending |
| `multiboundary_M4_S476_R476_WDge10cm.gif` | Unseen simultaneous multi-boundary forcing example for M4 (BC12, BC13, BC23, and BC123 rows) | 0.10 m | Prepared locally; upload link pending |
| `multiboundary_M4_S476_R476_WDge20cm.gif` | Same multi-boundary event using the 0.20 m wet-depth threshold | 0.20 m | Prepared locally; upload link pending |

## Local Source Folder

The current local copy prepared for Google Drive upload is located at:

`C:\CF_GNN\rollout_transfer_bundle_4_2\paper_final_results\reviewer_supplementary_gifs`

## Reviewer Note

The animations use fixed water-depth and absolute-error color scales across frames. Error maps display positive absolute water-depth error only; zero-error and dry-dry regions are transparent. Frame-wise RMSE, CSI, depth-agreement scatter, and relative L2 values are computed over the threshold-conditioned wet union.