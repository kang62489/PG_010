"""
Paint each zone's blob footprints and render the final colored zone map.

Every blob's exact footprint was already found once, in 01_detect_blobs.py,
and its zone id was assigned in 02_zone_groups.py -- this step does no
detection work at all, it just looks up each blob's stored pixel coordinates
and paints them in its zone's color.

Pipeline:
  1. Load zone_groups.csv (every blob, zone-assigned) + blob_footprints.npz
     (each blob's exact pixel coordinates)
  2. Paint each blob's footprint onto a full-size zone label map
  3. Render the painted zones over the stack's mean projection, save the
     picture and a per-zone table of which frames belong to which zone

Requires: zone_groups.csv, blob_footprints.npz (from 02_zone_groups.py / 01_detect_blobs.py), dff_utils.py
Run: uv run python 03_zone_map.py [stack.tif] [output_suffix]
Outputs: zone_map.png, zone_map_events.csv
"""

import sys

import numpy as np
import pandas as pd
from skimage.color import label2rgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dff_utils import load_dff

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 03_zone_map.py [stack.tif] [output_suffix]
#    Suffix must match the one used for 02_zone_groups.py's outputs.
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "2025_12_15-0012_BIEXP_GAUSS.tif"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""

ZONE_GROUPS_CSV = f"zone_groups{SUFFIX}.csv"
BLOB_FOOTPRINTS_NPZ = f"blob_footprints{SUFFIX}.npz"

ZONE_MAP_PNG = f"zone_map{SUFFIX}.png"
ZONE_MAP_EVENTS_CSV = f"zone_map_events{SUFFIX}.csv"


# ---------------------------------------------------------------------------
# 2. Paint each blob's already-known footprint onto a full-size zone label map
# ---------------------------------------------------------------------------

def build_zone_label_map(blobs: pd.DataFrame, footprints_path: str, H: int, W: int) -> np.ndarray:
    """Full-size (H, W) int array: 0 = background, otherwise the zone id of
    whichever blob's footprint covers that pixel."""
    zone_label_map = np.zeros((H, W), dtype=np.int32)
    with np.load(footprints_path) as footprints:
        for i, row in enumerate(blobs.itertuples()):
            coords = footprints[f"blob_{i}"]
            zone_label_map[coords[:, 0], coords[:, 1]] = row.zone
    return zone_label_map


# ---------------------------------------------------------------------------
# 3. Render the colored zone map
# ---------------------------------------------------------------------------

def render_zone_map(zone_label_map: np.ndarray, blobs: pd.DataFrame,
                     background: np.ndarray, out_path: str) -> None:
    bg_norm = (background - background.min()) / (background.max() - background.min())
    overlay = label2rgb(zone_label_map, image=bg_norm, bg_label=0, alpha=0.5,
                         colors=plt.cm.tab20.colors)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(overlay)
    for zone_id, group in blobs.groupby("zone"):
        cy, cx = group.y.mean(), group.x.mean()
        ax.text(cx, cy, str(zone_id), color="white", fontsize=11, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="circle", fc="black", alpha=0.6))
    ax.set_title(f"{blobs['zone'].nunique()} spatial zones (cross-correlation grouping)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)


# ---------------------------------------------------------------------------
# 4. Run the pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    stack, dff = load_dff(STACK_PATH)
    H, W = dff.shape[1], dff.shape[2]

    blobs = pd.read_csv(ZONE_GROUPS_CSV)
    zone_label_map = build_zone_label_map(blobs, BLOB_FOOTPRINTS_NPZ, H, W)

    background = stack.mean(axis=0)
    render_zone_map(zone_label_map, blobs, background, ZONE_MAP_PNG)
    print(f"saved {ZONE_MAP_PNG}")

    blobs.to_csv(ZONE_MAP_EVENTS_CSV, index=False)
    for zone_id, group in blobs.groupby("zone"):
        frames = " ".join(str(int(f)) for f in sorted(group["frame"]))
        print(f"Zone {zone_id} (n={len(group)}): {frames}")


if __name__ == "__main__":
    main()
