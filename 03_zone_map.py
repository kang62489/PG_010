"""
Merge each zone's blob footprints and render the final colored zone map.

Pipeline:
  1. Load zone_groups.csv (every blob detection, already zone-assigned by 02)
  2. Re-detect each blob's real footprint at its own frame (not just its
     centroid point) and paint it with its zone's color -- this is the one
     step that needs the raw stack again, so it runs last, once
  3. Render the painted zones over the stack's mean projection, save the
     picture and a per-zone table of which frames belong to which zone

Requires: zone_groups.csv (from 02_zone_groups.py), dff_utils.py
Run: uv run python 03_zone_map.py [stack.tif] [output_suffix]
Outputs: zone_map.png, zone_map_events.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.color import label2rgb
from skimage.measure import label, regionprops

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

THRESHOLD_PERCENTILE = 99.5  # must match 01_detect_blobs.py so footprints reproduce
MIN_FOOTPRINT_AREA = 1000    # ignore a re-detected blob if it's implausibly small

ZONE_MAP_PNG = f"zone_map{SUFFIX}.png"
ZONE_MAP_EVENTS_CSV = f"zone_map_events{SUFFIX}.csv"


# ---------------------------------------------------------------------------
# 2. Paint each blob's real footprint onto a full-size zone label map
# ---------------------------------------------------------------------------

def find_blob_footprint(frame: np.ndarray, threshold: float, near_yx: tuple[float, float]):
    """Re-detect blobs in one frame and return the pixel coords of whichever
    blob is closest to `near_yx` (the detection's recorded centroid)."""
    mask = frame > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    regions = regionprops(label(mask))
    if not regions:
        return None
    target = np.array(near_yx)
    best = min(regions, key=lambda r: np.hypot(*(np.array(r.centroid) - target)))
    if best.area < MIN_FOOTPRINT_AREA:
        return None
    return best.coords  # (N, 2) array of (y, x)


def build_zone_label_map(blobs: pd.DataFrame, dff: np.ndarray,
                          threshold: float) -> np.ndarray:
    """Full-size (H, W) int array: 0 = background, otherwise the zone id of
    whichever blob's footprint covers that pixel."""
    H, W = dff.shape[1], dff.shape[2]
    zone_label_map = np.zeros((H, W), dtype=np.int32)

    for row in blobs.itertuples():
        coords = find_blob_footprint(dff[int(row.frame)], threshold, (row.y, row.x))
        if coords is not None:
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
    threshold = np.percentile(dff, THRESHOLD_PERCENTILE)

    blobs = pd.read_csv(ZONE_GROUPS_CSV)
    zone_label_map = build_zone_label_map(blobs, dff, threshold)

    background = stack.mean(axis=0)
    render_zone_map(zone_label_map, blobs, background, ZONE_MAP_PNG)
    print(f"saved {ZONE_MAP_PNG}")

    blobs.to_csv(ZONE_MAP_EVENTS_CSV, index=False)
    for zone_id, group in blobs.groupby("zone"):
        frames = " ".join(str(int(f)) for f in sorted(group["frame"]))
        print(f"Zone {zone_id} (n={len(group)}): {frames}")


if __name__ == "__main__":
    main()
