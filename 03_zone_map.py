"""
Paint each zone's hotspot footprints and render the final colored zone map.
Does no detection work -- just looks up stored footprints/zone ids.

Pipeline:
  1. Load zone_groups.csv + hotspot_footprints.npz
  2. Paint each footprint onto a full-size zone label map
  3. Render over the stack's mean projection; save the picture + a
     per-zone table of which frames belong to which zone

Requires: zone_groups.csv, hotspot_footprints.npz (from 02_zone_groups.py / 01_detect_hotspots.py)
Run: uv run python 03_zone_map.py [stack.tif] [output_suffix]
Outputs: zone_map.png, zone_map_events.csv
"""

import sys

import numpy as np
import pandas as pd
import tifffile
from skimage.color import label2rgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 03_zone_map.py [stack.tif] [output_suffix]
#    Suffix must match the one used for 02_zone_groups.py's outputs.
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "2025_12_15-0012_BIEXP_GAUSS.tif"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""

ZONE_GROUPS_CSV = f"zone_groups{SUFFIX}.csv"
HOTSPOT_FOOTPRINTS_NPZ = f"hotspot_footprints{SUFFIX}.npz"

ZONE_MAP_PNG = f"zone_map{SUFFIX}.png"
ZONE_MAP_EVENTS_CSV = f"zone_map_events{SUFFIX}.csv"


# ---------------------------------------------------------------------------
# 2. Load the stack (already deltaF/F0 -- no baseline subtraction here)
# ---------------------------------------------------------------------------

def load_stack(stack_path: str) -> tuple[np.ndarray, np.ndarray]:
    stack = tifffile.imread(stack_path)
    stack_f16 = stack.astype(np.float16)
    return stack, stack_f16


# ---------------------------------------------------------------------------
# 3. Paint each hotspot's already-known footprint onto a full-size zone label map
# ---------------------------------------------------------------------------

def build_zone_label_map(hotspots: pd.DataFrame, footprints_path: str, H: int, W: int) -> np.ndarray:
    """Full-size (H, W) int array: 0 = background, otherwise the zone id of
    whichever hotspot's footprint covers that pixel."""
    zone_label_map = np.zeros((H, W), dtype=np.int32)
    with np.load(footprints_path) as footprints:
        for i, row in enumerate(hotspots.itertuples()):
            coords = footprints[f"hotspot_{i}"]
            zone_label_map[coords[:, 0], coords[:, 1]] = row.zone
    return zone_label_map


# ---------------------------------------------------------------------------
# 4. Render the colored zone map
# ---------------------------------------------------------------------------

def render_zone_map(zone_label_map: np.ndarray, hotspots: pd.DataFrame,
                     background: np.ndarray, out_path: str) -> None:
    bg_norm = (background - background.min()) / (background.max() - background.min())
    overlay = label2rgb(zone_label_map, image=bg_norm, bg_label=0, alpha=0.5,
                         colors=plt.cm.tab20.colors)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(overlay)
    for zone_id, group in hotspots.groupby("zone"):
        cy, cx = group.y.mean(), group.x.mean()
        ax.text(cx, cy, str(zone_id), color="white", fontsize=11, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="circle", fc="black", alpha=0.6))
    ax.set_title(f"{hotspots['zone'].nunique()} spatial zones (cross-correlation grouping)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)


# ---------------------------------------------------------------------------
# 5. Run the pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    stack, stack_f16 = load_stack(STACK_PATH)
    H, W = stack_f16.shape[1], stack_f16.shape[2]

    hotspots = pd.read_csv(ZONE_GROUPS_CSV)
    zone_label_map = build_zone_label_map(hotspots, HOTSPOT_FOOTPRINTS_NPZ, H, W)

    background = stack.mean(axis=0)
    render_zone_map(zone_label_map, hotspots, background, ZONE_MAP_PNG)
    print(f"saved {ZONE_MAP_PNG}")

    hotspots.to_csv(ZONE_MAP_EVENTS_CSV, index=False)
    for zone_id, group in hotspots.groupby("zone"):
        frames = " ".join(str(int(f)) for f in sorted(group["frame"]))
        print(f"Zone {zone_id} (n={len(group)}): {frames}")


if __name__ == "__main__":
    main()
