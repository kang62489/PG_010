"""
Paint each zone's hotspot footprints and render the final colored zone map.
Does no detection work -- just looks up stored footprints/zone assignments.

Pipeline:
  1. Load detections_raw.csv + hotspot_footprints.npz (from 02_zone_groups.py)
  2. Load corr_groups.xlsx's 3 sheets (groups, centroid_groups, resting) and
     assign every joint_label a single sequential zone id across all 3 sheets
  3. Union each zone's member joint_labels' footprints (across every frame
     they were detected in) onto a full-size zone label map
  4. Render over a background stack's mean projection; save as a png

Requires: detections_raw.csv, hotspot_footprints.npz, corr_groups.xlsx (from 02_zone_groups.py)
Run: uv run python 03_zone_map.py [background_stack.tif] [output_suffix] [mean|max] [gray|red|green|blue]
Outputs: zone_map.png
"""

import ast
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
#    Usage: uv run python 03_zone_map.py [background_stack.tif] [output_suffix]
#    Suffix must match the one used for 02_zone_groups.py's outputs.
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "raw_tiffs/2025_12_15-0003.tif"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else "_ALS"
PROJECTION = sys.argv[3] if len(sys.argv) > 3 else "mean"  # "mean" or "max"
BG_COLOR = sys.argv[4] if len(sys.argv) > 4 else "gray"  # "gray", "red", "green", or "blue"

RAW_DETECTIONS_CSV = f"results/detections_raw{SUFFIX}.csv"
HOTSPOT_FOOTPRINTS_NPZ = f"results/hotspot_footprints{SUFFIX}.npz"
CORR_GROUPS_XLSX = f"results/corr_groups{SUFFIX}.xlsx"

STACK_STEM = STACK_PATH.rsplit("/", 1)[-1].rsplit(".", 1)[0]
ZONE_MAP_PNG = f"results/zone_map{SUFFIX}_{STACK_STEM}_{PROJECTION}proj.png"


# ---------------------------------------------------------------------------
# 2. Load zone assignments -- every joint_label gets one sequential zone id
#    across all 3 sheets (groups, then centroid_groups, then resting)
# ---------------------------------------------------------------------------

def load_zone_assignments(xlsx_path: str) -> pd.DataFrame:
    """One row per zone: zone id, sheet name, and its member joint_labels."""
    xl = pd.ExcelFile(xlsx_path)

    groups = xl.parse("groups")
    groups["joint_labels"] = groups["labels"].apply(ast.literal_eval)

    centroid_groups = xl.parse("centroid_groups")
    centroid_groups["joint_labels"] = centroid_groups["labels"].apply(ast.literal_eval)

    resting = xl.parse("resting")
    resting["joint_labels"] = resting["joint_label"].apply(lambda l: [l])

    zones = pd.concat([
        groups[["joint_labels"]].assign(sheet="groups"),
        centroid_groups[["joint_labels"]].assign(sheet="centroid_groups"),
        resting[["joint_labels"]].assign(sheet="resting"),
    ], ignore_index=True)
    zones["zone"] = range(1, len(zones) + 1)  # 1-based, 0 stays background
    return zones


# ---------------------------------------------------------------------------
# 3. Paint each zone's member footprints onto a full-size zone label map
# ---------------------------------------------------------------------------

def build_zone_label_map(zones: pd.DataFrame, detections: pd.DataFrame, footprints_path: str,
                          H: int, W: int) -> np.ndarray:
    """Full-size (H, W) int array: 0 = background, otherwise the zone id of
    whichever zone's member footprint(s) cover that pixel."""
    zone_label_map = np.zeros((H, W), dtype=np.int32)
    joint_label_to_zone = {l: row.zone for row in zones.itertuples() for l in row.joint_labels}

    with np.load(footprints_path) as footprints:
        for i, row in enumerate(detections.itertuples()):
            zone = joint_label_to_zone.get(row.joint_label)
            if zone is None:
                continue
            coords = footprints[f"hotspot_{i}"]
            zone_label_map[coords[:, 0], coords[:, 1]] = zone
    return zone_label_map


# ---------------------------------------------------------------------------
# 4. Render the colored zone map
# ---------------------------------------------------------------------------

def tint_background(background: np.ndarray, color: str) -> np.ndarray:
    """Normalize to [0, 1] and, unless gray, place into a single RGB channel
    so label2rgb blends the zone colors over a solid-hue image instead of gray."""
    bg_norm = (background - background.min()) / (background.max() - background.min())
    if color == "gray":
        return bg_norm

    channel = {"red": 0, "green": 1, "blue": 2}[color]
    bg_rgb = np.zeros((*bg_norm.shape, 3), dtype=bg_norm.dtype)
    bg_rgb[..., channel] = bg_norm
    return bg_rgb


def render_zone_map(zone_label_map: np.ndarray, zones: pd.DataFrame, detections: pd.DataFrame,
                     background: np.ndarray, bg_color: str, out_path: str) -> None:
    bg_tinted = tint_background(background, bg_color)
    overlay = label2rgb(zone_label_map, image=bg_tinted, bg_label=0, alpha=0.5,
                         colors=plt.cm.tab20.colors, saturation=1)

    centroids = detections.groupby("joint_label")[["centroid_y", "centroid_x"]].mean()
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(overlay)
    for row in zones.itertuples():
        member_centroids = centroids.loc[centroids.index.intersection(row.joint_labels)]
        if member_centroids.empty:
            continue
        cy, cx = member_centroids["centroid_y"].mean(), member_centroids["centroid_x"].mean()
        ax.text(cx, cy, str(row.zone), color="white", fontsize=9, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="circle", fc="black", alpha=0.6))

    n_groups = (zones["sheet"] == "groups").sum()
    n_centroid_groups = (zones["sheet"] == "centroid_groups").sum()
    n_resting = (zones["sheet"] == "resting").sum()
    ax.set_title(f"{len(zones)} zones -- {n_groups} trace-corr, "
                 f"{n_centroid_groups} centroid, {n_resting} resting "
                 f"(on {STACK_STEM}, {PROJECTION} proj)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)


# ---------------------------------------------------------------------------
# 5. Run the pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    stack = tifffile.imread(STACK_PATH)
    H, W = stack.shape[1], stack.shape[2]

    detections = pd.read_csv(RAW_DETECTIONS_CSV)
    zones = load_zone_assignments(CORR_GROUPS_XLSX)
    zone_label_map = build_zone_label_map(zones, detections, HOTSPOT_FOOTPRINTS_NPZ, H, W)

    background = stack.max(axis=0) if PROJECTION == "max" else stack.mean(axis=0)
    render_zone_map(zone_label_map, zones, detections, background, BG_COLOR, ZONE_MAP_PNG)
    print(f"{len(zones)} zones ({(zones['sheet'] == 'groups').sum()} trace-corr, "
          f"{(zones['sheet'] == 'centroid_groups').sum()} centroid, "
          f"{(zones['sheet'] == 'resting').sum()} resting) -> {ZONE_MAP_PNG}")


if __name__ == "__main__":
    main()
