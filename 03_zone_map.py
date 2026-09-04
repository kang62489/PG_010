"""
Paint each zone's hotspot footprints and render per-zone + combined zone maps.
Does no detection work -- just looks up stored footprints/zone assignments.

Pipeline:
  1. Load detections_raw.csv + hotspot_footprints.npz (from 02_zone_groups.py)
  2. Load corr_groups.xlsx's 3 sheets (groups, centroid_groups, resting) and
     assign every joint_label a single sequential zone id across all 3 sheets
  3. Union each zone's member joint_labels' footprints (across every frame
     they were detected in) into its own full-size boolean mask
  4. Render one png per zone (that zone's thick contour outline over the
     background projection), plus one final combined png (all zones as
     translucent filled areas over the background projection)

Requires: detections_raw_{stack_stem}.csv, hotspot_footprints_{stack_stem}.npz,
  corr_groups_{stack_stem}.xlsx (from 02_zone_groups.py, for the same stack)
Run: uv run python 03_zone_map.py [stack.tif] [background_stack.tif] [mean|max] [gray|red|green|blue]
Outputs: results/zone_maps_{stack_stem}_{projection}proj/  (one png per zone + one combined png)
"""

import ast
import glob
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from skimage.color import label2rgb
from skimage.measure import find_contours

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 03_zone_map.py [stack.tif] [background_stack.tif] [mean|max] [gray|red|green|blue]
#    stack.tif must be the same stack passed to 01/02 for this run -- its
#    filename stem is the lookup suffix for their outputs, and it sets the
#    zone masks' H/W. background_stack.tif is a separate stack (defaults to
#    stack.tif) that gets mean/max-projected and tinted for the rendered
#    picture -- it does not need to be the same stack used for detection.
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "proc_tiffs/2025_12_15-0012_BIEXP_ALS.tif"
BACKGROUND_STACK_PATH = sys.argv[2] if len(sys.argv) > 2 else STACK_PATH
PROJECTION = sys.argv[3] if len(sys.argv) > 3 else "mean"  # "mean" or "max"
BG_COLOR = sys.argv[4] if len(sys.argv) > 4 else "gray"  # "gray", "red", "green", or "blue"

STACK_STEM = STACK_PATH.rsplit("/", 1)[-1].rsplit(".", 1)[0]
SUFFIX = f"_{STACK_STEM}"

RAW_DETECTIONS_CSV = f"results/detections_raw{SUFFIX}.csv"
HOTSPOT_FOOTPRINTS_NPZ = f"results/hotspot_footprints{SUFFIX}.npz"
CORR_GROUPS_XLSX = f"results/corr_groups{SUFFIX}.xlsx"
ZONE_CONTOURS_NPZ = f"results/zone_contours{SUFFIX}.npz"
ZONE_FOOTPRINTS_NPZ = f"results/zone_footprints{SUFFIX}.npz"

ZONE_MAP_DIR = f"results/zone_maps{SUFFIX}_{PROJECTION}proj"


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

    groups["origin"] = [f"group {i}" for i in range(len(groups))]
    centroid_groups["origin"] = [f"centroid_group {i}" for i in range(len(centroid_groups))]
    resting["origin"] = [f"resting row {i}" for i in range(len(resting))]

    zones = pd.concat([
        groups[["joint_labels", "origin"]].assign(sheet="groups"),
        centroid_groups[["joint_labels", "origin"]].assign(sheet="centroid_groups"),
        resting[["joint_labels", "origin"]].assign(sheet="resting"),
    ], ignore_index=True)
    zones["zone"] = range(1, len(zones) + 1)  # 1-based, 0 stays background
    return zones


# ---------------------------------------------------------------------------
# 3. Union each zone's member footprints into its own boolean mask
# ---------------------------------------------------------------------------

def build_zone_masks(zones: pd.DataFrame, detections: pd.DataFrame, footprints_path: str,
                      H: int, W: int) -> dict[int, np.ndarray]:
    """zone id -> full-size (H, W) boolean mask, True where any member
    footprint (across every frame) covers that pixel. Kept per-zone (instead
    of one shared int label map) so overlapping zones don't overwrite each
    other's pixels."""
    joint_label_to_zone = {l: row.zone for row in zones.itertuples() for l in row.joint_labels}
    zone_masks: dict[int, np.ndarray] = {}

    with np.load(footprints_path) as footprints:
        for i, row in enumerate(detections.itertuples()):
            zone = joint_label_to_zone.get(row.joint_label)
            if zone is None:
                continue
            coords = footprints[f"hotspot_{i}"]
            mask = zone_masks.setdefault(zone, np.zeros((H, W), dtype=bool))
            mask[coords[:, 0], coords[:, 1]] = True
    return zone_masks


# ---------------------------------------------------------------------------
# 4. Render helpers
# ---------------------------------------------------------------------------

def tint_background(background: np.ndarray, color: str) -> np.ndarray:
    """Normalize to [0, 1] and, unless gray, place into a single RGB channel
    so overlays blend the zone colors over a solid-hue image instead of gray."""
    bg_norm = (background - background.min()) / (background.max() - background.min())
    if color == "gray":
        return bg_norm

    channel = {"red": 0, "green": 1, "blue": 2}[color]
    bg_rgb = np.zeros((*bg_norm.shape, 3), dtype=bg_norm.dtype)
    bg_rgb[..., channel] = bg_norm
    return bg_rgb


def save_zone_footprints(zone_masks: dict[int, np.ndarray], out_path: str) -> None:
    """Save each zone's full unioned footprint as an (N, 2) array of (row,
    col) pixel coordinates, one key per zone: zone{id}."""
    footprints = {f"zone{zone_id}": np.argwhere(mask) for zone_id, mask in zone_masks.items()}
    np.savez(out_path, **footprints)


def save_zone_contours(zone_masks: dict[int, np.ndarray], out_path: str) -> None:
    """Save each zone's boundary line(s) as (row, col) float coordinate arrays.
    A zone can be split into disconnected pieces, so each piece is stored
    under its own key: zone{id}_part{k}."""
    contours = {}
    for zone_id, mask in zone_masks.items():
        for i, contour in enumerate(find_contours(mask.astype(float), level=0.5)):
            contours[f"zone{zone_id}_part{i}"] = contour
    np.savez(out_path, **contours)


def write_zone_sizes(zones: pd.DataFrame, zone_masks: dict[int, np.ndarray],
                      xlsx_path: str) -> None:
    """Append a 4th sheet to corr_groups.xlsx with each zone's pixel area."""
    sizes = zones[["zone", "origin", "joint_labels"]].copy()
    sizes["n_members"] = sizes["joint_labels"].apply(len)
    sizes["pixel_area"] = sizes["zone"].map(lambda z: int(zone_masks[z].sum()) if z in zone_masks else 0)
    sizes = sizes.drop(columns="joint_labels").sort_values("zone")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        sizes.to_excel(writer, sheet_name="zone_sizes", index=False)


def zone_centroid(zones: pd.DataFrame, detections: pd.DataFrame, zone_id: int):
    """(y, x) centroid of a zone's member joint_labels, or None if no detections."""
    centroids = detections.groupby("joint_label")[["centroid_y", "centroid_x"]].mean()
    joint_labels = zones.loc[zones["zone"] == zone_id, "joint_labels"].iloc[0]
    member_centroids = centroids.loc[centroids.index.intersection(joint_labels)]
    if member_centroids.empty:
        return None
    return member_centroids["centroid_y"].mean(), member_centroids["centroid_x"].mean()


def render_single_zone(zone_id: int, mask: np.ndarray, color, centroid, bg_tinted: np.ndarray,
                        bg_color: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(bg_tinted, cmap="gray" if bg_color == "gray" else None)
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=2.5)
    if centroid is not None:
        cy, cx = centroid
        ax.text(cx, cy, str(zone_id), color="white", fontsize=9, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="circle", fc="black", alpha=0.6))
    ax.set_title(f"zone {zone_id} (on {STACK_STEM}, {PROJECTION} proj)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def render_zone_overlay(zone_masks: dict[int, np.ndarray], zones: pd.DataFrame,
                         detections: pd.DataFrame, bg_tinted: np.ndarray, bg_color: str,
                         H: int, W: int, out_path: str) -> None:
    """All zones together as translucent filled areas (original overlay style).
    Painted largest-area zone first (bottom layer) down to smallest-area zone
    last (top layer), so a small zone overlapping a large one stays visible
    instead of being covered by it."""
    zone_label_map = np.zeros((H, W), dtype=np.int32)
    paint_order = sorted(zone_masks, key=lambda zone_id: zone_masks[zone_id].sum(), reverse=True)
    for zone_id in paint_order:
        zone_label_map[zone_masks[zone_id]] = zone_id

    overlay = label2rgb(zone_label_map, image=bg_tinted, bg_label=0, alpha=0.5,
                         colors=plt.cm.tab20.colors, saturation=1)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(overlay)
    for zone_id in sorted(zone_masks):
        centroid = zone_centroid(zones, detections, zone_id)
        if centroid is None:
            continue
        cy, cx = centroid
        ax.text(cx, cy, str(zone_id), color="white", fontsize=9, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="circle", fc="black", alpha=0.6))

    n_groups = (zones["sheet"] == "groups").sum()
    n_centroid_groups = (zones["sheet"] == "centroid_groups").sum()
    n_resting = (zones["sheet"] == "resting").sum()
    ax.set_title(f"{len(zone_masks)} zones -- {n_groups} trace-corr, "
                 f"{n_centroid_groups} centroid, {n_resting} resting "
                 f"(on {STACK_STEM}, {PROJECTION} proj)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Run the pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    stack = tifffile.imread(STACK_PATH)
    H, W = stack.shape[1], stack.shape[2]

    detections = pd.read_csv(RAW_DETECTIONS_CSV)
    zones = load_zone_assignments(CORR_GROUPS_XLSX)
    zone_masks = build_zone_masks(zones, detections, HOTSPOT_FOOTPRINTS_NPZ, H, W)

    background_stack = stack if BACKGROUND_STACK_PATH == STACK_PATH else tifffile.imread(BACKGROUND_STACK_PATH)
    background = background_stack.max(axis=0) if PROJECTION == "max" else background_stack.mean(axis=0)
    bg_tinted = tint_background(background, BG_COLOR)

    write_zone_sizes(zones, zone_masks, CORR_GROUPS_XLSX)
    save_zone_footprints(zone_masks, ZONE_FOOTPRINTS_NPZ)
    save_zone_contours(zone_masks, ZONE_CONTOURS_NPZ)
    os.makedirs(ZONE_MAP_DIR, exist_ok=True)
    for stale_png in glob.glob(f"{ZONE_MAP_DIR}/*.png"):
        os.remove(stale_png)

    present_zones = sorted(zone_masks)
    n_zones = len(present_zones)
    width = len(str(n_zones + 1))
    zone_color = {z: plt.cm.tab20.colors[i % len(plt.cm.tab20.colors)]
                  for i, z in enumerate(present_zones)}

    overlay_path = f"{ZONE_MAP_DIR}/{1:0{width}d}_all_zones.png"
    render_zone_overlay(zone_masks, zones, detections, bg_tinted, BG_COLOR, H, W, overlay_path)

    for i, zone_id in enumerate(present_zones, start=2):
        centroid = zone_centroid(zones, detections, zone_id)
        out_path = f"{ZONE_MAP_DIR}/{i:0{width}d}_zone{zone_id}.png"
        render_single_zone(zone_id, zone_masks[zone_id], zone_color[zone_id], centroid,
                            bg_tinted, BG_COLOR, out_path)

    print(f"{n_zones} zones ({(zones['sheet'] == 'groups').sum()} trace-corr, "
          f"{(zones['sheet'] == 'centroid_groups').sum()} centroid, "
          f"{(zones['sheet'] == 'resting').sum()} resting) -> "
          f"{n_zones + 1} pngs in {ZONE_MAP_DIR}/")


if __name__ == "__main__":
    main()
