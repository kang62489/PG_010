"""
Hotspot -> zone pipeline for a deltaF/F0 stack.

  Step 1. Detect   : threshold every frame -> cleaned hotspot mask
  Step 2. Group    : per-frame hotspots -> tracks -> correlated / nearby groups
  Step 3. Map      : groups -> zones -> zone masks, outlines, and PNG maps

Run:
  uv run python sp_ach_zones.py [stack.tif] [--sigma 2] [--bg other.tif]
                                [--proj mean|max] [--color gray|red|green|blue]

All outputs go to results/, suffixed with the stack's filename stem.
"""

## Modules
# Standard
import argparse
import glob
import os
import time
from functools import partial
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import tifffile
from rich.console import Console
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from skimage.color import label2rgb
from skimage.measure import find_contours, regionprops

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

console = Console()


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

DEFAULT_STACK = "proc_tiffs/2025_12_15-0012_BIEXP_ALS.tif"
RESULTS_DIR = "results"

# --- Step 1: detect --------------------------------------------------------

HISTOGRAM_BINS = 512  # bins for the pixel-value histogram used to find the background peak

HIST_RANGE_PCT = (0.1, 99.9)  # histogram spans these percentiles, so rare outliers can't widen the bins

CROSSOVER_RATIO = 2   # default threshold = background peak + this many fitted sigmas (--sigma)

TH_SMALL_OBJ = 4000   # px: drop per-frame blobs smaller than this (noise speckle)

CLOSE_RADIUS = 3      # px: morphological closing to fill small notches in a blob's shape

# --- Step 2: group ---------------------------------------------------------

TH_SMALL_HOTSPOTS = 3000       # px: drop merged hotspots smaller than this

CONNECT_RADIUS = 75            # px: merge same-frame fragments whose boundaries are this close

MAX_CENTROID_DEVIATION = 115   # px: max centroid distance for frame-to-frame chaining
                               #     and for the 2nd (centroid) grouping

MIN_GROUP_CORR = 0.95          # 1st (trace) grouping: every pair in a group has r >= this

# --- Step 3: map -----------------------------------------------------------

FRAME_RATE_HZ = 20             # imaging rate, converts frames -> seconds for zone event stats


def output_paths(stem: str, projection: str) -> dict[str, str]:
    """All output file paths for one stack."""
    s = f"_{stem}"
    return {
        "mask_tif":        f"{RESULTS_DIR}/mask{s}.tif",
        "detections_csv":  f"{RESULTS_DIR}/detections_raw{s}.csv",
        "hotspots_npz":    f"{RESULTS_DIR}/hotspot_footprints{s}.npz",
        "groups_xlsx":     f"{RESULTS_DIR}/corr_groups{s}.xlsx",
        "zone_fp_npz":     f"{RESULTS_DIR}/zone_footprints{s}.npz",
        "zone_contour_npz": f"{RESULTS_DIR}/zone_contours{s}.npz",
        "zone_map_dir":    f"{RESULTS_DIR}/zone_maps{s}_{projection}proj",
    }


# ===========================================================================
#
#   STEP 1 -- DETECT: threshold every frame -> cleaned hotspot mask
#
# ===========================================================================

def load_stack(stack_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw stack, float16 copy). Stack is already deltaF/F0."""
    stack = tifffile.imread(stack_path)
    return stack, stack.astype(np.float16)


# --- 1a. Background threshold ----------------------------------------------

def gaussian(x: np.ndarray, amplitude: float, sigma: float, peak_value: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - peak_value) / sigma) ** 2)


def find_background_threshold(stack_f16: np.ndarray, sigma_ratio: float,
                               bins: int = HISTOGRAM_BINS) -> float:
    """Fit a Gaussian to the histogram's left side; return peak + sigma_ratio * sigma."""
    values = stack_f16.ravel().astype(np.float32)
    hist_range = np.percentile(values, HIST_RANGE_PCT)
    hist, bin_edges = np.histogram(values, bins=bins, range=hist_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peak_idx = np.argmax(hist)
    peak_value = bin_centers[peak_idx]
    peak_height = hist[peak_idx]

    # left side only -- bright hotspot pixels sit on the right of the peak
    left_mask = bin_centers <= peak_value
    popt, _ = curve_fit(partial(gaussian, peak_value=peak_value),
                         bin_centers[left_mask], hist[left_mask],
                         p0=[peak_height, 0.0016])
    _, sigma = popt

    return float(peak_value + sigma_ratio * sigma)


# --- 1b. Per-frame mask ----------------------------------------------------

def remove_small_objects(mask: np.ndarray, th_small_obj: int) -> np.ndarray:
    """Drop blobs smaller than th_small_obj px, frame by frame."""
    cleaned = np.zeros_like(mask)
    for t in range(mask.shape[0]):
        labeled, n = ndimage.label(mask[t])
        if n == 0:
            continue
        sizes = ndimage.sum(mask[t], labeled, index=np.arange(1, n + 1))
        keep = np.nonzero(sizes >= th_small_obj)[0] + 1
        cleaned[t] = np.isin(labeled, keep)
    return cleaned


def detect_all_frames(stack_f16: np.ndarray, threshold: float,
                       th_small_obj: int = TH_SMALL_OBJ, close_radius: int = CLOSE_RADIUS) -> np.ndarray:
    """Threshold -> open -> close -> fill holes -> drop small blobs."""
    mask = stack_f16 > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((1, 3, 3)))  # drop lone bright pixels
    mask = ndimage.binary_closing(mask, structure=np.ones((1, close_radius, close_radius)))  # close small notches
    for t in range(mask.shape[0]):
        mask[t] = ndimage.binary_fill_holes(mask[t])
    return remove_small_objects(mask, th_small_obj)


# ===========================================================================
#
#   STEP 2 -- GROUP: per-frame hotspots -> tracks -> groups
#
# ===========================================================================

# --- Union-find helpers ----------------------------------------------------

def _find(parent: dict, x):
    """Union-find root, with path halving."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _union(parent: dict, a, b) -> None:
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a != root_b:
        parent[root_a] = root_b


# --- 2a. Label each frame, merge nearby fragments --------------------------

def merge_adjacent_hotspots(labeled_frame: np.ndarray, connect_radius: float) -> np.ndarray:
    """Merge same-frame labels whose boundaries are within connect_radius."""
    regions = regionprops(labeled_frame)
    if len(regions) <= 1:
        return labeled_frame

    boundaries: dict[int, np.ndarray] = {}
    centroids: dict[int, np.ndarray] = {}
    radii: dict[int, float] = {}
    for region in regions:
        eroded = ndimage.binary_erosion(region.image)
        boundary_local = region.image & ~eroded
        ys, xs = np.nonzero(boundary_local)
        min_row, min_col = region.bbox[0], region.bbox[1]
        pts = np.column_stack([ys + min_row, xs + min_col]).astype(np.float64)

        centroid = np.array(region.centroid)
        boundaries[region.label] = pts
        centroids[region.label] = centroid
        radii[region.label] = float(np.linalg.norm(pts - centroid, axis=1).max()) if len(pts) else 0.0

    labels = [region.label for region in regions]
    parent = {label: label for label in labels}

    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            centroid_a, centroid_b = centroids[label_a], centroids[label_b]
            direction = centroid_b - centroid_a
            centroid_dist = float(np.linalg.norm(direction))

            # prefilter 1: too far apart even in the best case
            if centroid_dist > radii[label_a] + radii[label_b] + connect_radius:
                continue

            # prefilter 2: only the boundary halves facing each other can hold the closest points
            pts_a, pts_b = boundaries[label_a], boundaries[label_b]
            if centroid_dist > 0:
                facing_a = pts_a[(pts_a - centroid_a) @ direction >= 0]
                facing_b = pts_b[(pts_b - centroid_b) @ -direction >= 0]
            else:
                facing_a, facing_b = pts_a, pts_b
            if len(facing_a) == 0 or len(facing_b) == 0:
                continue

            min_dist = float(KDTree(facing_b).query(facing_a, k=1)[0].min())
            if min_dist <= connect_radius:
                _union(parent, label_a, label_b)

    remap = np.zeros(int(labeled_frame.max()) + 1, dtype=np.int32)
    for label in labels:
        remap[label] = _find(parent, label)
    return remap[labeled_frame]


def spatiotemporally_connect_hotspots(mask: np.ndarray, th_small_hotspots: int,
                                      connect_radius: int) -> tuple[pd.DataFrame, list]:
    """Per frame: label, merge fragments, keep big hotspots -> (detections table, footprints)."""
    t_start = time.time()
    console.print("[cyan]spatiotemporally_connect_hotspots:[/cyan] starting...")

    hotspots_props = []
    footprints = []
    label_offset = 0

    for frame_id in range(mask.shape[0]):
        frame_mask = mask[frame_id]
        labeled_frame, n_frame_labels = ndimage.label(frame_mask, structure=np.ones((3, 3)))
        if n_frame_labels > 1:
            labeled_frame = merge_adjacent_hotspots(labeled_frame, connect_radius)

        for joint_label_at_frame_id in np.unique(labeled_frame[frame_mask]):
            footprint_mask = frame_mask & (labeled_frame == joint_label_at_frame_id)
            area = int(footprint_mask.sum())
            if area < th_small_hotspots:
                continue

            coords = np.argwhere(footprint_mask)
            footprints.append(coords)
            hotspots_props.append({
                "frame": frame_id + 1,  # 1-based for the CSV
                "joint_label": int(joint_label_at_frame_id) + label_offset,
                "centroid_y": float(coords[:, 0].mean()),
                "centroid_x": float(coords[:, 1].mean()),
                "area": area,
            })

        label_offset += n_frame_labels

    n_after = pd.DataFrame(hotspots_props)["joint_label"].nunique() if hotspots_props else 0
    console.print(f"hotspots after spatial connecting: [bold]{n_after}[/bold]")
    console.print(f"[green]spatiotemporally_connect_hotspots: done[/green], {len(hotspots_props)} detections "
                  f"[dim]({time.time() - t_start:.1f}s total)[/dim]")
    return pd.DataFrame(hotspots_props), footprints


# --- 2b. Chain frame-adjacent hotspots into tracks -------------------------

def chain_frame_adjacent_centroids(hotspots_props: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    """Link joint_labels in consecutive frames whose centroids are < max_dist apart."""
    rows = hotspots_props.groupby("joint_label").agg(
        frame=("frame", "first"),
        centroid_y=("centroid_y", "mean"),
        centroid_x=("centroid_x", "mean"),
    ).reset_index()

    labels = rows["joint_label"].tolist()
    parent = {label: label for label in labels}

    by_frame = {frame: g for frame, g in rows.groupby("frame")}
    for frame, current in by_frame.items():
        next_frame = by_frame.get(frame + 1)
        if next_frame is None:
            continue
        for _, row_a in current.iterrows():
            for _, row_b in next_frame.iterrows():
                dist = np.hypot(row_a["centroid_y"] - row_b["centroid_y"], row_a["centroid_x"] - row_b["centroid_x"])
                if dist < max_dist:
                    _union(parent, row_a["joint_label"], row_b["joint_label"])

    roots = [_find(parent, label) for label in labels]
    group_id_map = {root: new for new, root in enumerate(pd.unique(np.array(roots)))}
    group_ids = [group_id_map[root] for root in roots]

    return pd.DataFrame({"joint_label": labels, "group": group_ids})


def assign_frame_adjacent_joint_labels(hotspots_props: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    """Give every detection in one chained track the same joint_label."""
    result = chain_frame_adjacent_centroids(hotspots_props, max_dist)
    label_to_group = dict(zip(result["joint_label"], result["group"]))

    hotspots_props = hotspots_props.copy()
    hotspots_props["joint_label"] = hotspots_props["joint_label"].map(label_to_group)
    return hotspots_props


# --- 2c. Per-track deltaF/F0 traces ----------------------------------------

def detection_traces(footprints: list, stack_f16: np.ndarray) -> np.ndarray:
    """(n_detections, n_frames): mean trace over each detection's own footprint."""
    n_frames, _, width = stack_f16.shape
    counts = np.array([len(coords) for coords in footprints])

    all_coords = np.concatenate(footprints, axis=0)
    linear_idx = all_coords[:, 0] * width + all_coords[:, 1]
    group_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    # vectorized gather + reduceat, chunked over frames to cap memory at ~512 MB
    total_px = len(linear_idx)
    frame_chunk_budget = 512 * 1024 * 1024  # bytes
    bytes_per_frame_row = total_px * 4  # float32
    chunk_size = max(1, int(frame_chunk_budget // bytes_per_frame_row))

    stack_flat = stack_f16.reshape(n_frames, -1)
    means = np.empty((n_frames, len(footprints)), dtype=np.float32)
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        gathered = stack_flat[start:end, linear_idx].astype(np.float32)  # float32 to keep sum precision
        sums = np.add.reduceat(gathered, group_starts, axis=1)
        means[start:end] = sums / counts[np.newaxis, :]

    return means.T.astype(np.float16)


def track_mean_traces(hotspots_props: pd.DataFrame, detection_traces: np.ndarray) -> pd.DataFrame:
    """One averaged trace per joint_label."""
    rows = []
    for joint_label, group in hotspots_props.groupby("joint_label"):
        rows.append({"joint_label": joint_label, "trace": detection_traces[group.index].mean(axis=0)})
    return pd.DataFrame(rows)


def track_centroids(hotspots_props: pd.DataFrame) -> pd.DataFrame:
    """One mean (y, x) centroid per joint_label."""
    grouped = hotspots_props.groupby("joint_label")[["centroid_y", "centroid_x"]].mean()
    mean_centroid = list(zip(grouped["centroid_y"], grouped["centroid_x"]))
    return pd.DataFrame({"joint_label": grouped.index, "mean_centroid": mean_centroid})


# --- 2d. Two-stage grouping ------------------------------------------------

def first_grouping(tracks: pd.DataFrame, min_corr: float) -> pd.DataFrame:
    """Complete-linkage clustering on trace correlation, cut at r = min_corr."""
    labels = tracks["joint_label"].tolist()
    if len(labels) == 1:
        return pd.DataFrame({"joint_label": labels, "group": [0]})

    traces = np.stack(tracks["trace"].values)

    corr = np.corrcoef(traces)
    distance = 1 - corr
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2  # force exact symmetry

    linkage_matrix = linkage(squareform(distance, checks=False), method="complete")
    raw_group_ids = fcluster(linkage_matrix, t=1 - min_corr, criterion="distance")

    # renumber 0, 1, 2, ...
    group_id_map = {raw: new for new, raw in enumerate(pd.unique(raw_group_ids))}
    group_ids = [group_id_map[raw] for raw in raw_group_ids]

    return pd.DataFrame({"joint_label": labels, "group": group_ids})


def second_grouping(centroids: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    """Complete-linkage clustering on centroid distance, cut at max_dist."""
    labels = centroids["joint_label"].tolist()
    if len(labels) == 0:
        return pd.DataFrame({"joint_label": [], "group": []})
    if len(labels) == 1:
        return pd.DataFrame({"joint_label": labels, "group": [0]})

    coords = np.stack(centroids["mean_centroid"].values)

    distance = squareform(pdist(coords))

    linkage_matrix = linkage(squareform(distance, checks=False), method="complete")
    raw_group_ids = fcluster(linkage_matrix, t=max_dist, criterion="distance")

    # renumber 0, 1, 2, ...
    group_id_map = {raw: new for new, raw in enumerate(pd.unique(raw_group_ids))}
    group_ids = [group_id_map[raw] for raw in raw_group_ids]

    return pd.DataFrame({"joint_label": labels, "group": group_ids})


def group_tracks(hotspots_props: pd.DataFrame, footprints: list,
                 stack_f16: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trace-corr grouping, then centroid grouping of leftovers -> (groups, centroid_groups, resting)."""
    t_start = time.time()
    console.print("[cyan]group_tracks:[/cyan] starting...")

    # traces
    t_stage = time.time()
    det_traces = detection_traces(footprints, stack_f16)
    tracks = track_mean_traces(hotspots_props, det_traces)
    frames_by_label = hotspots_props.groupby("joint_label")["frame"].apply(lambda s: sorted(s.tolist()))
    console.print(f"[cyan]group_tracks:[/cyan] traces built [dim]({time.time() - t_stage:.1f}s)[/dim]")

    # 1st grouping: trace correlation
    t_stage = time.time()
    result1 = first_grouping(tracks, MIN_GROUP_CORR)
    sizes1 = result1.groupby("group")["joint_label"].transform("size")
    grouped1 = result1[sizes1 > 1]
    ungrouped_labels = result1.loc[sizes1 == 1, "joint_label"].tolist()

    groups = grouped1.groupby("group")["joint_label"].apply(list).reset_index(name="labels")
    groups["group"] = range(len(groups))
    groups["frames"] = groups["labels"].apply(lambda labs: sorted(f for l in labs for f in frames_by_label[l]))
    console.print(f"[cyan]group_tracks:[/cyan] first_grouping [dim]({time.time() - t_stage:.1f}s)[/dim]")

    # 2nd grouping: centroid distance, on the leftovers only
    t_stage = time.time()
    centroids = track_centroids(hotspots_props)
    sub_centroids = centroids[centroids["joint_label"].isin(ungrouped_labels)].reset_index(drop=True)
    result2 = second_grouping(sub_centroids, MAX_CENTROID_DEVIATION)
    sizes2 = result2.groupby("group")["joint_label"].transform("size")
    grouped2 = result2[sizes2 > 1]
    resting_labels = result2.loc[sizes2 == 1, "joint_label"].tolist()

    centroid_groups = grouped2.groupby("group")["joint_label"].apply(list).reset_index(name="labels")
    centroid_groups["group"] = range(len(centroid_groups))
    centroid_groups["frames"] = centroid_groups["labels"].apply(
        lambda labs: sorted(f for l in labs for f in frames_by_label[l]))

    # whatever is still alone
    resting = pd.DataFrame({"joint_label": sorted(resting_labels)})
    resting["frames"] = resting["joint_label"].apply(lambda l: frames_by_label[l])
    console.print(f"[cyan]group_tracks:[/cyan] second_grouping [dim]({time.time() - t_stage:.1f}s)[/dim]")

    console.print(f"[green]group_tracks: done[/green] [dim]({time.time() - t_start:.1f}s total)[/dim]")
    return groups, centroid_groups, resting


# ===========================================================================
#
#   STEP 3 -- MAP: groups -> zones -> masks, outlines, PNG maps
#
# ===========================================================================

# --- 3a. Zones and their masks ---------------------------------------------

def build_zones(groups: pd.DataFrame, centroid_groups: pd.DataFrame, resting: pd.DataFrame) -> pd.DataFrame:
    """One row per zone (ids 1..N across groups, centroid_groups, resting)."""
    zones = pd.concat([
        pd.DataFrame({"joint_labels": groups["labels"],
                      "origin": [f"group {i}" for i in range(len(groups))],
                      "sheet": "groups"}),
        pd.DataFrame({"joint_labels": centroid_groups["labels"],
                      "origin": [f"centroid_group {i}" for i in range(len(centroid_groups))],
                      "sheet": "centroid_groups"}),
        pd.DataFrame({"joint_labels": resting["joint_label"].apply(lambda l: [l]),
                      "origin": [f"resting row {i}" for i in range(len(resting))],
                      "sheet": "resting"}),
    ], ignore_index=True)
    zones["zone"] = range(1, len(zones) + 1)  # 1-based, 0 stays background
    return zones


def build_zone_masks(zones: pd.DataFrame, detections: pd.DataFrame, footprints: list,
                     H: int, W: int) -> dict[int, np.ndarray]:
    """zone id -> (H, W) bool mask: union of all member footprints over all frames."""
    joint_label_to_zone = {l: row.zone for row in zones.itertuples() for l in row.joint_labels}
    zone_masks: dict[int, np.ndarray] = {}

    for i, row in enumerate(detections.itertuples()):
        zone = joint_label_to_zone.get(row.joint_label)
        if zone is None:
            continue
        coords = footprints[i]
        mask = zone_masks.setdefault(zone, np.zeros((H, W), dtype=bool))
        mask[coords[:, 0], coords[:, 1]] = True
    return zone_masks


# --- 3b. Save zone data ----------------------------------------------------

def save_zone_footprints(zone_masks: dict[int, np.ndarray], out_path: str) -> None:
    """Save zone pixel coords to npz, keyed zone{id}."""
    footprints = {f"zone{zone_id}": np.argwhere(mask) for zone_id, mask in zone_masks.items()}
    np.savez(out_path, **footprints)


def save_zone_contours(zone_masks: dict[int, np.ndarray], out_path: str) -> None:
    """Save zone outlines to npz, keyed zone{id}_part{k}."""
    contours = {}
    for zone_id, mask in zone_masks.items():
        for i, contour in enumerate(find_contours(mask.astype(float), level=0.5)):
            contours[f"zone{zone_id}_part{i}"] = contour
    np.savez(out_path, **contours)


def zone_event_stats(frames: np.ndarray, n_frames: int, fps: float) -> tuple[int, float, float]:
    """(n_events, period_s, freq_hz); an event is a run of consecutive frames.
    Period = mean start-to-start interval, or the whole recording if only 1 event."""
    frames = np.unique(frames)
    if len(frames) == 0:
        return 0, np.nan, np.nan

    starts = frames[np.r_[True, np.diff(frames) > 1]]
    period_s = np.diff(starts).mean() / fps if len(starts) > 1 else n_frames / fps
    return len(starts), float(period_s), float(1 / period_s)


def write_zone_sizes(zones: pd.DataFrame, zone_masks: dict[int, np.ndarray], detections: pd.DataFrame,
                     n_frames: int, xlsx_path: str) -> None:
    """Add a 'zone_sizes' sheet (area + event frequency per zone) to the groups xlsx."""
    sizes = zones[["zone", "origin", "joint_labels"]].copy()
    sizes["n_members"] = sizes["joint_labels"].apply(len)
    sizes["pixel_area"] = sizes["zone"].map(lambda z: int(zone_masks[z].sum()) if z in zone_masks else 0)

    frames_by_label = detections.groupby("joint_label")["frame"].apply(np.array)
    event_stats = sizes["joint_labels"].apply(lambda labels: zone_event_stats(
        np.concatenate([frames_by_label.get(l, np.array([], dtype=int)) for l in labels]),
        n_frames, FRAME_RATE_HZ))
    sizes[["n_events", "period_s", "freq_hz"]] = pd.DataFrame(event_stats.tolist(), index=sizes.index)

    sizes = sizes.drop(columns="joint_labels").sort_values("zone")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        sizes.to_excel(writer, sheet_name="zone_sizes", index=False)


# --- 3c. Render PNGs -------------------------------------------------------

def tint_background(background: np.ndarray, color: str) -> np.ndarray:
    """Normalize to [0, 1]; if not gray, put it in one RGB channel."""
    bg_norm = (background - background.min()) / (background.max() - background.min())
    if color == "gray":
        return bg_norm

    channel = {"red": 0, "green": 1, "blue": 2}[color]
    bg_rgb = np.zeros((*bg_norm.shape, 3), dtype=bg_norm.dtype)
    bg_rgb[..., channel] = bg_norm
    return bg_rgb


def zone_centroid(zones: pd.DataFrame, detections: pd.DataFrame, zone_id: int):
    """Mean (y, x) of a zone's member tracks, or None."""
    centroids = detections.groupby("joint_label")[["centroid_y", "centroid_x"]].mean()
    joint_labels = zones.loc[zones["zone"] == zone_id, "joint_labels"].iloc[0]
    member_centroids = centroids.loc[centroids.index.intersection(joint_labels)]
    if member_centroids.empty:
        return None
    return member_centroids["centroid_y"].mean(), member_centroids["centroid_x"].mean()


def _label_zone(ax, zone_id: int, centroid) -> None:
    cy, cx = centroid
    ax.text(cx, cy, str(zone_id), color="white", fontsize=9, fontweight="bold",
            ha="center", va="center", bbox=dict(boxstyle="circle", fc="black", alpha=0.6))


def render_single_zone(zone_id: int, mask: np.ndarray, color, centroid, bg_tinted: np.ndarray,
                       bg_color: str, title_tag: str, out_path: str) -> None:
    """One zone's outline over the background."""
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(bg_tinted, cmap="gray" if bg_color == "gray" else None)
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=2.5)
    if centroid is not None:
        _label_zone(ax, zone_id, centroid)
    ax.set_title(f"zone {zone_id} ({title_tag})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def render_zone_overlay(zone_masks: dict[int, np.ndarray], zones: pd.DataFrame,
                        detections: pd.DataFrame, bg_tinted: np.ndarray,
                        H: int, W: int, title_tag: str, out_path: str) -> None:
    """All zones as translucent fills; largest painted first so small zones stay on top."""
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
        if centroid is not None:
            _label_zone(ax, zone_id, centroid)

    n_groups = (zones["sheet"] == "groups").sum()
    n_centroid_groups = (zones["sheet"] == "centroid_groups").sum()
    n_resting = (zones["sheet"] == "resting").sum()
    ax.set_title(f"{len(zone_masks)} zones -- {n_groups} trace-corr, "
                 f"{n_centroid_groups} centroid, {n_resting} resting ({title_tag})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def render_zone_maps(zone_masks: dict[int, np.ndarray], zones: pd.DataFrame, detections: pd.DataFrame,
                     bg_tinted: np.ndarray, bg_color: str, title_tag: str, out_dir: str) -> int:
    """Write 01_all_zones.png + one NN_zone{id}.png per zone; return zone count."""
    os.makedirs(out_dir, exist_ok=True)
    for stale_png in glob.glob(f"{out_dir}/*.png"):
        os.remove(stale_png)

    H, W = bg_tinted.shape[:2]
    present_zones = sorted(zone_masks)
    width = len(str(len(present_zones) + 1))
    zone_color = {z: plt.cm.tab20.colors[i % len(plt.cm.tab20.colors)]
                  for i, z in enumerate(present_zones)}

    render_zone_overlay(zone_masks, zones, detections, bg_tinted, H, W, title_tag,
                        f"{out_dir}/{1:0{width}d}_all_zones.png")

    for i, zone_id in enumerate(present_zones, start=2):
        render_single_zone(zone_id, zone_masks[zone_id], zone_color[zone_id],
                           zone_centroid(zones, detections, zone_id),
                           bg_tinted, bg_color, title_tag,
                           f"{out_dir}/{i:0{width}d}_zone{zone_id}.png")

    return len(present_zones)


# ===========================================================================
#
#   RUN
#
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hotspot -> zone pipeline.")
    parser.add_argument("stack", nargs="?", default=DEFAULT_STACK, help="deltaF/F0 stack (.tif)")
    parser.add_argument("--sigma", type=float, default=CROSSOVER_RATIO,
                        help="threshold = background peak + this many sigmas")
    parser.add_argument("--bg", default=None, help="stack used as map background (default: the input stack)")
    parser.add_argument("--proj", choices=["mean", "max"], default="mean", help="background projection")
    parser.add_argument("--color", choices=["gray", "red", "green", "blue"], default="gray",
                        help="background tint")
    return parser.parse_args()


def main() -> None:
    t_start = time.time()
    args = parse_args()

    stem = Path(args.stack).stem
    paths = output_paths(stem, args.proj)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stack, stack_f16 = load_stack(args.stack)
    H, W = stack.shape[1], stack.shape[2]


    # -----------------------------------------------------------------------
    # Step 1. Detect
    # -----------------------------------------------------------------------
    console.rule("[bold]Step 1 - Detect")

    threshold = find_background_threshold(stack_f16, args.sigma)
    console.print(f"[cyan]background threshold[/cyan] (peak + {args.sigma} sigma): [bold]{threshold:.5f}[/bold]")

    mask = detect_all_frames(stack_f16, threshold)

    tifffile.imwrite(paths["mask_tif"], mask.astype(np.uint8) * 255)
    console.print(f"[green]saved[/green] cleaned mask {mask.shape} -> {paths['mask_tif']}")


    # -----------------------------------------------------------------------
    # Step 2. Group
    # -----------------------------------------------------------------------
    console.rule("[bold]Step 2 - Group")

    # 2a. per-frame hotspots
    detections, footprints = spatiotemporally_connect_hotspots(mask, TH_SMALL_HOTSPOTS, CONNECT_RADIUS)

    # 2b. chain across frames into tracks
    n_before = detections["joint_label"].nunique()
    detections = assign_frame_adjacent_joint_labels(detections, MAX_CENTROID_DEVIATION)
    console.print(f"[cyan]assign_frame_adjacent_joint_labels:[/cyan] {n_before} -> "
                  f"[bold]{detections['joint_label'].nunique()}[/bold] joint_labels "
                  f"(<{MAX_CENTROID_DEVIATION}px chains)")

    detections.to_csv(paths["detections_csv"], index=False)
    np.savez(paths["hotspots_npz"], background_threshold=threshold,
             **{f"hotspot_{i}": coords for i, coords in enumerate(footprints)})
    console.print(f"[green]saved[/green] {len(detections)} detections -> "
                  f"{paths['detections_csv']}, {paths['hotspots_npz']}")

    # 2c + 2d. traces -> two-stage grouping
    groups, centroid_groups, resting = group_tracks(detections, footprints, stack_f16)

    with pd.ExcelWriter(paths["groups_xlsx"]) as writer:
        groups.to_excel(writer, sheet_name="groups", index=False)
        centroid_groups.to_excel(writer, sheet_name="centroid_groups", index=False)
        resting.to_excel(writer, sheet_name="resting", index=False)
    console.print(f"[green]saved[/green] {len(groups)} trace-corr groups (r>{MIN_GROUP_CORR}), "
                  f"{len(centroid_groups)} centroid groups (<{MAX_CENTROID_DEVIATION}px), "
                  f"{len(resting)} resting -> {paths['groups_xlsx']}")


    # -----------------------------------------------------------------------
    # Step 3. Map
    # -----------------------------------------------------------------------
    console.rule("[bold]Step 3 - Map")

    # 3a. zones and masks
    zones = build_zones(groups, centroid_groups, resting)
    zone_masks = build_zone_masks(zones, detections, footprints, H, W)

    # 3b. save zone data
    write_zone_sizes(zones, zone_masks, detections, stack.shape[0], paths["groups_xlsx"])
    save_zone_footprints(zone_masks, paths["zone_fp_npz"])
    save_zone_contours(zone_masks, paths["zone_contour_npz"])

    # 3c. render PNGs
    bg_stack = stack if args.bg is None else tifffile.imread(args.bg)
    background = bg_stack.max(axis=0) if args.proj == "max" else bg_stack.mean(axis=0)
    bg_tinted = tint_background(background, args.color)

    n_zones = render_zone_maps(zone_masks, zones, detections, bg_tinted, args.color,
                               f"on {stem}, {args.proj} proj", paths["zone_map_dir"])
    console.print(f"[green]saved[/green] {n_zones} zones -> {n_zones + 1} pngs in {paths['zone_map_dir']}/")


    console.rule(f"[dim]Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
