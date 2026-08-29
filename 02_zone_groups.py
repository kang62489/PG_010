"""
Categorize all detected hotspots as "zone" based on temporal correlation, centroid proximity, and a few other heuristics.

Pipeline:
  1. Load stack + cleaned boolean mask (from 01)
  2. Bridge spatial/temporal gaps in the mask, label in 3D, and build every
     detection's frame, joint_label, centroid, and area
  3. Collapse to one representative centroid per track
  4. Pull each track's fixed-ROI deltaF/F0 trace, across the whole stack
  5. Cluster tracks by trace correlation (gated by centroid distance) ->
     zones, saved as zone_corr_matrix.csv
  6. Map each detection's zone from its joint_label and save zone_groups.csv

Requires: mask.tif (from 01_detect_hotspots.py)
Run: uv run python 02_zone_groups.py [stack.tif] [output_suffix]
Outputs: detections_raw.csv, hotspot_footprints.npz, zone_groups.csv, zone_traces.png, zone_corr_matrix.csv
"""

import importlib
import sys
import time

import numpy as np
import pandas as pd
import tifffile
from rich.console import Console
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

detect_hotspots = importlib.import_module("01_detect_hotspots")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

console = Console()

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 02_zone_groups.py [stack.tif] [output_suffix]
#    Suffix must match the one used for 01_detect_hotspots.py's outputs.
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "proc_tiffs/2025_12_15-0012_BIEXP_ALS.tif"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else "_ALS"

# per-frame cleaned boolean mask, as a uint8 TIFF (from 01_detect_hotspots.py)
MASK_TIF = f"results/mask{SUFFIX}.tif"

TH_SMALL_HOTSPOTS = 3000  # remove hotspots which is too small (px)

CONNECT_RADIUS = 5  # px: bridge gaps up to this wide between fragments before
                      # labeling, so one real hotspot broken apart by a noisy
                      # dip is still detected as a single hotspot, not several

TEMPORAL_GAP = 1  # frames: also bridge across this many frames before/after,
                   # so two same-frame fragments that are really one hotspot
                   # (confirmed by both touching a single hotspot in a
                   # neighboring frame) get merged instead of counted twice

# frame, joint_label, centroid_y, centroid_x, area
RAW_DETECTIONS_CSV = f"results/detections_raw{SUFFIX}.csv"

# pixel coordinates of all detected hotspots
HOTSPOT_FOOTPRINTS_NPZ = f"results/hotspot_footprints{SUFFIX}.npz"

MAX_CENTROID_DEVIATION = 75  # px: second-stage grouping -- tracks left
                               # ungrouped by trace correlation can still
                               # join a group if their centroids stay within
                               # this of every other member (complete-linkage).

MIN_GROUP_CORR = 0.95  # first-pass grouping: complete-linkage clustering
                        # cut at this Pearson r -- every pair inside a group
                        # is guaranteed to correlate at least this much.

ZONE_GROUPS_CSV = f"results/zone_groups{SUFFIX}.csv"
ZONE_TRACES_PNG = f"results/zone_traces{SUFFIX}.png"
ZONE_CORR_MATRIX_CSV = f"results/zone_corr_matrix{SUFFIX}.csv"

CORR_GROUPS_XLSX = f"results/corr_groups{SUFFIX}.xlsx"


# ---------------------------------------------------------------------------
# 2. Bridge the mask in 3D (spatial only) and build per-detection rows
# ---------------------------------------------------------------------------

def spatiotemporally_connect_hotspots(mask: np.ndarray, th_small_hotspots: int,
                                connect_radius: int, temporal_gap: int) -> tuple[pd.DataFrame, list]:
    t_start = time.time()
    console.print("[cyan]spatiotemporally_connect_hotspots:[/cyan] starting...")

    # count hotspots in the input mask before any bridging, for comparison
    _, n_before = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    console.print(f"[cyan]spatiotemporally_connect_hotspots:[/cyan] counted {n_before} raw hotspots "
                  f"[dim]({time.time() - t_start:.1f}s)[/dim]")

    # use dilation to check if two adjacent hotspots are actually one hotspot, if so, connect them (using OR operation).
    bridged_mask = ndimage.binary_dilation(mask, structure=np.ones((1, connect_radius, connect_radius)))
    # using a 3D array, dilation with a 2D structure array stacked across 3 frames reaches the frame before and after the current frame (determined by temporal_gap = 1).
    bridged_mask = ndimage.binary_dilation(bridged_mask, structure=np.ones((1 + 2 * temporal_gap, 1, 1)))
    # apply new label index to bridged_mask for several measurements with regionprops later
    labeled_bridged_mask, n_after = ndimage.label(bridged_mask, structure=np.ones((3, 3, 3)), output=np.int32)
    del bridged_mask
    console.print(f"[cyan]spatiotemporally_connect_hotspots:[/cyan] labeling done [dim]({time.time() - t_start:.1f}s)[/dim]")

    # remove the connecting pixels (dilated pixels) from labeled_bridged_mask by AND operation with the original mask
    hotspots_props = []
    footprints = []


    joint_label_count = 1
    joint_label_history: dict[int, int] = {}  # joint_label_at_frame_id -> joint_label currently in use
    last_seen_frame: dict[int, int] = {}     # joint_label_at_frame_id -> last frame_id it produced a row for

    # Interate through each frame of mask to break the joint labels that supposed to be separate
    for frame_id in range(mask.shape[0]):
        for joint_label_at_frame_id in np.unique(labeled_bridged_mask[frame_id][mask[frame_id]]):
            # above line using boolean mask indexing to get pixels that are truely bright in the original mask (not the dialated)
            footprint_mask = mask[frame_id] & (labeled_bridged_mask[frame_id] == joint_label_at_frame_id)
            area = int(footprint_mask.sum())
            if area < th_small_hotspots:
                continue

            # Temproal Gap Detection
            joint_label_at_frame_id = int(joint_label_at_frame_id)
            # gap: frames since this raw label was last seen -- unseen labels default to a
            # gap that's always > 1, so "first time seen" and "real gap" hit the same branch.
            gap = frame_id - last_seen_frame.get(joint_label_at_frame_id, -2)
            if gap > 1:
                joint_label = joint_label_count            # add the current joint_label_at_frame_id to for counting the next
                joint_label_count += 1
            else:
                joint_label = joint_label_history[joint_label_at_frame_id]
            joint_label_history[joint_label_at_frame_id] = joint_label
            last_seen_frame[joint_label_at_frame_id] = frame_id                             # will be updated in the next frame if the same joint_label_at_frame_id is seen again, otherwise will be not executed and gap will be detected.

            coords = np.argwhere(footprint_mask)
            footprints.append(coords)
            hotspots_props.append({
                "frame": frame_id + 1,  # 1-based for the CSV; loop/array indexing stays 0-based
                "joint_label": joint_label,
                "centroid_y": float(coords[:, 0].mean()),
                "centroid_x": float(coords[:, 1].mean()),
                "area": area,
            })

    n_after = joint_label_count - 1
    console.print(f"hotspots in input mask: {n_before} -> after spatial/temporal connecting: [bold]{n_after}[/bold] "
                  f"([yellow]{n_before - n_after} merged[/yellow])")
    console.print(f"[green]spatiotemporally_connect_hotspots: done[/green], {len(hotspots_props)} detections "
                  f"[dim]({time.time() - t_start:.1f}s total)[/dim]")
    return pd.DataFrame(hotspots_props), footprints


# ---------------------------------------------------------------------------
# 3. Load the stack, build each track's representative trace, pairwise
#    cross-correlation, and first grouping (corr > 0.95, complete-linkage)
# ---------------------------------------------------------------------------

def load_stack(stack_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Already deltaF/F0 -- no baseline subtraction here."""
    stack = tifffile.imread(stack_path)
    stack_f16 = stack.astype(np.float16)
    return stack, stack_f16


def detection_traces(footprints: list, stack_f16: np.ndarray) -> np.ndarray:
    """(n_detections, n_frames) matrix: one trace per detection, using each
    detection's own footprint pixels -- a joint_label detected at 3 frames
    gets 3 separate traces here, not one."""
    return np.stack([stack_f16[:, coords[:, 0], coords[:, 1]].mean(axis=1) for coords in footprints])


def track_mean_traces(hotspots_props: pd.DataFrame, detection_traces: np.ndarray) -> pd.DataFrame:
    """One row per joint_label: the average of its own detection traces
    (track_mean_traces) across every frame-appearance -- collapses a
    joint_label's several own-footprint traces into the single trace used
    for pairwise correlation."""
    rows = []
    for joint_label, group in hotspots_props.groupby("joint_label"):
        rows.append({"joint_label": joint_label, "trace": detection_traces[group.index].mean(axis=0)})
    return pd.DataFrame(rows)


def first_grouping(tracks: pd.DataFrame, min_corr: float) -> pd.DataFrame:
    """First-pass grouping by trace correlation alone: complete-linkage
    clustering, cut at distance (1 - min_corr) -- guarantees every pair
    inside a group correlates at least min_corr, unlike single-linkage/
    connected-components, which can chain two poorly-correlated tracks
    together through an intermediate one."""
    labels = tracks["joint_label"].tolist()
    traces = np.stack(tracks["trace"].values)

    corr = np.corrcoef(traces)
    distance = 1 - corr
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2  # force exact symmetry (fp round-trip)

    linkage_matrix = linkage(squareform(distance, checks=False), method="complete")
    raw_group_ids = fcluster(linkage_matrix, t=1 - min_corr, criterion="distance")

    # renumber sequentially (0, 1, 2, ...) instead of scipy's arbitrary/gappy ids
    group_id_map = {raw: new for new, raw in enumerate(pd.unique(raw_group_ids))}
    group_ids = [group_id_map[raw] for raw in raw_group_ids]

    return pd.DataFrame({"joint_label": labels, "group": group_ids})


# ---------------------------------------------------------------------------
# 4. Cluster tracks: centroid deviation gate + trace correlation
# ---------------------------------------------------------------------------

def track_centroids(hotspots_props: pd.DataFrame) -> pd.DataFrame:
    """One row per joint_label: its mean (y, x) centroid, averaged across
    every detection (frame-appearance) confirmed to be that same physical
    hotspot."""
    grouped = hotspots_props.groupby("joint_label")[["centroid_y", "centroid_x"]].mean()
    mean_centroid = list(zip(grouped["centroid_y"], grouped["centroid_x"]))
    return pd.DataFrame({"joint_label": grouped.index, "mean_centroid": mean_centroid})


def second_grouping(centroids: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    """Second-pass grouping, run on tracks left ungrouped by first_grouping:
    complete-linkage clustering on centroid distance alone, cut at max_dist --
    guarantees every pair inside a group has centroids within max_dist of
    each other, same complete-linkage guarantee as first_grouping."""
    labels = centroids["joint_label"].tolist()
    coords = np.stack(centroids["mean_centroid"].values)

    distance = squareform(pdist(coords))

    linkage_matrix = linkage(squareform(distance, checks=False), method="complete")
    raw_group_ids = fcluster(linkage_matrix, t=max_dist, criterion="distance")

    # renumber sequentially (0, 1, 2, ...) instead of scipy's arbitrary/gappy ids
    group_id_map = {raw: new for new, raw in enumerate(pd.unique(raw_group_ids))}
    group_ids = [group_id_map[raw] for raw in raw_group_ids]

    return pd.DataFrame({"joint_label": labels, "group": group_ids})


def cluster_by_centroid_deviation(corr: np.ndarray, centroid_dist: np.ndarray,
                            min_trace_corr: float, max_centroid_deviation: float) -> np.ndarray:
    """Hierarchical (average-linkage) clustering on 1 - correlation as the
    distance, cut at distance (1 - min_trace_corr) -- except any pair whose
    centroids are farther apart than max_centroid_deviation is forced to
    distance 1 (== zero correlation) first, so two far-apart hotspots can
    never share a zone no matter how well their traces correlate."""
    distance = 1 - corr
    distance[centroid_dist > max_centroid_deviation] = 1

    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2  # force exact symmetry (fp round-trip)

    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    return fcluster(linkage_matrix, t=1 - min_trace_corr, criterion="distance")


def export_corr_groups(hotspots_props: pd.DataFrame, footprints: list, stack_f16: np.ndarray,
                        out_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage grouping over every joint_label: first by trace correlation
    (first_grouping), then the trace-ungrouped leftovers by centroid distance
    (second_grouping). Saves 3 sheets to out_path -- groups, centroid_groups,
    resting -- and returns the same three DataFrames."""
    t_start = time.time()
    console.print("[cyan]export_corr_groups:[/cyan] starting...")

    det_traces = detection_traces(footprints, stack_f16)
    tracks = track_mean_traces(hotspots_props, det_traces)
    frames_by_label = hotspots_props.groupby("joint_label")["frame"].apply(lambda s: sorted(s.tolist()))
    console.print(f"[cyan]export_corr_groups:[/cyan] traces built [dim]({time.time() - t_start:.1f}s)[/dim]")

    result1 = first_grouping(tracks, MIN_GROUP_CORR)
    sizes1 = result1.groupby("group")["joint_label"].transform("size")
    grouped1 = result1[sizes1 > 1]
    ungrouped_labels = result1.loc[sizes1 == 1, "joint_label"].tolist()

    groups = grouped1.groupby("group")["joint_label"].apply(list).reset_index(name="labels")
    groups["group"] = range(len(groups))
    groups["frames"] = groups["labels"].apply(lambda labs: sorted(f for l in labs for f in frames_by_label[l]))

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

    resting = pd.DataFrame({"joint_label": sorted(resting_labels)})
    resting["frames"] = resting["joint_label"].apply(lambda l: frames_by_label[l])

    with pd.ExcelWriter(out_path) as writer:
        groups.to_excel(writer, sheet_name="groups", index=False)
        centroid_groups.to_excel(writer, sheet_name="centroid_groups", index=False)
        resting.to_excel(writer, sheet_name="resting", index=False)

    console.print(f"[green]export_corr_groups: done[/green] [dim]({time.time() - t_start:.1f}s total)[/dim]")
    return groups, centroid_groups, resting


# ---------------------------------------------------------------------------
# 5. Render each zone's traces, for the sanity-check plot only.
# ---------------------------------------------------------------------------

def render_zone_traces(traces: np.ndarray, zones: np.ndarray, out_path: str) -> None:
    zone_ids = sorted(set(zones))
    n_cols = 4
    n_rows = int(np.ceil(len(zone_ids) / n_cols))
    _, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.2 * n_rows), squeeze=False)

    for ax, zone_id in zip(axes.flat, zone_ids):
        for trace in traces[zones == zone_id]:
            ax.plot(trace, linewidth=0.5)
        ax.set_title(f"zone {zone_id} (n_hotspots={np.sum(zones == zone_id)})", fontsize=9)
        ax.tick_params(labelsize=6)

    for ax in axes.flat[len(zone_ids):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)


# ---------------------------------------------------------------------------
# 6. Run the pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    _, stack_f16 = load_stack(STACK_PATH)

    mask = tifffile.imread(MASK_TIF) > 0

    background_threshold = detect_hotspots.find_background_threshold(stack_f16)

    hotspots_props, footprints = spatiotemporally_connect_hotspots(mask, TH_SMALL_HOTSPOTS, CONNECT_RADIUS, TEMPORAL_GAP)
    hotspots_props.to_csv(RAW_DETECTIONS_CSV, index=False)
    np.savez(HOTSPOT_FOOTPRINTS_NPZ, background_threshold=background_threshold,
             **{f"hotspot_{i}": coords for i, coords in enumerate(footprints)})
    console.print(f"[bold]{len(hotspots_props)}[/bold] raw hotspot detections above area {TH_SMALL_HOTSPOTS} -> "
                  f"{RAW_DETECTIONS_CSV}, {HOTSPOT_FOOTPRINTS_NPZ}")

    groups, centroid_groups, resting = export_corr_groups(hotspots_props, footprints, stack_f16, CORR_GROUPS_XLSX)
    console.print(f"[bold]{len(groups)}[/bold] trace-corr groups (r>{MIN_GROUP_CORR}), "
                  f"[bold]{len(centroid_groups)}[/bold] centroid groups (<{MAX_CENTROID_DEVIATION}px), "
                  f"[bold]{len(resting)}[/bold] resting -> {CORR_GROUPS_XLSX}")

    # centroid_dist = squareform(pdist(centroids))
    # corr = np.corrcoef(traces)
    # pd.DataFrame(corr).to_csv(ZONE_CORR_MATRIX_CSV, index=False)
    # console.print(f"[green]saved[/green] {len(joint_labels)}x{len(joint_labels)} track-level ROI trace correlation "
    #               f"matrix -> {ZONE_CORR_MATRIX_CSV}")
    #
    # track_zones = cluster_by_centroid_deviation(corr, centroid_dist, MIN_GROUP_CORR, MAX_CENTROID_DEVIATION)
    # zone_by_track = dict(zip(joint_labels, track_zones))
    #
    # hotspots_props = hotspots_props.copy()
    # hotspots_props["zone"] = hotspots_props["joint_label"].map(zone_by_track)
    # hotspots_props.to_csv(ZONE_GROUPS_CSV, index=False)
    #
    # own_traces = detection_traces(footprints, stack_f16)
    # render_zone_traces(own_traces, hotspots_props["zone"].values, ZONE_TRACES_PNG)
    #
    # console.print(f"[bold]{hotspots_props['zone'].nunique()}[/bold] zones from {len(joint_labels)} tracks "
    #               f"({len(hotspots_props)} detections) [dim](max_centroid_deviation={MAX_CENTROID_DEVIATION}px, "
    #               f"min_group_corr={MIN_GROUP_CORR})[/dim]")
    # console.print(f"[green]saved[/green] {ZONE_GROUPS_CSV}, {ZONE_TRACES_PNG}")


if __name__ == "__main__":
    main()
