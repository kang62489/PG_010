"""
Group hotspot tracks into spatial zones by ROI-centroid temporal
correlation only -- no shape comparison. See track_centroids(),
fixed_roi_traces(), and cluster_by_correlation() for how.

Pipeline:
  1. Load detections_raw.csv + hotspot_footprints.npz (from 01), each row
     tagged with a joint_label
  2. Collapse to one representative centroid per track
  3. Pull each track's fixed-ROI deltaF/F0 trace, across the whole stack
  4. Cluster tracks by trace correlation (gated by centroid distance) ->
     zones, saved as zone_corr_matrix.csv
  5. Map each detection's zone from its joint_label and save zone_groups.csv

Requires: detections_raw.csv, hotspot_footprints.npz (from 01_detect_hotspots.py)
Run: uv run python 02_zone_groups.py [stack.tif] [output_suffix]
Outputs: zone_groups.csv, zone_traces.png, zone_corr_matrix.csv
"""

import sys

import numpy as np
import pandas as pd
import tifffile
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform, pdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 02_zone_groups.py [stack.tif] [output_suffix]
#    Suffix must match the one used for 01_detect_hotspots.py's outputs.
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "2025_12_15-0012_BIEXP_GAUSS.tif"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""

RAW_DETECTIONS_CSV = f"detections_raw{SUFFIX}.csv"
HOTSPOT_FOOTPRINTS_NPZ = f"hotspot_footprints{SUFFIX}.npz"

MAX_CENTROID_DEVIATION = 30  # px: two tracks can only be the same zone if
                              # their centroids stay within this of each
                              # other -- a hard requirement, checked before
                              # any correlation is even computed.

ROI_RADIUS = 20  # px: fixed half-width of the square ROI used to pull each
                  # track's trace, centered on its own centroid -- same size
                  # for every track regardless of its own footprint's size,
                  # so traces are measured on equal footing.

MIN_TRACE_CORR = 0.75  # two tracks join the same zone only if their
                        # fixed-ROI deltaF/F0 traces correlate (Pearson r)
                        # at least this much.

ZONE_GROUPS_CSV = f"zone_groups{SUFFIX}.csv"
ZONE_TRACES_PNG = f"zone_traces{SUFFIX}.png"
ZONE_CORR_MATRIX_CSV = f"zone_corr_matrix{SUFFIX}.csv"


# ---------------------------------------------------------------------------
# 2. Load the stack (already deltaF/F0 -- no baseline subtraction here)
# ---------------------------------------------------------------------------

def load_stack(stack_path: str) -> tuple[np.ndarray, np.ndarray]:
    stack = tifffile.imread(stack_path)
    stack_f16 = stack.astype(np.float16)
    return stack, stack_f16


# ---------------------------------------------------------------------------
# 3. One representative centroid per track + its fixed-ROI trace
# ---------------------------------------------------------------------------

def track_centroids(hotspots: pd.DataFrame) -> tuple[list, np.ndarray]:
    """One (y, x) centroid per joint_label, averaged across every detection
    (frame-appearance) confirmed to be that same physical hotspot."""
    grouped = hotspots.groupby("joint_label")[["y", "x"]].mean()
    return grouped.index.tolist(), grouped.values


def fixed_roi_traces(stack_f16: np.ndarray, centroids: np.ndarray, roi_radius: int) -> np.ndarray:
    """(n_tracks, n_frames) matrix: one trace per track, using a FIXED-size
    square ROI centered on the track's own centroid -- not its actual
    footprint, which varies in size/shape from track to track and would
    bias the comparison. The ROI can extend past the track's own footprint
    boundary; it's a standardized measuring window, not a shape
    descriptor."""
    H, W = stack_f16.shape[1], stack_f16.shape[2]
    traces = []
    for cy, cx in centroids:
        y0 = max(0, int(round(cy)) - roi_radius)
        y1 = min(H, int(round(cy)) + roi_radius + 1)
        x0 = max(0, int(round(cx)) - roi_radius)
        x1 = min(W, int(round(cx)) + roi_radius + 1)
        traces.append(stack_f16[:, y0:y1, x0:x1].mean(axis=(1, 2)))
    return np.array(traces)


# ---------------------------------------------------------------------------
# 4. Cluster tracks: centroid deviation gate + trace correlation
# ---------------------------------------------------------------------------

def cluster_by_correlation(corr: np.ndarray, centroid_dist: np.ndarray,
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


# ---------------------------------------------------------------------------
# 5. Pull each detection's own-footprint brightness trace, for the
#    sanity-check plot only -- no part of clustering.
# ---------------------------------------------------------------------------

def all_hotspot_traces(stack_f16: np.ndarray, n_hotspots: int, footprints_path: str) -> np.ndarray:
    """(n_hotspots, n_frames) matrix: one trace per detection, using each
    detection's own footprint pixels (not the fixed ROI used for clustering)."""
    with np.load(footprints_path) as footprints:
        coords = [footprints[f"hotspot_{i}"] for i in range(n_hotspots)]
    return np.stack([stack_f16[:, c[:, 0], c[:, 1]].mean(axis=1) for c in coords])


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

    hotspots = pd.read_csv(RAW_DETECTIONS_CSV)
    print(f"{len(hotspots)} hotspot detections")

    track_ids, centroids = track_centroids(hotspots)
    print(f"{len(track_ids)} 3D-confirmed hotspot tracks")

    centroid_dist = squareform(pdist(centroids))

    traces = fixed_roi_traces(stack_f16, centroids, ROI_RADIUS)
    corr = np.corrcoef(traces)
    pd.DataFrame(corr).to_csv(ZONE_CORR_MATRIX_CSV, index=False)
    print(f"saved {len(track_ids)}x{len(track_ids)} track-level ROI trace correlation matrix -> {ZONE_CORR_MATRIX_CSV}")

    track_zones = cluster_by_correlation(corr, centroid_dist, MIN_TRACE_CORR, MAX_CENTROID_DEVIATION)
    zone_by_track = dict(zip(track_ids, track_zones))

    hotspots = hotspots.copy()
    hotspots["zone"] = hotspots["joint_label"].map(zone_by_track)
    hotspots.to_csv(ZONE_GROUPS_CSV, index=False)

    detection_traces = all_hotspot_traces(stack_f16, len(hotspots), HOTSPOT_FOOTPRINTS_NPZ)
    render_zone_traces(detection_traces, hotspots["zone"].values, ZONE_TRACES_PNG)

    print(f"{hotspots['zone'].nunique()} zones from {len(track_ids)} tracks ({len(hotspots)} detections) "
          f"(max_centroid_deviation={MAX_CENTROID_DEVIATION}px, roi_radius={ROI_RADIUS}px, "
          f"min_trace_corr={MIN_TRACE_CORR})")
    print(f"saved {ZONE_GROUPS_CSV}, {ZONE_TRACES_PNG}")


if __name__ == "__main__":
    main()
