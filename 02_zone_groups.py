"""
Group per-frame blob detections into spatial zones using cross-correlation
of their own temporal traces, instead of a distance/frame-gap heuristic.

Why cross-correlation instead of "close in space + close in time":
  A recurring flare's blob detections drift a little frame to frame and can
  vanish for a while between occurrences. Measuring straight-line distance
  between centroids (and guessing a frame-gap cutoff) is a crude proxy for
  "is this the same physical flare". Two blobs that really are the same
  flare will light up together across the WHOLE 1200-frame stack -- so their
  footprint-averaged brightness traces will be strongly correlated, no
  matter how far apart in time the individual detections were. Two blobs
  that are merely nearby but physically different flares won't share a
  lighting-up pattern, so their traces won't correlate.

Each blob's trace is the mean dff over its OWN exact footprint pixels
(from 01_detect_blobs.py) -- no synthetic circular ROI, no radius guess.

Pipeline:
  1. Load detections_raw.csv + blob_footprints.npz (each row: one blob, one
     frame, with its real pixel coordinates, from 01)
  2. Pull each blob's mean-dff trace, over its own footprint, across all
     1200 frames
  3. Cross-correlate every pair of blob traces (zero-lag Pearson, all pairs
     at once) -- exported as zone_corr_matrix.csv for inspection
  4. Cluster blobs into zones: hierarchical (average-linkage) clustering on
     a combined distance -- 1 - correlation, but forced to "maximally far"
     for any pair beyond MAX_SPATIAL_DISTANCE apart. Average-linkage only
     merges two groups when their AVERAGE cross-pairwise distance is low, so
     a single distant pair discourages (not just forbids one edge, the way a
     plain adjacency graph would) the whole groups from merging -- that
     graph version was tried first and produced worse chaining, not less.
  5. Save every blob with its assigned zone id

Requires: detections_raw.csv, blob_footprints.npz (from 01_detect_blobs.py), dff_utils.py
Run: uv run python 02_zone_groups.py [stack.tif] [output_suffix]
Outputs: zone_groups.csv, zone_traces.png, zone_corr_matrix.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform, pdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dff_utils import load_dff

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 02_zone_groups.py [stack.tif] [output_suffix]
#    Suffix must match the one used for 01_detect_blobs.py's outputs.
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "2025_12_15-0012_BIEXP_GAUSS.tif"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""

RAW_DETECTIONS_CSV = f"detections_raw{SUFFIX}.csv"
BLOB_FOOTPRINTS_NPZ = f"blob_footprints{SUFFIX}.npz"

CORR_THRESHOLD = 0.75    # two blobs join the same zone if their full-length traces
                         # correlate above this. Same-flare detections (even from
                         # frames 1000 apart) reliably correlate ~0.9+; unrelated
                         # nearby blobs sit near 0 -- 0.5 sits well clear of both.

MAX_SPATIAL_DISTANCE = 250  # px: correlation alone isn't enough -- two blobs on
                             # opposite sides of the image can coincidentally
                             # correlate above threshold (small sample size, shared
                             # residual drift) and get merged despite being nowhere
                             # near each other. A blob pair only joins the same zone
                             # if it clears BOTH the correlation bar AND this
                             # distance bar.

ZONE_GROUPS_CSV = f"zone_groups{SUFFIX}.csv"
ZONE_TRACES_PNG = f"zone_traces{SUFFIX}.png"
ZONE_CORR_MATRIX_CSV = f"zone_corr_matrix{SUFFIX}.csv"


# ---------------------------------------------------------------------------
# 2. Pull each blob's mean-dff trace over its own exact footprint
# ---------------------------------------------------------------------------

def all_blob_traces(dff: np.ndarray, n_blobs: int, footprints_path: str) -> np.ndarray:
    """(n_blobs, n_frames) matrix: one trace per blob, using each blob's
    own footprint pixels (not an approximated ROI)."""
    with np.load(footprints_path) as footprints:
        coords = [footprints[f"blob_{i}"] for i in range(n_blobs)]
    return np.stack([dff[:, c[:, 0], c[:, 1]].mean(axis=1) for c in coords])


# ---------------------------------------------------------------------------
# 3. Cross-correlate every pair of traces, cluster into zones
# ---------------------------------------------------------------------------

def correlation_matrix(traces: np.ndarray) -> np.ndarray:
    """Zero-lag Pearson correlation between every pair of blob traces, all
    N*(N-1)/2 pairs at once (np.corrcoef), not one pair at a time."""
    return np.corrcoef(traces)


def cluster_by_correlation(corr: np.ndarray, blobs: pd.DataFrame,
                            corr_threshold: float, max_spatial_distance: float) -> np.ndarray:
    """Hierarchical (average-linkage) clustering on 1 - correlation as the
    distance, cut at distance (1 - corr_threshold) -- except any pair beyond
    max_spatial_distance apart is forced to distance 1 (== zero correlation)
    first, so a coincidentally-correlated but far-apart pair can never pull
    two real clusters together."""
    distance = 1 - np.clip(corr, 0, 1)  # anti-correlated blobs are just as
                                         # "different" as uncorrelated ones

    centroid_dist = squareform(pdist(blobs[["y", "x"]].values))
    distance[centroid_dist > max_spatial_distance] = 1

    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2  # force exact symmetry (fp round-trip)

    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    return fcluster(linkage_matrix, t=1 - corr_threshold, criterion="distance")


# ---------------------------------------------------------------------------
# 4. Render one trace subplot per zone, for a visual sanity check
# ---------------------------------------------------------------------------

def render_zone_traces(traces: np.ndarray, zones: np.ndarray, out_path: str) -> None:
    zone_ids = sorted(set(zones))
    n_cols = 4
    n_rows = int(np.ceil(len(zone_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.2 * n_rows), squeeze=False)

    for ax, zone_id in zip(axes.flat, zone_ids):
        for trace in traces[zones == zone_id]:
            ax.plot(trace, linewidth=0.5)
        ax.set_title(f"zone {zone_id} (n_blobs={np.sum(zones == zone_id)})", fontsize=9)
        ax.tick_params(labelsize=6)

    for ax in axes.flat[len(zone_ids):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)


# ---------------------------------------------------------------------------
# 5. Run the pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    _, dff = load_dff(STACK_PATH)

    blobs = pd.read_csv(RAW_DETECTIONS_CSV)
    traces = all_blob_traces(dff, len(blobs), BLOB_FOOTPRINTS_NPZ)
    print(f"{len(blobs)} blob traces, each length {traces.shape[1]} frames")

    corr = correlation_matrix(traces)
    pd.DataFrame(corr).to_csv(ZONE_CORR_MATRIX_CSV, index=False)
    print(f"saved {len(blobs)}x{len(blobs)} correlation matrix -> {ZONE_CORR_MATRIX_CSV}")

    zones = cluster_by_correlation(corr, blobs, CORR_THRESHOLD, MAX_SPATIAL_DISTANCE)

    blobs = blobs.copy()
    blobs["zone"] = zones
    blobs.to_csv(ZONE_GROUPS_CSV, index=False)
    render_zone_traces(traces, zones, ZONE_TRACES_PNG)

    print(f"{len(set(zones))} zones from {len(blobs)} blobs (corr_threshold={CORR_THRESHOLD})")
    print(f"saved {ZONE_GROUPS_CSV}, {ZONE_TRACES_PNG}")


if __name__ == "__main__":
    main()
