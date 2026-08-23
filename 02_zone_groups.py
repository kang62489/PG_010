"""
Group per-frame blob detections into spatial zones using cross-correlation
of their own temporal traces, instead of a distance/frame-gap heuristic.

Why cross-correlation instead of "close in space + close in time":
  A recurring flare's blob detections drift a little frame to frame and can
  vanish for a while between occurrences. Measuring straight-line distance
  between centroids (and guessing a frame-gap cutoff) is a crude proxy for
  "is this the same physical flare". Two blobs that really are the same
  flare will light up together across the WHOLE 1200-frame stack -- so their
  ROI-averaged brightness traces will be strongly correlated, no matter how
  far apart in time the individual detections were. Two blobs that are
  merely nearby but physically different flares won't share a lighting-up
  pattern, so their traces won't correlate.

Pipeline:
  1. Load detections_raw.csv (each row: one blob, one frame, from 01)
  2. For each blob, build a small ROI at its own centroid, sized from its
     own area, and pull its mean-dff trace across all 1200 frames
  3. Cross-correlate every pair of blob traces
  4. Cluster blobs into zones: two blobs join the same zone if their traces
     are correlated above CORR_THRESHOLD (transitively, via hierarchical
     clustering on 1 - correlation as the distance)
  5. Save every blob with its assigned zone id

Requires: detections_raw.csv (from 01_detect_blobs.py), dff_utils.py
Run: uv run python 02_zone_groups.py [stack.tif] [output_suffix]
Outputs: zone_groups.csv, zone_traces.png
"""

import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

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

ROI_RADIUS_SCALE = 0.6  # shrink factor applied to each blob's own equivalent-circle
                         # radius -- a smaller, centered ROI has cleaner contrast
                         # than one spanning the whole (noisier-edged) blob
ROI_RADIUS_MIN = 15     # px floor, in case a blob's area is tiny

CORR_THRESHOLD = 0.5    # two blobs join the same zone if their full-length traces
                         # correlate above this. Same-flare detections (even from
                         # frames 1000 apart) reliably correlate ~0.9+; unrelated
                         # nearby blobs sit near 0 -- 0.5 sits well clear of both.

ZONE_GROUPS_CSV = f"zone_groups{SUFFIX}.csv"
ZONE_TRACES_PNG = f"zone_traces{SUFFIX}.png"


# ---------------------------------------------------------------------------
# 2. Build one small circular ROI per blob and pull its full trace
# ---------------------------------------------------------------------------

def blob_trace(dff: np.ndarray, cy: float, cx: float, radius: float) -> np.ndarray:
    """Mean dff inside a small ROI around (cy, cx), per frame -> length-1200 trace.
    Restricted to a local bounding box, not the full frame, for speed."""
    H, W = dff.shape[1], dff.shape[2]
    y0, y1 = max(0, int(cy - radius)), min(H, int(cy + radius) + 1)
    x0, x1 = max(0, int(cx - radius)), min(W, int(cx + radius) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = np.hypot(yy - cy, xx - cx) <= radius
    sub = dff[:, y0:y1, x0:x1]
    return sub[:, mask].mean(axis=1)


def all_blob_traces(dff: np.ndarray, blobs: pd.DataFrame) -> np.ndarray:
    """(n_blobs, n_frames) matrix: one trace per blob."""
    radii = np.maximum(ROI_RADIUS_MIN, ROI_RADIUS_SCALE * np.sqrt(blobs["area"] / np.pi))
    traces = np.stack([
        blob_trace(dff, row.y, row.x, r)
        for row, r in zip(blobs.itertuples(), radii)
    ])
    return traces


# ---------------------------------------------------------------------------
# 3. Cross-correlate every pair of traces, cluster into zones
# ---------------------------------------------------------------------------

def cluster_by_correlation(traces: np.ndarray, corr_threshold: float) -> np.ndarray:
    """Zero-lag Pearson correlation between every pair of blob traces,
    hierarchically clustered on (1 - correlation) as the distance."""
    corr = np.corrcoef(traces)
    distance = 1 - np.clip(corr, 0, 1)  # anti-correlated blobs are just as
                                         # "different" as uncorrelated ones
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
    traces = all_blob_traces(dff, blobs)
    zones = cluster_by_correlation(traces, CORR_THRESHOLD)

    blobs = blobs.copy()
    blobs["zone"] = zones
    blobs.to_csv(ZONE_GROUPS_CSV, index=False)
    render_zone_traces(traces, zones, ZONE_TRACES_PNG)

    print(f"{len(set(zones))} zones from {len(blobs)} blobs (corr_threshold={CORR_THRESHOLD})")
    print(f"saved {ZONE_GROUPS_CSV}, {ZONE_TRACES_PNG}")


if __name__ == "__main__":
    main()
