"""
Detect bright-zone flare blobs, frame by frame.

This step does one job only: for every frame, find each connected bright
blob and record its centroid and size. It does NOT try to decide which
blobs across different frames belong to the same physical flare -- that
grouping happens in 02_zone_groups.py, using each blob's own temporal trace
instead of a frame-gap/distance heuristic.

Pipeline:
  1. Load stack, compute dff (see dff_utils.py)
  2. Threshold each frame and label connected bright blobs
  3. Save centroid (x, y), area, and mean intensity for every blob found

Requires: dff_utils.py
Run: uv run python 01_detect_blobs.py [stack.tif] [output_suffix]
Outputs: detections_raw.csv
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import label, regionprops

from dff_utils import load_dff

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 01_detect_blobs.py [stack.tif] [output_suffix]
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "2025_12_15-0012_BIEXP_GAUSS.tif"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""

THRESHOLD_PERCENTILE = 99.5  # global dff percentile used as the "is this pixel lit" cutoff
MIN_AREA = 3000              # drop connected blobs smaller than this (noise speckle) -
                              # all 89 speckle regions seen at frame 145 were < 761 px

RAW_DETECTIONS_CSV = f"detections_raw{SUFFIX}.csv"


# ---------------------------------------------------------------------------
# 2. Per-frame blob detection
# ---------------------------------------------------------------------------

def detect_blobs_in_frame(frame: np.ndarray, threshold: float, min_area: int) -> list[dict]:
    """Threshold one frame, denoise the mask, and return each surviving blob."""
    mask = frame > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))  # drop lone bright pixels
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))  # fill small holes

    blobs = []
    for region in regionprops(label(mask), intensity_image=frame):
        if region.area < min_area:
            continue
        blobs.append({
            "y": region.centroid[0],
            "x": region.centroid[1],
            "area": region.area,
            "mean_intensity": region.intensity_mean,
        })
    return blobs


def detect_all_frames(dff: np.ndarray, threshold: float, min_area: int) -> pd.DataFrame:
    """Run blob detection independently on every frame in the stack."""
    detections = []
    for t in range(dff.shape[0]):
        for blob in detect_blobs_in_frame(dff[t], threshold, min_area):
            detections.append({"frame": t, **blob})
    return pd.DataFrame(detections)


# ---------------------------------------------------------------------------
# 3. Run the pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    _, dff = load_dff(STACK_PATH)
    threshold = np.percentile(dff, THRESHOLD_PERCENTILE)

    det_df = detect_all_frames(dff, threshold, MIN_AREA)
    det_df.to_csv(RAW_DETECTIONS_CSV, index=False)

    print(f"{len(det_df)} raw blob detections above area {MIN_AREA} -> {RAW_DETECTIONS_CSV}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
