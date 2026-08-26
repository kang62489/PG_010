"""
Detect bright hotspots in each frame of a stack.

Pipeline:
  1. Load stack (already deltaF/F0)
  2. Find the background threshold from the stack's own histogram
  3. Threshold, clean, strip noise speckle, then bridge + label in 3D
  4. Save every detection's frame, joint_label, centroid, area,
     and exact pixel footprint

Run: uv run python 01_detect_hotspots.py [stack.tif] [output_suffix]

Outputs:
  - centroid, area (detections_raw.csv)
  - pixel coordinates of each hotspot (hotspot_footprints.npz)
"""

## Modules
# Standard
import sys
import time
from functools import partial

# Third-party
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 01_detect_hotspots.py [stack.tif] [output_suffix]
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "2025_12_15-0012_BIEXP_GAUSS.tif"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""

HISTOGRAM_BINS = 512  # resolution/division of the range/distribution of pixel values (df/F0)
                      # (max - min) / HISTOGRAM_BINS = width of each bin in the histogram
                      # x is the center of each bin, y is the count of pixels in that bin.

CROSSOVER_RATIO = 3.0  # the ratio used to determine if the current x (right side of the bell curve) 
                       # is enough to be the right edge (thresold) of the background.

TH_SMALL_HOTSPOTS = 3000  # remove hotspots which is too small (px)

TH_SMALL_OBJ = 800  # remove small objects (px) before linking, general cleaning of noise speckle.

CONNECT_RADIUS = 10  # px: bridge gaps up to this wide between fragments before
                      # labeling, so one real hotspot broken apart by a noisy
                      # dip is still detected as a single hotspot, not several

TEMPORAL_GAP = 1  # frames: also bridge across this many frames before/after,
                   # so two same-frame fragments that are really one hotspot
                   # (confirmed by both touching a single hotspot in a
                   # neighboring frame) get merged instead of counted twice

RAW_DETECTIONS_CSV = f"detections_raw{SUFFIX}.csv"
HOTSPOT_FOOTPRINTS_NPZ = f"hotspot_footprints{SUFFIX}.npz"


# ---------------------------------------------------------------------------
# 2. Load the stack (already deltaF/F0 -- no baseline subtraction here)
# ---------------------------------------------------------------------------

def load_stack(stack_path: str) -> tuple[np.ndarray, np.ndarray]:
    stack = tifffile.imread(stack_path)
    stack_f16 = stack.astype(np.float16)
    return stack, stack_f16


# ---------------------------------------------------------------------------
# 3. Find the background threshold from the stack's own histogram
# ---------------------------------------------------------------------------

def gaussian(x: np.ndarray, amplitude: float, sigma: float, peak_value: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - peak_value) / sigma) ** 2)


def find_background_threshold(stack_f16: np.ndarray, bins: int = HISTOGRAM_BINS,
                               crossover_ratio: float = CROSSOVER_RATIO) -> float:
    """fit histogram with gaussian; sweep x and then compare the ratio actual amplitude to fitted amplitude (> crossover_ratio)"""
    # generate histogram
    values = stack_f16.ravel().astype(np.float32)
    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peak_idx = np.argmax(hist)
    peak_value = bin_centers[peak_idx]
    peak_height = hist[peak_idx]

    # fit left side (because of sigma, mean, amplitude. our bright pixels are at the right side of the peak.)
    left_mask = bin_centers <= peak_value
    popt, _ = curve_fit(partial(gaussian, peak_value=peak_value),
                         bin_centers[left_mask], hist[left_mask],
                         p0=[peak_height, 0.0016])
    fit_y = gaussian(bin_centers, *popt, peak_value=peak_value)

    for i in range(peak_idx, len(hist) - 3):
        if all(hist[i + k] > fit_y[i + k] * crossover_ratio for k in range(3)):
            return float(bin_centers[i])

    raise RuntimeError("no crossover found -- histogram never exceeds the fitted "
                        "background curve by the given ratio")


# ---------------------------------------------------------------------------
# 4. Per-frame hotspot detection
# ---------------------------------------------------------------------------

def remove_small_objects(mask: np.ndarray, th_small_obj: int) -> np.ndarray:
    """clean up small noise speckles in each frame."""
    cleaned = np.zeros_like(mask)
    for t in range(mask.shape[0]):
        labeled, n = ndimage.label(mask[t])
        if n == 0:
            continue
        sizes = ndimage.sum(mask[t], labeled, index=np.arange(1, n + 1))
        keep = np.nonzero(sizes >= th_small_obj)[0] + 1
        cleaned[t] = np.isin(labeled, keep)
    return cleaned


def detect_all_frames(stack_f16: np.ndarray, threshold: float, th_small_hotspots: int,
                       th_small_obj: int = TH_SMALL_OBJ,
                       connect_radius: int = CONNECT_RADIUS,
                       temporal_gap: int = TEMPORAL_GAP) -> tuple[pd.DataFrame, list]:
    
    # convert each frame of the tiff stack into a binary mask
    mask = stack_f16 > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((1, 3, 3)))  # drop lone bright pixels
    for t in range(mask.shape[0]):
        mask[t] = ndimage.binary_fill_holes(mask[t])
    mask = remove_small_objects(mask, th_small_obj)

    # use dilation to check if two adjacent hotspots are actually one hotspot, if so, connect them (using OR operation).
    bridged = ndimage.binary_dilation(mask, structure=np.ones((1, connect_radius, connect_radius)))
    # using a 3D array, dilation with a 2D structure array stacked across 3 frames reaches the frame before and after the current frame (determined by temporal_gap = 1).
    bridged = ndimage.binary_dilation(bridged, structure=np.ones((1 + 2 * temporal_gap, 1, 1)))

    # label for several measurements with regionprops
    labeled, _ = ndimage.label(bridged, structure=np.ones((3, 3, 3)), output=np.int32)
    del bridged

    detections = []
    footprints = []
    for t in range(mask.shape[0]):
        frame_mask = mask[t]
        frame_labels = labeled[t]
        for region_label in np.unique(frame_labels[frame_mask]):
            footprint_mask = frame_mask & (frame_labels == region_label)
            area = int(footprint_mask.sum())
            if area < th_small_hotspots:
                continue
            coords = np.argwhere(footprint_mask)
            footprints.append(coords)
            detections.append({
                "frame": t,
                "joint_label": int(region_label),
                "y": float(coords[:, 0].mean()),
                "x": float(coords[:, 1].mean()),
                "area": area,
            })
    return pd.DataFrame(detections), footprints


# ---------------------------------------------------------------------------
# 5. Run the pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    _, stack_f16 = load_stack(STACK_PATH)
    threshold = find_background_threshold(stack_f16)
    print(f"background threshold (Gaussian-fit crossover, ratio={CROSSOVER_RATIO}): {threshold:.5f}")

    det_df, footprints = detect_all_frames(stack_f16, threshold, TH_SMALL_HOTSPOTS)
    det_df.to_csv(RAW_DETECTIONS_CSV, index=False)
    np.savez(HOTSPOT_FOOTPRINTS_NPZ, **{f"hotspot_{i}": coords for i, coords in enumerate(footprints)})

    print(f"{len(det_df)} raw hotspot detections above area {TH_SMALL_HOTSPOTS} -> {RAW_DETECTIONS_CSV}, {HOTSPOT_FOOTPRINTS_NPZ}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
