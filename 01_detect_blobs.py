"""
Detect bright hotspots and link them across space AND time.

Finds each connected bright hotspot per frame (centroid, area, mean
intensity, exact pixel footprint). Detections that are the same physical
hotspot across nearby/consecutive frames are linked via 3D (time, y, x)
connectivity and share a track_id -- see find_background_threshold() and
detect_all_frames() for how. Deciding if different tracks are the same
*recurring* zone is left to 02_zone_groups.py.

Pipeline:
  1. Load stack (already deltaF/F0 -- no baseline subtraction here)
  2. Find the background threshold from the stack's own histogram
  3. Threshold, clean, strip noise speckle, then bridge + label in 3D
  4. Save every detection's frame, track_id, centroid, area, intensity,
     and exact pixel footprint

Run: uv run python 01_detect_blobs.py [stack.tif] [output_suffix]
Outputs: detections_raw.csv, hotspot_footprints.npz
"""

import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 01_detect_blobs.py [stack.tif] [output_suffix]
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "2025_12_15-0012_BIEXP_GAUSS.tif"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""

HISTOGRAM_BINS = 512  # resolution used to find the background peak below

CROSSOVER_RATIO = 3.0  # threshold = point where real pixel counts exceed the
                        # fitted background Gaussian by this factor, sustained
                        # for 3 consecutive bins

MIN_AREA = 3000  # drop connected hotspots smaller than this (noise speckle) -
                  # all 89 speckle regions seen at frame 145 were < 761 px

NOISE_FLOOR = 800  # px: noise speckle is well under this (89 speckle regions
                    # seen at frame 145 were all < 761px) -- dropped BEFORE
                    # any bridging, so it can never act as a stepping stone
                    # that chains unrelated real hotspots together across
                    # frames. MIN_AREA is the real, higher bar applied after
                    # merging; this is just noise triage.

CONNECT_RADIUS = 10  # px: bridge gaps up to this wide between fragments before
                      # labeling, so one real hotspot broken apart by a noisy
                      # dip is still detected as a single hotspot, not several

TEMPORAL_GAP = 1  # frames: also bridge across this many frames before/after,
                   # so two same-frame fragments that are really one hotspot
                   # (confirmed by both touching a single blob in a
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
    """Background pixel values across the whole stack pile up into one tall
    peak. Fit a Gaussian to just the LEFT half of that peak (real hotspot
    pixels can only add excess counts on the bright/right side, so the left
    side is clean background) -- then walk outward from the peak and find
    where the real histogram counts exceed the fitted curve by
    crossover_ratio, sustained for 3 consecutive bins. Past that point,
    pixels are no longer explained by background noise alone."""
    values = stack_f16.ravel().astype(np.float32)
    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peak_idx = np.argmax(hist)
    peak_value = bin_centers[peak_idx]
    peak_height = hist[peak_idx]

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

def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components under min_size, labeling each frame
    independently (no time connectivity) -- used to strip noise speckle
    BEFORE any bridging, so it can never act as a stepping stone that chains
    unrelated real hotspots together across frames."""
    cleaned = np.zeros_like(mask)
    for t in range(mask.shape[0]):
        labeled, n = ndimage.label(mask[t])
        if n == 0:
            continue
        sizes = ndimage.sum(mask[t], labeled, index=np.arange(1, n + 1))
        keep = np.nonzero(sizes >= min_size)[0] + 1
        cleaned[t] = np.isin(labeled, keep)
    return cleaned


def detect_all_frames(stack_f16: np.ndarray, threshold: float, min_area: int,
                       noise_floor: int = NOISE_FLOOR,
                       connect_radius: int = CONNECT_RADIUS,
                       temporal_gap: int = TEMPORAL_GAP) -> tuple[pd.DataFrame, list]:
    """Threshold the whole stack at once and label it in 3D (time, y, x), so
    that a single physical hotspot never gets miscounted as several separate
    ones just because it pinches off briefly within one frame.

    Per-frame cleanup: opening drops lone bright speckle, then each frame's
    holes are fully filled (not just small ones) -- both applied to every
    frame at once via a z-extent-1 structuring element, which is equivalent
    to looping per frame but vectorized. Background noise is persistent
    across essentially every frame at a low level, so before any bridging,
    remove_small_components() strips clearly-noise fragments under
    noise_floor -- without this, noise below MIN_AREA still gets pulled into
    the bridging step below and can chain real, unrelated hotspots across
    the whole video into one bogus giant component.

    3D connectivity: a copy of the cleaned mask is dilated by connect_radius
    in-plane (bridges nearby fragments within one frame, as before) AND by
    temporal_gap frames before/after (bridges a hotspot that briefly drops
    below threshold across a frame or two). Labeling this bridged volume in
    3D means two same-frame fragments that both touch one blob in a
    neighboring frame -- direct evidence they're the same physical hotspot,
    per the "check the frame before/after" idea -- are merged into a single
    3D-connected component. Each hotspot's saved footprint still only
    contains its own frame's real (non-dilated) pixels, so area/shape stay
    accurate; the 3D label is also saved as track_id so later steps know
    which frame-appearances are already confirmed to be the same hotspot."""
    mask = stack_f16 > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((1, 3, 3)))  # drop lone bright pixels
    for t in range(mask.shape[0]):
        mask[t] = ndimage.binary_fill_holes(mask[t])
    mask = remove_small_components(mask, noise_floor)

    bridged = ndimage.binary_dilation(mask, structure=np.ones((1, connect_radius, connect_radius)))
    bridged = ndimage.binary_dilation(bridged, structure=np.ones((1 + 2 * temporal_gap, 1, 1)))

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
            if area < min_area:
                continue
            coords = np.argwhere(footprint_mask)
            footprints.append(coords)
            detections.append({
                "frame": t,
                "track_id": int(region_label),
                "y": float(coords[:, 0].mean()),
                "x": float(coords[:, 1].mean()),
                "area": area,
                "mean_intensity": float(stack_f16[t][footprint_mask].mean()),
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

    det_df, footprints = detect_all_frames(stack_f16, threshold, MIN_AREA)
    det_df.to_csv(RAW_DETECTIONS_CSV, index=False)
    np.savez(HOTSPOT_FOOTPRINTS_NPZ, **{f"hotspot_{i}": coords for i, coords in enumerate(footprints)})

    print(f"{len(det_df)} raw hotspot detections above area {MIN_AREA} -> {RAW_DETECTIONS_CSV}, {HOTSPOT_FOOTPRINTS_NPZ}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
