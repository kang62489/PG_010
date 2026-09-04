"""
Detect bright hotspots in each frame of a stack.

Pipeline:
  1. Load stack (already deltaF/F0)
  2. Find the background threshold from the stack's own histogram
  3. Threshold, clean, strip noise speckle -> save the cleaned boolean mask

Run: uv run python 01_detect_hotspots.py [stack.tif]

Outputs:
  - per-frame cleaned boolean mask, as a viewable uint8 TIFF
    (results/mask_{stack_stem}.tif)
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
from rich.console import Console
from scipy import ndimage
from scipy.optimize import curve_fit

console = Console()

# ---------------------------------------------------------------------------
# 1. Config
#    Usage: uv run python 01_detect_hotspots.py [stack.tif] [crossover_sigma]
#    Output suffix is derived from the stack's own filename stem, so it
#    always matches across 01/02/03 without a separately typed argument.
#    crossover_sigma must match the value passed to 02_zone_groups.py for
#    the same stack (it re-derives the same threshold for its own record).
# ---------------------------------------------------------------------------

STACK_PATH = sys.argv[1] if len(sys.argv) > 1 else "proc_tiffs/2025_12_15-0012_BIEXP_ALS.tif"
SUFFIX = f"_{STACK_PATH.rsplit('/', 1)[-1].rsplit('.', 1)[0]}"

HISTOGRAM_BINS = 512  # resolution/division of the range/distribution of pixel values (df/F0)
                      # (max - min) / HISTOGRAM_BINS = width of each bin in the histogram
                      # x is the center of each bin, y is the count of pixels in that bin.

CROSSOVER_RATIO = float(sys.argv[2]) if len(sys.argv) > 2 else 2  # background threshold =
                       # peak + this many fitted-Gaussian sigmas -- the wider the
                       # background's noise spread, the farther out this pushes the threshold.

TH_SMALL_OBJ = 4000  # remove small objects (px) before linking, general cleaning of noise speckle.

CLOSE_RADIUS = 3  # px: morphological closing (dilate then erode) after opening --
                   # bridges small notches/gaps in a hotspot's own per-frame shape
                   # so it fills in as one simple blob, not a jagged/split one.

MASK_TIF = f"results/mask{SUFFIX}.tif"


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
                               sigma_ratio: float = CROSSOVER_RATIO) -> float:
    """fit histogram with gaussian; threshold = peak + sigma_ratio * fitted sigma"""
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
    _, sigma = popt

    return float(peak_value + sigma_ratio * sigma)


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


def detect_all_frames(stack_f16: np.ndarray, threshold: float,
                       th_small_obj: int = TH_SMALL_OBJ, close_radius: int = CLOSE_RADIUS) -> np.ndarray:

    # convert each frame of the tiff stack into a binary mask
    mask = stack_f16 > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((1, 3, 3)))  # drop lone bright pixels
    mask = ndimage.binary_closing(mask, structure=np.ones((1, close_radius, close_radius)))  # simplify shape: close small notches/gaps
    for t in range(mask.shape[0]):
        mask[t] = ndimage.binary_fill_holes(mask[t])
    mask = remove_small_objects(mask, th_small_obj)

    return mask


# ---------------------------------------------------------------------------
# 5. Run the pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    _, stack_f16 = load_stack(STACK_PATH)
    threshold = find_background_threshold(stack_f16)
    console.print(f"[cyan]background threshold[/cyan] (peak + {CROSSOVER_RATIO} sigma): "
                  f"[bold]{threshold:.5f}[/bold]")

    mask = detect_all_frames(stack_f16, threshold)
    tifffile.imwrite(MASK_TIF, (mask.astype(np.uint8) * 255))

    console.print(f"[green]saved[/green] cleaned mask {mask.shape} -> [bold]{MASK_TIF}[/bold]")
    console.print(f"[dim]Total time: {time.time() - t_start:.1f}s[/dim]")


if __name__ == "__main__":
    main()
