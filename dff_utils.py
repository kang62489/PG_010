"""Shared by every pipeline step -- load a stack and compute its dff once.

dff ("delta from baseline") = each frame minus the per-pixel median across
time. The median is robust to the flares themselves (transient, sparse in
time), so it gives a clean "resting" background per pixel.
"""

import numpy as np
import tifffile


def load_dff(stack_path: str) -> tuple[np.ndarray, np.ndarray]:
    stack = tifffile.imread(stack_path)
    baseline = np.median(stack, axis=0)
    dff = stack.astype(np.float32) - baseline
    return stack, dff
