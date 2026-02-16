"""
Annotation painting utilities for simulated corrective annotation.

Provides low-level tools to paint on RGBA annotation arrays matching
RootPainter's format: channel 0 = foreground, channel 1 = background,
channels 2-3 unused.
"""
import numpy as np


def disk(radius):
    """Return a boolean 2D array (disk structuring element) of diameter 2*radius+1."""
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]
    return (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2


def paint(annot, center, radius, channel):
    """Paint a filled circle on the given channel of an annotation array.

    Args:
        annot: (H, W, 4) uint8 array (modified in-place)
        center: (row, col) tuple
        radius: brush radius in pixels
        channel: 0 for foreground, 1 for background
    """
    row, col = center
    h, w = annot.shape[:2]
    d = disk(radius)
    r0 = row - radius
    c0 = col - radius
    d_size = 2 * radius + 1

    # clip to image bounds
    dr_start = max(0, -r0)
    dc_start = max(0, -c0)
    dr_end = min(d_size, h - r0)
    dc_end = min(d_size, w - c0)

    if dr_start >= dr_end or dc_start >= dc_end:
        return

    brush = d[dr_start:dr_end, dc_start:dc_end]
    annot[r0 + dr_start:r0 + dr_end,
          c0 + dc_start:c0 + dc_end,
          channel][brush] = 255


def new_annot(height, width):
    """Return a blank (H, W, 4) uint8 annotation array (all zeros)."""
    return np.zeros((height, width, 4), dtype=np.uint8)
